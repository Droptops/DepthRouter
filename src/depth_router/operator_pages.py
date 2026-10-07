from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class InputPageSketch:
    page_id: int
    weight: Tensor
    bias: Tensor | None

    @property
    def metadata_bytes(self) -> int:
        total = self.weight.numel() * self.weight.element_size()
        if self.bias is not None:
            total += self.bias.numel() * self.bias.element_size()
        return int(total)


@dataclass(frozen=True)
class OperatorPage:
    page_id: int
    start: int
    end: int
    weight_bytes: int

    @property
    def width(self) -> int:
        return self.end - self.start


def contiguous_mlp_pages(
    linear1: nn.Linear,
    linear2: nn.Linear,
    *,
    units_per_page: int,
) -> list[OperatorPage]:
    """Partition a dense two-layer MLP by intermediate neurons.

    A page owns rows of linear1 and the matching columns of linear2. Bias from
    linear2 is treated as resident/shared because it is not attributable to one
    intermediate neuron.
    """

    if units_per_page < 1:
        raise ValueError("units_per_page must be >= 1")
    if linear1.out_features != linear2.in_features:
        raise ValueError("linear layers do not share an intermediate width")

    width = linear1.out_features
    pages: list[OperatorPage] = []
    for page_id, start in enumerate(range(0, width, units_per_page)):
        end = min(start + units_per_page, width)

        bytes_1 = linear1.weight[start:end].numel() * linear1.weight.element_size()
        if linear1.bias is not None:
            bytes_1 += linear1.bias[start:end].numel() * linear1.bias.element_size()

        bytes_2 = linear2.weight[:, start:end].numel() * linear2.weight.element_size()
        pages.append(
            OperatorPage(
                page_id=page_id,
                start=start,
                end=end,
                weight_bytes=int(bytes_1 + bytes_2),
            )
        )

    return pages



def page_output_weight_norms(
    linear2: nn.Linear,
    pages: list[OperatorPage],
) -> Tensor:
    """Precompute one resident scalar norm per cold output-weight page."""

    if not pages:
        raise ValueError("at least one page is required")
    norms = []
    for page in pages:
        if page.start < 0 or page.end > linear2.in_features or page.start >= page.end:
            raise ValueError("invalid page range")
        weight = linear2.weight[:, page.start : page.end]
        norms.append(weight.detach().float().norm())
    return torch.stack(norms)


def activation_address_scores(
    activation: Tensor,
    pages: list[OperatorPage],
    output_weight_norms: Tensor,
) -> Tensor:
    """Turn MLP activation energy into a zero-parameter page-address distribution.

    The score for page j is ||a_j||_2 * ||W2_j||_F, an upper-bound-style proxy
    for the magnitude of that page's output contribution. W2 norms are tiny
    resident metadata precomputed once; the cold matrix itself is not read here.
    """

    if not pages:
        raise ValueError("at least one page is required")
    if output_weight_norms.ndim != 1 or output_weight_norms.numel() != len(pages):
        raise ValueError("one output weight norm is required per page")

    scores = []
    for idx, page in enumerate(pages):
        if page.start < 0 or page.end > activation.shape[-1] or page.start >= page.end:
            raise ValueError("invalid page range")
        act_norm = activation[..., page.start : page.end].float().norm(dim=-1)
        scores.append(act_norm * output_weight_norms[idx].to(act_norm.device))

    stacked = torch.stack(scores, dim=-1)
    return stacked / stacked.sum(dim=-1, keepdim=True).clamp_min(1e-12)



def make_input_page_sketches(
    linear1: nn.Linear,
    pages: list[OperatorPage],
    *,
    sketch_dim: int,
    seed: int = 0,
) -> list[InputPageSketch]:
    """Compress each cold input-projection page into resident JL metadata.

    A page owns rows of linear1. For a random sign matrix R with shape
    [sketch_dim, page_width], store R W1_j and R b1_j. At runtime the resident
    sketch estimates the page's preactivation energy from the MLP input without
    reading the cold W1_j payload itself.
    """

    if sketch_dim < 1:
        raise ValueError("sketch_dim must be >= 1")
    if not pages:
        raise ValueError("at least one page is required")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    sketches: list[InputPageSketch] = []

    for page in pages:
        if page.start < 0 or page.end > linear1.out_features or page.start >= page.end:
            raise ValueError("invalid page range")
        if sketch_dim > page.width:
            raise ValueError("sketch_dim cannot exceed page width")

        signs = torch.randint(
            0,
            2,
            (sketch_dim, page.width),
            generator=generator,
            dtype=torch.int64,
        ).float()
        projection = (signs.mul_(2).sub_(1)) / (float(sketch_dim) ** 0.5)
        projection = projection.to(
            device=linear1.weight.device,
            dtype=linear1.weight.dtype,
        )

        weight = linear1.weight[page.start : page.end]
        compressed_weight = (projection @ weight).detach()
        compressed_bias = None
        if linear1.bias is not None:
            compressed_bias = (
                projection @ linear1.bias[page.start : page.end]
            ).detach()

        sketches.append(
            InputPageSketch(
                page_id=page.page_id,
                weight=compressed_weight,
                bias=compressed_bias,
            )
        )

    return sketches


def input_sketch_address_scores(
    mlp_input: Tensor,
    pages: list[OperatorPage],
    sketches: list[InputPageSketch],
    output_weight_norms: Tensor,
) -> Tensor:
    """Estimate cold-page value using only resident metadata and MLP input.

    The score is estimated preactivation energy times the resident Frobenius
    norm of the matching W2 page. This does not read either cold W1 or W2.
    """

    if len(sketches) != len(pages):
        raise ValueError("one input sketch is required per page")
    if output_weight_norms.ndim != 1 or output_weight_norms.numel() != len(pages):
        raise ValueError("one output weight norm is required per page")

    scores = []
    for idx, (page, sketch) in enumerate(zip(pages, sketches, strict=True)):
        if sketch.page_id != page.page_id:
            raise ValueError("sketch page id does not match layout")
        if sketch.weight.ndim != 2 or sketch.weight.shape[1] != mlp_input.shape[-1]:
            raise ValueError("sketch input width does not match MLP input")

        estimate = torch.matmul(
            mlp_input.to(sketch.weight.dtype),
            sketch.weight.T,
        )
        if sketch.bias is not None:
            estimate = estimate + sketch.bias
        energy = estimate.float().norm(dim=-1)
        scores.append(
            energy * output_weight_norms[idx].to(energy.device)
        )

    stacked = torch.stack(scores, dim=-1)
    return stacked / stacked.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def input_sketch_metadata_bytes(sketches: list[InputPageSketch]) -> int:
    if not sketches:
        raise ValueError("at least one sketch is required")
    return sum(sketch.metadata_bytes for sketch in sketches)

def make_output_page_sketches(
    linear2: nn.Linear,
    pages: list[OperatorPage],
    *,
    sketch_dim: int,
    seed: int = 0,
) -> list[Tensor]:
    """Precompute tiny random projections of cold output-weight pages.

    For page j with payload W_j, store R W_j where R is a shared random sign
    projection. Runtime scoring can estimate ||W_j a_j|| from this metadata
    without reading W_j itself.
    """

    if sketch_dim < 1:
        raise ValueError("sketch_dim must be >= 1")
    if sketch_dim > linear2.out_features:
        raise ValueError("sketch_dim cannot exceed output width")
    if not pages:
        raise ValueError("at least one page is required")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    signs = torch.randint(
        0,
        2,
        (sketch_dim, linear2.out_features),
        generator=generator,
        dtype=torch.int64,
    ).float()
    projection = (signs.mul_(2).sub_(1)) / (float(sketch_dim) ** 0.5)
    projection = projection.to(
        device=linear2.weight.device,
        dtype=linear2.weight.dtype,
    )

    sketches: list[Tensor] = []
    for page in pages:
        if page.start < 0 or page.end > linear2.in_features or page.start >= page.end:
            raise ValueError("invalid page range")
        payload = linear2.weight[:, page.start : page.end]
        sketches.append((projection @ payload).detach())
    return sketches


def sketch_address_scores(
    activation: Tensor,
    pages: list[OperatorPage],
    sketches: list[Tensor],
) -> Tensor:
    """Estimate page contribution magnitude from resident random sketches."""

    if len(sketches) != len(pages):
        raise ValueError("one sketch is required per page")
    if not pages:
        raise ValueError("at least one page is required")

    scores = []
    for page, sketch in zip(pages, sketches, strict=True):
        if page.start < 0 or page.end > activation.shape[-1] or page.start >= page.end:
            raise ValueError("invalid page range")
        if sketch.ndim != 2 or sketch.shape[1] != page.width:
            raise ValueError("sketch width does not match page")
        x = activation[..., page.start : page.end].to(sketch.dtype)
        estimate = torch.matmul(x, sketch.T).float().norm(dim=-1)
        scores.append(estimate)

    stacked = torch.stack(scores, dim=-1)
    return stacked / stacked.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def output_page_payload_bytes(
    linear2: nn.Linear,
    pages: list[OperatorPage],
) -> Tensor:
    """Bytes in each cold linear2 payload, excluding resident address metadata."""

    if not pages:
        raise ValueError("at least one page is required")
    return torch.tensor(
        [
            linear2.weight[:, page.start : page.end].numel()
            * linear2.weight.element_size()
            for page in pages
        ],
        dtype=torch.long,
        device=linear2.weight.device,
    )

def linear2_page_contributions(
    activation: Tensor,
    linear2: nn.Linear,
    pages: list[OperatorPage],
) -> Tensor:
    """Return additive output contribution from each intermediate-neuron page.

    Shape is activation.shape[:-1] + (number_of_pages, linear2.out_features).
    """

    if activation.shape[-1] != linear2.in_features:
        raise ValueError("activation width does not match linear2")
    if not pages:
        raise ValueError("at least one page is required")

    contributions = []
    for page in pages:
        if page.start < 0 or page.end > linear2.in_features or page.start >= page.end:
            raise ValueError("invalid page range")
        x = activation[..., page.start : page.end]
        weight = linear2.weight[:, page.start : page.end]
        contributions.append(torch.matmul(x, weight.T))

    return torch.stack(contributions, dim=-2)


def reconstruct_linear2(
    page_contributions: Tensor,
    linear2: nn.Linear,
    *,
    page_mask: Tensor | None = None,
) -> Tensor:
    """Reconstruct dense or selected-page linear2 output."""

    if page_contributions.shape[-1] != linear2.out_features:
        raise ValueError("contribution output width does not match linear2")

    if page_mask is None:
        selected = page_contributions
    else:
        if page_mask.shape != page_contributions.shape[:-1]:
            raise ValueError(
                "page_mask must match contribution shape except output width"
            )
        selected = page_contributions * page_mask[..., None].to(
            dtype=page_contributions.dtype
        )

    output = selected.sum(dim=-2)
    if linear2.bias is not None:
        output = output + linear2.bias
    return output


def page_bytes_tensor(
    pages: list[OperatorPage],
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    if not pages:
        raise ValueError("at least one page is required")
    return torch.tensor(
        [page.weight_bytes for page in pages],
        dtype=torch.long,
        device=device,
    )
