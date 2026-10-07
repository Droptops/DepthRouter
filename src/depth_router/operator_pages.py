from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


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
