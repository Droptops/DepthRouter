from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .causal_memory import greedy_affinity_layout, posterior_affinity


@dataclass(frozen=True)
class SwiGLUPage:
    page_id: int
    start: int
    end: int
    payload_bytes: int

    @property
    def width(self) -> int:
        return self.end - self.start


def swiglu_intermediate(
    hidden: Tensor,
    gate_proj: nn.Linear,
    up_proj: nn.Linear,
) -> Tensor:
    if gate_proj.in_features != up_proj.in_features:
        raise ValueError("gate_proj and up_proj must share input width")
    if gate_proj.out_features != up_proj.out_features:
        raise ValueError("gate_proj and up_proj must share intermediate width")
    return torch.nn.functional.silu(gate_proj(hidden)) * up_proj(hidden)


def swiglu_output(
    hidden: Tensor,
    gate_proj: nn.Linear,
    up_proj: nn.Linear,
    down_proj: nn.Linear,
) -> Tensor:
    intermediate = swiglu_intermediate(hidden, gate_proj, up_proj)
    if down_proj.in_features != intermediate.shape[-1]:
        raise ValueError("down_proj input width does not match intermediate width")
    return down_proj(intermediate)


def contiguous_swiglu_pages(
    gate_proj: nn.Linear,
    up_proj: nn.Linear,
    down_proj: nn.Linear,
    *,
    units_per_page: int,
) -> list[SwiGLUPage]:
    if units_per_page < 1:
        raise ValueError("units_per_page must be >= 1")
    if gate_proj.out_features != up_proj.out_features:
        raise ValueError("gate_proj and up_proj must share intermediate width")
    if down_proj.in_features != gate_proj.out_features:
        raise ValueError("down_proj input width must match intermediate width")

    pages: list[SwiGLUPage] = []
    width = gate_proj.out_features
    element_size = gate_proj.weight.element_size()

    for page_id, start in enumerate(range(0, width, units_per_page)):
        end = min(start + units_per_page, width)
        payload = (
            gate_proj.weight[start:end].numel()
            + up_proj.weight[start:end].numel()
            + down_proj.weight[:, start:end].numel()
        ) * element_size

        if gate_proj.bias is not None:
            payload += gate_proj.bias[start:end].numel() * gate_proj.bias.element_size()
        if up_proj.bias is not None:
            payload += up_proj.bias[start:end].numel() * up_proj.bias.element_size()

        pages.append(
            SwiGLUPage(
                page_id=page_id,
                start=start,
                end=end,
                payload_bytes=int(payload),
            )
        )

    return pages


def swiglu_page_contributions(
    intermediate: Tensor,
    down_proj: nn.Linear,
    pages: list[SwiGLUPage],
) -> Tensor:
    if intermediate.shape[-1] != down_proj.in_features:
        raise ValueError("intermediate width does not match down_proj")
    if not pages:
        raise ValueError("at least one page is required")

    contributions = []
    for page in pages:
        x = intermediate[..., page.start : page.end]
        weight = down_proj.weight[:, page.start : page.end]
        contributions.append(torch.matmul(x, weight.T))

    return torch.stack(contributions, dim=-2)


def learn_codemand_permutation(
    intermediate: Tensor,
    down_proj: nn.Linear,
    *,
    units_per_page: int,
) -> Tensor:
    """Learn a function-preserving neuron order from co-demand traces.

    The demand proxy is |activation_i| * ||W_down[:, i]||. Neurons that are
    simultaneously important receive high co-posterior affinity and are packed
    into the same physical page.
    """

    if intermediate.ndim < 2:
        raise ValueError("intermediate must contain examples and neurons")
    if intermediate.shape[-1] != down_proj.in_features:
        raise ValueError("intermediate width does not match down_proj")
    if units_per_page < 1:
        raise ValueError("units_per_page must be >= 1")

    flat = intermediate.reshape(-1, intermediate.shape[-1]).float()
    column_norm = down_proj.weight.detach().float().norm(dim=0).to(flat.device)
    importance = flat.abs() * column_norm[None, :]
    importance = importance / importance.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    layout = greedy_affinity_layout(
        posterior_affinity(importance.cpu()),
        states_per_page=units_per_page,
        page_bytes=1,
    )
    return torch.argsort(layout.state_to_page)


def learn_simhash_codemand_permutation(
    intermediate: Tensor,
    down_proj: nn.Linear,
    *,
    bits: int = 16,
    max_trace_tokens: int = 2048,
    seed: int = 0,
) -> Tensor:
    """Scalable locality-sensitive ordering of neuron demand signatures.

    The exact affinity compiler materializes an N x N neuron affinity matrix.
    This alternative gives each neuron a SimHash code over its demand trajectory
    and sorts by that code. Memory is O(TN + Nb) instead of O(N^2).
    """

    if not 1 <= bits <= 62:
        raise ValueError("bits must be between 1 and 62")
    if max_trace_tokens < 1:
        raise ValueError("max_trace_tokens must be >= 1")
    if intermediate.ndim < 2:
        raise ValueError("intermediate must contain examples and neurons")
    if intermediate.shape[-1] != down_proj.in_features:
        raise ValueError("intermediate width does not match down_proj")

    flat = intermediate.reshape(-1, intermediate.shape[-1]).float().cpu()
    if flat.shape[0] > max_trace_tokens:
        generator = torch.Generator().manual_seed(seed)
        selection = torch.randperm(flat.shape[0], generator=generator)[:max_trace_tokens]
        flat = flat.index_select(0, selection)

    column_norm = down_proj.weight.detach().float().norm(dim=0).cpu()
    demand = flat.abs() * column_norm[None, :]
    signatures = demand.T
    signatures = signatures / signatures.norm(dim=1, keepdim=True).clamp_min(1e-12)

    generator = torch.Generator().manual_seed(seed + 1)
    hyperplanes = torch.randn(
        signatures.shape[1],
        bits,
        generator=generator,
        dtype=signatures.dtype,
    )
    binary = (signatures @ hyperplanes) >= 0

    powers = (1 << torch.arange(bits, dtype=torch.int64))[None, :]
    codes = (binary.to(torch.int64) * powers).sum(dim=-1)

    # Stable secondary key keeps equal hashes deterministic.
    indices = torch.arange(codes.numel(), dtype=torch.int64)
    composite = codes * (codes.numel() + 1) + indices
    return torch.argsort(composite)


@torch.no_grad()
def permute_swiglu_neurons_(
    gate_proj: nn.Linear,
    up_proj: nn.Linear,
    down_proj: nn.Linear,
    permutation: Tensor,
) -> None:
    """Apply an exact SwiGLU hidden-neuron permutation in place.

    Rows of gate/up projections and matching columns of down_proj are permuted
    together. This changes physical layout but not the dense function.
    """

    width = gate_proj.out_features
    if up_proj.out_features != width or down_proj.in_features != width:
        raise ValueError("SwiGLU projections do not share intermediate width")
    if permutation.ndim != 1 or permutation.numel() != width:
        raise ValueError("permutation must have one entry per intermediate neuron")

    permutation = permutation.to(dtype=torch.long)
    expected = torch.arange(width, dtype=torch.long)
    if not torch.equal(torch.sort(permutation.cpu()).values, expected):
        raise ValueError("permutation must contain every neuron exactly once")

    gate_order = permutation.to(gate_proj.weight.device)
    up_order = permutation.to(up_proj.weight.device)
    down_order = permutation.to(down_proj.weight.device)

    gate_proj.weight.copy_(gate_proj.weight.index_select(0, gate_order))
    up_proj.weight.copy_(up_proj.weight.index_select(0, up_order))
    down_proj.weight.copy_(down_proj.weight.index_select(1, down_order))

    if gate_proj.bias is not None:
        gate_proj.bias.copy_(gate_proj.bias.index_select(0, gate_order))
    if up_proj.bias is not None:
        up_proj.bias.copy_(up_proj.bias.index_select(0, up_order))


def page_importance(
    intermediate: Tensor,
    down_proj: nn.Linear,
    pages: list[SwiGLUPage],
) -> Tensor:
    """Page-level contribution proxy used only for structural locality metrics."""

    if intermediate.shape[-1] != down_proj.in_features:
        raise ValueError("intermediate width does not match down_proj")
    column_norm = down_proj.weight.detach().float().norm(dim=0).to(intermediate.device)
    neuron = intermediate.float().abs() * column_norm
    page_scores = [
        neuron[..., page.start : page.end].sum(dim=-1)
        for page in pages
    ]
    scores = torch.stack(page_scores, dim=-1)
    return scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def pages_for_mass(page_scores: Tensor, *, mass_target: float) -> Tensor:
    if not 0 < mass_target <= 1:
        raise ValueError("mass_target must be in (0, 1]")
    if page_scores.ndim < 2:
        raise ValueError("page_scores must have a page dimension")
    sorted_scores = page_scores.sort(dim=-1, descending=True).values
    cumulative = sorted_scores.cumsum(dim=-1)
    before = (cumulative < mass_target).sum(dim=-1)
    return (before + 1).clamp_max(page_scores.shape[-1])
