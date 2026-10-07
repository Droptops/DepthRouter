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



def learn_balanced_codemand_permutation(
    intermediate: Tensor,
    down_proj: nn.Linear,
    *,
    units_per_page: int,
    iterations: int = 6,
    max_trace_tokens: int = 2048,
    seed: int = 0,
) -> Tensor:
    """Balanced spherical clustering of neuron demand trajectories.

    Co-demand affinity is a Gram matrix of per-neuron demand signatures. Rather
    than materializing the O(N^2) Gram matrix, cluster those signatures directly.
    The resulting pages have bounded capacity and approximate the same objective
    with O(TN + NP) working memory, where P is the number of pages.
    """

    if units_per_page < 1:
        raise ValueError("units_per_page must be >= 1")
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if max_trace_tokens < 1:
        raise ValueError("max_trace_tokens must be >= 1")
    if intermediate.ndim < 2:
        raise ValueError("intermediate must contain examples and neurons")
    if intermediate.shape[-1] != down_proj.in_features:
        raise ValueError("intermediate width does not match down_proj")

    flat = intermediate.reshape(-1, intermediate.shape[-1]).float().cpu()
    generator = torch.Generator().manual_seed(seed)
    if flat.shape[0] > max_trace_tokens:
        selection = torch.randperm(flat.shape[0], generator=generator)[:max_trace_tokens]
        flat = flat.index_select(0, selection)

    column_norm = down_proj.weight.detach().float().norm(dim=0).cpu()
    demand = flat.abs() * column_norm[None, :]
    signatures = demand.T
    signatures = signatures / signatures.norm(dim=1, keepdim=True).clamp_min(1e-12)

    num_neurons = signatures.shape[0]
    num_pages = (num_neurons + units_per_page - 1) // units_per_page
    seed_order = torch.randperm(num_neurons, generator=generator)
    centroids = signatures.index_select(0, seed_order[:num_pages]).clone()
    centroids = centroids / centroids.norm(dim=1, keepdim=True).clamp_min(1e-12)

    assignment = torch.full((num_neurons,), -1, dtype=torch.long)
    capacities = torch.full((num_pages,), units_per_page, dtype=torch.long)
    final_capacity = num_neurons - units_per_page * (num_pages - 1)
    capacities[-1] = final_capacity

    for _ in range(iterations):
        similarity = signatures @ centroids.T
        confidence = similarity.max(dim=1).values
        neuron_order = torch.argsort(confidence, descending=True)

        assignment.fill_(-1)
        remaining = capacities.clone()
        preferences = torch.argsort(similarity, dim=1, descending=True)

        for neuron in neuron_order.tolist():
            for page in preferences[neuron].tolist():
                if int(remaining[page]) > 0:
                    assignment[neuron] = page
                    remaining[page] -= 1
                    break

        if bool((assignment < 0).any()):
            raise RuntimeError("balanced assignment left neurons unassigned")

        updated = []
        for page in range(num_pages):
            members = signatures[assignment == page]
            if members.numel() == 0:
                updated.append(centroids[page])
            else:
                center = members.mean(dim=0)
                center = center / center.norm().clamp_min(1e-12)
                updated.append(center)
        centroids = torch.stack(updated)

    similarity = signatures @ centroids.T
    own_similarity = similarity[
        torch.arange(num_neurons),
        assignment,
    ]
    # Sorting by page gives contiguous physical pages. Within each page, place
    # the most centroid-aligned neurons first for deterministic partial pages.
    composite = assignment.to(torch.float64) * 2.0 - own_similarity.to(torch.float64)
    return torch.argsort(composite)

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



@dataclass(frozen=True)
class PagedSwiGLUStats:
    requested_page_uses: int
    unique_pages_loaded: int
    selected_payload_bytes: int


def paged_swiglu_reference(
    hidden: Tensor,
    gate_proj: nn.Linear,
    up_proj: nn.Linear,
    down_proj: nn.Linear,
    pages: list[SwiGLUPage],
    page_mask: Tensor,
) -> tuple[Tensor, PagedSwiGLUStats]:
    """CPU/PyTorch semantic reference for page-selective SwiGLU execution.

    This groups work by physical page. Each selected page is conceptually loaded
    once for every batch window and serves all rows requesting it. The function
    is a correctness model for a future fused GPU kernel; PyTorch slicing itself
    is not evidence of reduced HBM traffic.
    """

    if hidden.ndim != 2:
        raise ValueError("hidden must have shape [rows, d_model]")
    if page_mask.shape != (hidden.shape[0], len(pages)):
        raise ValueError("page_mask must have shape [rows, pages]")
    if gate_proj.in_features != hidden.shape[-1]:
        raise ValueError("hidden width does not match gate projection")
    if gate_proj.out_features != up_proj.out_features:
        raise ValueError("gate/up intermediate widths do not match")
    if down_proj.in_features != gate_proj.out_features:
        raise ValueError("down projection width does not match SwiGLU width")

    output = torch.zeros(
        hidden.shape[0],
        down_proj.out_features,
        device=hidden.device,
        dtype=hidden.dtype,
    )
    if down_proj.bias is not None:
        output = output + down_proj.bias.to(output.dtype)

    requested = int(page_mask.to(torch.long).sum().item())
    unique = 0
    selected_payload = 0

    for page_idx, page in enumerate(pages):
        active = page_mask[:, page_idx].to(dtype=torch.bool, device=hidden.device)
        if not bool(active.any()):
            continue

        unique += 1
        selected_payload += page.payload_bytes

        x = hidden[active]
        gate_weight = gate_proj.weight[page.start : page.end]
        up_weight = up_proj.weight[page.start : page.end]
        down_weight = down_proj.weight[:, page.start : page.end]

        gate = torch.nn.functional.linear(
            x,
            gate_weight,
            (
                gate_proj.bias[page.start : page.end]
                if gate_proj.bias is not None
                else None
            ),
        )
        up = torch.nn.functional.linear(
            x,
            up_weight,
            (
                up_proj.bias[page.start : page.end]
                if up_proj.bias is not None
                else None
            ),
        )
        intermediate = torch.nn.functional.silu(gate) * up
        contribution = torch.nn.functional.linear(
            intermediate,
            down_weight,
            bias=None,
        )
        output[active] = output[active] + contribution

    return output, PagedSwiGLUStats(
        requested_page_uses=requested,
        unique_pages_loaded=unique,
        selected_payload_bytes=selected_payload,
    )


@torch.no_grad()
def greedy_pages_for_relative_output_error(
    intermediate: Tensor,
    down_proj: nn.Linear,
    pages: list[SwiGLUPage],
    *,
    relative_error: float,
) -> Tensor:
    """Greedy page count needed to reproduce the dense MLP contribution.

    The metric operates on the additive down-projection contribution before any
    residual connection. At each step it chooses the page that most reduces the
    remaining squared output error for that row.
    """

    if not 0 <= relative_error < 1:
        raise ValueError("relative_error must be in [0, 1)")
    if intermediate.ndim != 2:
        raise ValueError("intermediate must have shape [rows, neurons]")

    contributions = swiglu_page_contributions(
        intermediate,
        down_proj,
        pages,
    ).float()
    full = contributions.sum(dim=1)
    residual = full.clone()
    denominator = full.pow(2).sum(dim=-1).clamp_min(1e-20)
    target = float(relative_error) ** 2

    selected = torch.zeros(
        contributions.shape[:2],
        dtype=torch.bool,
        device=contributions.device,
    )
    counts = torch.zeros(
        contributions.shape[0],
        dtype=torch.long,
        device=contributions.device,
    )
    done = residual.pow(2).sum(dim=-1) / denominator <= target
    rows = torch.arange(contributions.shape[0], device=contributions.device)

    for _ in range(contributions.shape[1]):
        active = ~done
        if not bool(active.any()):
            break

        candidate_residual = residual[:, None, :] - contributions
        candidate_error = candidate_residual.pow(2).sum(dim=-1)
        candidate_error = candidate_error.masked_fill(selected, float("inf"))
        best = candidate_error.argmin(dim=-1)

        selected[rows[active], best[active]] = True
        residual[active] = (
            residual[active]
            - contributions[rows[active], best[active]]
        )
        counts = counts + active.long()
        done = residual.pow(2).sum(dim=-1) / denominator <= target

    if not bool(done.all()):
        raise RuntimeError("full page set failed to reconstruct dense output")
    return counts

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
