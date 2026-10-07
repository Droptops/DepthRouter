from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def _as_probabilities(posterior: Tensor) -> Tensor:
    if posterior.ndim < 1:
        raise ValueError("posterior must have at least one dimension")
    if posterior.shape[-1] < 1:
        raise ValueError("posterior must contain at least one state")
    if not torch.is_floating_point(posterior):
        posterior = posterior.float()
    if bool((posterior < 0).any()):
        raise ValueError("posterior probabilities must be non-negative")
    total = posterior.sum(dim=-1, keepdim=True)
    if bool((total <= 0).any()):
        raise ValueError("posterior rows must have positive mass")
    return posterior / total


def posterior_entropy(posterior: Tensor, eps: float = 1e-12) -> Tensor:
    """Shannon entropy of a categorical posterior, in nats."""

    p = _as_probabilities(posterior)
    return -(p * p.clamp_min(eps).log()).sum(dim=-1)


@dataclass(frozen=True)
class PageLayout:
    """Physical placement of causal states into fixed-size pages."""

    state_to_page: Tensor
    states_per_page: int
    page_bytes: int

    def __post_init__(self) -> None:
        if self.state_to_page.ndim != 1:
            raise ValueError("state_to_page must have shape [states]")
        if self.state_to_page.numel() < 1:
            raise ValueError("layout must contain at least one state")
        if self.states_per_page < 1:
            raise ValueError("states_per_page must be >= 1")
        if self.page_bytes < 1:
            raise ValueError("page_bytes must be >= 1")
        if bool((self.state_to_page < 0).any()):
            raise ValueError("page ids must be non-negative")

    @property
    def num_states(self) -> int:
        return int(self.state_to_page.numel())

    @property
    def num_pages(self) -> int:
        return int(self.state_to_page.max().item()) + 1


def page_masses(posterior: Tensor, layout: PageLayout) -> Tensor:
    """Project state posterior mass onto physical pages."""

    p = _as_probabilities(posterior)
    if p.shape[-1] != layout.num_states:
        raise ValueError("posterior state dimension does not match layout")

    flat = p.reshape(-1, layout.num_states)
    out = torch.zeros(
        flat.shape[0],
        layout.num_pages,
        dtype=flat.dtype,
        device=flat.device,
    )
    index = layout.state_to_page.to(flat.device).expand(flat.shape[0], -1)
    out.scatter_add_(1, index, flat)
    return out.reshape(*p.shape[:-1], layout.num_pages)


def working_set_pages(
    posterior: Tensor,
    layout: PageLayout,
    *,
    mass_target: float = 0.95,
) -> Tensor:
    """Minimum fixed-size pages needed to cover mass_target posterior mass.

    Because every page has equal byte size, sorting page masses descending is
    exact for the minimum-page coverage problem.
    """

    if not 0 < mass_target <= 1:
        raise ValueError("mass_target must be in (0, 1]")

    masses = page_masses(posterior, layout)
    sorted_mass = masses.sort(dim=-1, descending=True).values
    cumulative = sorted_mass.cumsum(dim=-1)
    # Number of pages before the threshold plus the threshold-crossing page.
    before = (cumulative < mass_target).sum(dim=-1)
    return (before + 1).clamp_max(layout.num_pages)


def predictive_working_set_bytes(
    posterior: Tensor,
    layout: PageLayout,
    *,
    mass_target: float = 0.95,
) -> Tensor:
    return working_set_pages(
        posterior,
        layout,
        mass_target=mass_target,
    ) * layout.page_bytes


def posterior_affinity(posterior: Tensor) -> Tensor:
    """Co-posterior affinity A_ij = E[q_i q_j].

    States that are simultaneously plausible receive large affinity. Packing
    them together makes uncertainty cheaper because one page covers multiple
    competing hypotheses.
    """

    p = _as_probabilities(posterior)
    flat = p.reshape(-1, p.shape[-1])
    affinity = flat.T @ flat / max(flat.shape[0], 1)
    affinity = 0.5 * (affinity + affinity.T)
    affinity.fill_diagonal_(0)
    return affinity


def greedy_affinity_layout(
    affinity: Tensor,
    *,
    states_per_page: int,
    page_bytes: int,
) -> PageLayout:
    """Pack highly co-posterior states onto the same physical page.

    This is a deterministic greedy graph-partition heuristic. It is deliberately
    simple so the first experiment tests the placement objective rather than a
    sophisticated partitioner.
    """

    if affinity.ndim != 2 or affinity.shape[0] != affinity.shape[1]:
        raise ValueError("affinity must be square")
    if states_per_page < 1:
        raise ValueError("states_per_page must be >= 1")
    if page_bytes < 1:
        raise ValueError("page_bytes must be >= 1")

    n = int(affinity.shape[0])
    if n == 0:
        raise ValueError("affinity must contain at least one state")

    scores = affinity.sum(dim=1)
    unassigned = set(range(n))
    assignment = torch.full((n,), -1, dtype=torch.long)
    page = 0

    while unassigned:
        seed = max(unassigned, key=lambda i: (float(scores[i]), -i))
        members = [seed]
        unassigned.remove(seed)

        while unassigned and len(members) < states_per_page:
            best = max(
                unassigned,
                key=lambda i: (
                    float(affinity[i, members].sum()),
                    float(scores[i]),
                    -i,
                ),
            )
            members.append(best)
            unassigned.remove(best)

        assignment[members] = page
        page += 1

    return PageLayout(
        state_to_page=assignment,
        states_per_page=states_per_page,
        page_bytes=page_bytes,
    )


def random_layout(
    num_states: int,
    *,
    states_per_page: int,
    page_bytes: int,
    seed: int,
) -> PageLayout:
    if num_states < 1:
        raise ValueError("num_states must be >= 1")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(num_states, generator=generator)
    assignment = torch.empty(num_states, dtype=torch.long)
    for rank, state in enumerate(order.tolist()):
        assignment[state] = rank // states_per_page
    return PageLayout(assignment, states_per_page, page_bytes)


def sequential_layout(
    num_states: int,
    *,
    states_per_page: int,
    page_bytes: int,
) -> PageLayout:
    if num_states < 1:
        raise ValueError("num_states must be >= 1")
    assignment = torch.arange(num_states, dtype=torch.long) // states_per_page
    return PageLayout(assignment, states_per_page, page_bytes)


def requested_pages(
    posterior: Tensor,
    layout: PageLayout,
    *,
    mass_target: float = 0.95,
) -> list[list[int]]:
    """Return each token's page set needed to cover mass_target posterior mass."""

    masses = page_masses(posterior, layout)
    flat = masses.reshape(-1, masses.shape[-1])
    requests: list[list[int]] = []

    for row in flat:
        order = torch.argsort(row, descending=True)
        total = 0.0
        pages: list[int] = []
        for idx in order.tolist():
            pages.append(int(idx))
            total += float(row[idx])
            if total >= mass_target:
                break
        requests.append(pages)

    return requests


def coalesced_transfer_bytes(
    posterior: Tensor,
    layout: PageLayout,
    *,
    mass_target: float = 0.95,
) -> tuple[int, int, int]:
    """Return requested pages, unique transfers, and coalesced transfer bytes."""

    requests = requested_pages(posterior, layout, mass_target=mass_target)
    requested = sum(len(row) for row in requests)
    unique = len({page for row in requests for page in row})
    return requested, unique, unique * layout.page_bytes
