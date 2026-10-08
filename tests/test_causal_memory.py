import torch

from depth_router.causal_memory import (
    PageLayout,
    coalesced_transfer_bytes,
    greedy_affinity_layout,
    page_masses,
    posterior_affinity,
    posterior_entropy,
    predictive_working_set_bytes,
    smooth_address_entropy_nats,
    working_set_pages,
)


def test_page_projection_preserves_probability_mass() -> None:
    posterior = torch.tensor([[0.4, 0.3, 0.2, 0.1]])
    layout = PageLayout(
        state_to_page=torch.tensor([0, 0, 1, 1]),
        states_per_page=2,
        page_bytes=1024,
    )
    masses = page_masses(posterior, layout)

    assert torch.allclose(masses, torch.tensor([[0.7, 0.3]]))
    assert torch.allclose(masses.sum(dim=-1), torch.ones(1))


def test_working_set_is_exact_for_uniform_page_sizes() -> None:
    posterior = torch.tensor([[0.40, 0.35, 0.15, 0.10]])
    layout = PageLayout(
        state_to_page=torch.tensor([0, 0, 1, 2]),
        states_per_page=2,
        page_bytes=4096,
    )

    assert working_set_pages(posterior, layout, mass_target=0.75).item() == 1
    assert predictive_working_set_bytes(
        posterior,
        layout,
        mass_target=0.90,
    ).item() == 8192


def test_entropy_falls_for_more_concentrated_posterior() -> None:
    diffuse = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
    sharp = torch.tensor([[0.85, 0.05, 0.05, 0.05]])
    assert posterior_entropy(sharp).item() < posterior_entropy(diffuse).item()


def test_affinity_layout_colocates_competing_hypotheses() -> None:
    posterior = torch.tensor(
        [
            [0.50, 0.45, 0.03, 0.02],
            [0.45, 0.50, 0.02, 0.03],
            [0.02, 0.03, 0.50, 0.45],
            [0.03, 0.02, 0.45, 0.50],
        ]
    )
    layout = greedy_affinity_layout(
        posterior_affinity(posterior),
        states_per_page=2,
        page_bytes=1024,
    )

    assert layout.state_to_page[0].item() == layout.state_to_page[1].item()
    assert layout.state_to_page[2].item() == layout.state_to_page[3].item()
    assert layout.state_to_page[0].item() != layout.state_to_page[2].item()


def test_colocation_reduces_working_set() -> None:
    posterior = torch.tensor([[0.46, 0.44, 0.05, 0.05]])
    good = PageLayout(torch.tensor([0, 0, 1, 1]), 2, 1024)
    bad = PageLayout(torch.tensor([0, 1, 0, 1]), 2, 1024)

    assert working_set_pages(posterior, good, mass_target=0.90).item() == 1
    assert working_set_pages(posterior, bad, mass_target=0.90).item() == 2


def test_coalescing_counts_unique_physical_pages() -> None:
    posterior = torch.tensor(
        [
            [0.90, 0.10, 0.00, 0.00],
            [0.80, 0.20, 0.00, 0.00],
            [0.00, 0.00, 0.85, 0.15],
        ]
    )
    layout = PageLayout(torch.tensor([0, 0, 1, 1]), 2, 4096)

    requested, unique, nbytes = coalesced_transfer_bytes(
        posterior,
        layout,
        mass_target=0.80,
    )
    assert requested == 3
    assert unique == 2
    assert nbytes == 8192


def test_smooth_address_entropy_exponentiates_to_page_count() -> None:
    posterior = torch.tensor([[0.46, 0.44, 0.05, 0.05]])
    layout = PageLayout(torch.tensor([0, 0, 1, 1]), 2, 1024)

    pages = working_set_pages(posterior, layout, mass_target=0.90).float()
    entropy = smooth_address_entropy_nats(
        posterior,
        layout,
        mass_target=0.90,
    )
    assert torch.allclose(entropy.exp(), pages)
