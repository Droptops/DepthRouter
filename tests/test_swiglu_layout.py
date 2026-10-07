import torch
from torch import nn

from depth_router.swiglu_layout import (
    contiguous_swiglu_pages,
    learn_codemand_permutation,
    learn_simhash_codemand_permutation,
    page_importance,
    paged_swiglu_reference,
    pages_for_mass,
    permute_swiglu_neurons_,
    swiglu_intermediate,
    swiglu_output,
    swiglu_page_contributions,
)


def make_layers() -> tuple[nn.Linear, nn.Linear, nn.Linear]:
    gate = nn.Linear(6, 12, bias=False)
    up = nn.Linear(6, 12, bias=False)
    down = nn.Linear(12, 5, bias=False)
    return gate, up, down


def test_swiglu_pages_reconstruct_dense_output() -> None:
    torch.manual_seed(0)
    gate, up, down = make_layers()
    hidden = torch.randn(7, 6)
    intermediate = swiglu_intermediate(hidden, gate, up)
    pages = contiguous_swiglu_pages(gate, up, down, units_per_page=4)
    contributions = swiglu_page_contributions(intermediate, down, pages)

    reconstructed = contributions.sum(dim=-2)
    dense = swiglu_output(hidden, gate, up, down)
    assert torch.allclose(reconstructed, dense, atol=1e-6)


def test_swiglu_neuron_permutation_preserves_dense_function_exactly() -> None:
    torch.manual_seed(1)
    gate, up, down = make_layers()
    hidden = torch.randn(11, 6)
    before = swiglu_output(hidden, gate, up, down)

    permutation = torch.randperm(12)
    permute_swiglu_neurons_(gate, up, down, permutation)
    after = swiglu_output(hidden, gate, up, down)

    assert torch.allclose(before, after, atol=1e-6, rtol=1e-6)


def test_codemand_compiler_groups_jointly_active_neurons() -> None:
    torch.manual_seed(2)
    _, _, down = make_layers()

    # Two hidden demand families. The raw neuron order intentionally interleaves
    # them, so contiguous pages are poor before compilation.
    intermediate = torch.zeros(200, 12)
    family_a = torch.tensor([0, 2, 4, 6, 8, 10])
    family_b = torch.tensor([1, 3, 5, 7, 9, 11])
    intermediate[:100, family_a] = 1.0
    intermediate[100:, family_b] = 1.0
    down.weight.data.fill_(1.0)

    permutation = learn_codemand_permutation(
        intermediate,
        down,
        units_per_page=6,
    )

    first_page = set(permutation[:6].tolist())
    assert (
        first_page == set(family_a.tolist())
        or first_page == set(family_b.tolist())
    )


def test_compiled_page_layout_can_reduce_mass_working_set() -> None:
    torch.manual_seed(3)
    gate, up, down = make_layers()
    down.weight.data.fill_(1.0)

    intermediate = torch.zeros(100, 12)
    active = torch.tensor([0, 2, 4, 6, 8, 10])
    intermediate[:, active] = 1.0

    raw_pages = contiguous_swiglu_pages(gate, up, down, units_per_page=6)
    raw_scores = page_importance(intermediate, down, raw_pages)
    raw_count = pages_for_mass(raw_scores, mass_target=0.95).float().mean()

    permutation = learn_codemand_permutation(
        intermediate,
        down,
        units_per_page=6,
    )
    permuted_intermediate = intermediate[:, permutation]
    # Permuting down columns is enough for the structural page metric.
    down.weight.data.copy_(down.weight.data[:, permutation])
    compiled_pages = contiguous_swiglu_pages(gate, up, down, units_per_page=6)
    compiled_scores = page_importance(
        permuted_intermediate,
        down,
        compiled_pages,
    )
    compiled_count = pages_for_mass(
        compiled_scores,
        mass_target=0.95,
    ).float().mean()

    assert compiled_count < raw_count


def test_simhash_codemand_permutation_is_valid_and_deterministic() -> None:
    torch.manual_seed(4)
    _, _, down = make_layers()
    intermediate = torch.randn(128, 12)

    first = learn_simhash_codemand_permutation(
        intermediate,
        down,
        bits=8,
        max_trace_tokens=64,
        seed=11,
    )
    second = learn_simhash_codemand_permutation(
        intermediate,
        down,
        bits=8,
        max_trace_tokens=64,
        seed=11,
    )

    assert torch.equal(first, second)
    assert torch.equal(torch.sort(first).values, torch.arange(12))


def test_paged_swiglu_full_mask_matches_dense_and_coalesces_page_loads() -> None:
    torch.manual_seed(5)
    gate, up, down = make_layers()
    hidden = torch.randn(9, 6)
    pages = contiguous_swiglu_pages(gate, up, down, units_per_page=4)
    mask = torch.ones(9, len(pages), dtype=torch.bool)

    got, stats = paged_swiglu_reference(
        hidden,
        gate,
        up,
        down,
        pages,
        mask,
    )
    expected = swiglu_output(hidden, gate, up, down)

    assert torch.allclose(got, expected, atol=1e-6)
    assert stats.requested_page_uses == 9 * len(pages)
    assert stats.unique_pages_loaded == len(pages)
    assert stats.selected_payload_bytes == sum(page.payload_bytes for page in pages)


def test_paged_swiglu_partial_mask_groups_tokens_by_page() -> None:
    torch.manual_seed(6)
    gate, up, down = make_layers()
    hidden = torch.randn(6, 6)
    pages = contiguous_swiglu_pages(gate, up, down, units_per_page=6)
    mask = torch.tensor(
        [
            [True, False],
            [True, False],
            [True, False],
            [False, True],
            [False, True],
            [False, True],
        ]
    )

    _, stats = paged_swiglu_reference(
        hidden,
        gate,
        up,
        down,
        pages,
        mask,
    )

    assert stats.requested_page_uses == 6
    assert stats.unique_pages_loaded == 2
    assert stats.selected_payload_bytes == sum(page.payload_bytes for page in pages)
