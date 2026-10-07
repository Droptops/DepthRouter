import torch
from torch import nn

from depth_router.operator_pages import (
    activation_address_scores,
    contiguous_mlp_pages,
    linear2_page_contributions,
    make_output_page_sketches,
    page_bytes_tensor,
    page_output_weight_norms,
    reconstruct_linear2,
    sketch_address_scores,
)


def test_page_contributions_reconstruct_dense_linear2() -> None:
    torch.manual_seed(0)
    linear1 = nn.Linear(5, 12)
    linear2 = nn.Linear(12, 4)
    activation = torch.randn(3, 2, 12)

    pages = contiguous_mlp_pages(
        linear1,
        linear2,
        units_per_page=4,
    )
    contributions = linear2_page_contributions(
        activation,
        linear2,
        pages,
    )
    reconstructed = reconstruct_linear2(contributions, linear2)

    assert contributions.shape == (3, 2, 3, 4)
    assert torch.allclose(
        reconstructed,
        linear2(activation),
        atol=1e-6,
    )


def test_page_mask_skips_unselected_operator_pages() -> None:
    torch.manual_seed(1)
    linear1 = nn.Linear(4, 8)
    linear2 = nn.Linear(8, 3)
    activation = torch.randn(2, 8)
    pages = contiguous_mlp_pages(
        linear1,
        linear2,
        units_per_page=4,
    )
    contributions = linear2_page_contributions(
        activation,
        linear2,
        pages,
    )
    mask = torch.tensor(
        [
            [True, False],
            [False, True],
        ]
    )

    got = reconstruct_linear2(
        contributions,
        linear2,
        page_mask=mask,
    )

    expected0 = (
        activation[0, :4] @ linear2.weight[:, :4].T
        + linear2.bias
    )
    expected1 = (
        activation[1, 4:] @ linear2.weight[:, 4:].T
        + linear2.bias
    )
    assert torch.allclose(got[0], expected0, atol=1e-6)
    assert torch.allclose(got[1], expected1, atol=1e-6)


def test_page_bytes_cover_first_and_second_projection_slices() -> None:
    linear1 = nn.Linear(5, 8, bias=True)
    linear2 = nn.Linear(8, 3, bias=True)
    pages = contiguous_mlp_pages(
        linear1,
        linear2,
        units_per_page=4,
    )
    sizes = page_bytes_tensor(pages)

    element_size = linear1.weight.element_size()
    expected_per_page = (4 * 5 + 4 + 3 * 4) * element_size
    assert sizes.tolist() == [expected_per_page, expected_per_page]


def test_activation_address_scores_are_normalized_without_reading_payload_pages() -> None:
    torch.manual_seed(2)
    linear1 = nn.Linear(4, 8)
    linear2 = nn.Linear(8, 3)
    pages = contiguous_mlp_pages(
        linear1,
        linear2,
        units_per_page=4,
    )
    activation = torch.tensor(
        [
            [3.0, 3.0, 3.0, 3.0, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1, 3.0, 3.0, 3.0, 3.0],
        ]
    )
    norms = page_output_weight_norms(linear2, pages)
    scores = activation_address_scores(activation, pages, norms)

    assert scores.shape == (2, 2)
    assert torch.allclose(scores.sum(dim=-1), torch.ones(2))
    assert scores[0, 0] > scores[0, 1]
    assert scores[1, 1] > scores[1, 0]


def test_sketch_address_scores_are_normalized() -> None:
    torch.manual_seed(3)
    linear1 = nn.Linear(4, 8)
    linear2 = nn.Linear(8, 4)
    pages = contiguous_mlp_pages(linear1, linear2, units_per_page=4)
    activation = torch.randn(5, 8)
    sketches = make_output_page_sketches(
        linear2,
        pages,
        sketch_dim=2,
        seed=11,
    )
    scores = sketch_address_scores(activation, pages, sketches)

    assert scores.shape == (5, 2)
    assert torch.allclose(scores.sum(dim=-1), torch.ones(5), atol=1e-6)
