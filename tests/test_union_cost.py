import math

import torch

from depth_router.union_cost import (
    coalescing_gain,
    expected_independent_bytes,
    expected_union_bytes,
    normalized_independent_cost,
    normalized_union_cost,
    page_fault_probability,
)


def test_union_cost_counts_one_shared_transfer() -> None:
    probabilities = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ]
    )
    page_bytes = torch.tensor([100.0, 200.0])

    assert expected_union_bytes(probabilities, page_bytes).item() == 100.0
    assert expected_independent_bytes(probabilities, page_bytes).item() == 300.0
    assert coalescing_gain(probabilities, page_bytes).item() == 3.0


def test_union_probability_matches_closed_form() -> None:
    probabilities = torch.tensor(
        [
            [0.5, 0.2],
            [0.5, 0.4],
        ]
    )
    got = page_fault_probability(probabilities)

    assert torch.allclose(
        got,
        torch.tensor(
            [
                1.0 - (0.5 * 0.5),
                1.0 - (0.8 * 0.6),
            ]
        ),
    )


def test_resident_page_is_free_in_incremental_traffic() -> None:
    probabilities = torch.ones(4, 2)
    page_bytes = torch.tensor([100.0, 200.0])
    resident = torch.tensor([1.0, 0.0])

    assert (
        expected_union_bytes(
            probabilities,
            page_bytes,
            resident_probability=resident,
        ).item()
        == 200.0
    )


def test_union_cost_is_differentiable() -> None:
    logits = torch.tensor(
        [
            [0.0, 1.0],
            [0.5, -0.5],
        ],
        requires_grad=True,
    )
    probabilities = torch.sigmoid(logits)
    cost = expected_union_bytes(
        probabilities,
        torch.tensor([100.0, 100.0]),
    )
    cost.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert bool((logits.grad > 0).all())


def test_normalized_costs_have_expected_scale() -> None:
    probabilities = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
        ]
    )
    page_bytes = torch.tensor([100.0, 100.0])

    assert math.isclose(
        normalized_union_cost(probabilities, page_bytes).item(),
        0.5,
    )
    assert math.isclose(
        normalized_independent_cost(probabilities, page_bytes).item(),
        0.5,
    )
