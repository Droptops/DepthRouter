import torch

from depth_router.conformal_memory import (
    aps_calibration_threshold,
    aps_prediction_mask,
    anytime_bonferroni_thresholds,
    bytes_for_mask,
    empirical_coverage,
    mean_set_size,
    nested_anytime_masks,
)


def test_aps_mask_contains_top_pages_until_threshold() -> None:
    probabilities = torch.tensor([[0.60, 0.25, 0.10, 0.05]])
    mask = aps_prediction_mask(probabilities, 0.80)

    assert mask.tolist() == [[True, True, False, False]]
    assert mean_set_size(mask) == 2.0


def test_calibration_threshold_uses_true_page_rank_mass() -> None:
    probabilities = torch.tensor(
        [
            [0.70, 0.20, 0.10],
            [0.20, 0.70, 0.10],
            [0.45, 0.35, 0.20],
            [0.60, 0.25, 0.15],
        ]
    )
    labels = torch.tensor([0, 1, 1, 0])

    threshold = aps_calibration_threshold(probabilities, labels, alpha=0.25)
    assert 0.7 <= threshold <= 0.8


def test_nested_anytime_sets_never_expand() -> None:
    spins = [
        torch.tensor([[0.40, 0.30, 0.20, 0.10]]),
        torch.tensor([[0.60, 0.20, 0.15, 0.05]]),
        torch.tensor([[0.80, 0.10, 0.07, 0.03]]),
    ]
    thresholds = [0.90, 0.80, 0.80]
    nested = nested_anytime_masks(spins, thresholds)

    sizes = [int(mask.sum()) for mask in nested]
    assert sizes[1] <= sizes[0]
    assert sizes[2] <= sizes[1]


def test_empirical_coverage_and_bytes() -> None:
    mask = torch.tensor(
        [
            [True, False],
            [True, True],
        ]
    )
    labels = torch.tensor([0, 1])

    assert empirical_coverage(mask, labels) == 1.0
    assert bytes_for_mask(mask, page_bytes=4096).tolist() == [4096, 8192]


def test_anytime_calibration_returns_one_threshold_per_spin() -> None:
    labels = torch.tensor([0, 1, 0, 1])
    spins = [
        torch.tensor(
            [
                [0.8, 0.2],
                [0.3, 0.7],
                [0.6, 0.4],
                [0.4, 0.6],
            ]
        ),
        torch.tensor(
            [
                [0.9, 0.1],
                [0.2, 0.8],
                [0.7, 0.3],
                [0.3, 0.7],
            ]
        ),
    ]

    thresholds = anytime_bonferroni_thresholds(spins, labels, alpha=0.10)
    assert len(thresholds) == 2
    assert all(0 < threshold <= 1 for threshold in thresholds)
