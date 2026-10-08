import torch

from depth_router.conformal_memory import (
    anytime_bonferroni_thresholds,
    aps_calibration_threshold,
    aps_prediction_mask,
    bytes_for_mask,
    calibrate_required_set_tiers,
    distortion_coverage,
    distortion_prefix_calibration_threshold,
    empirical_coverage,
    exclusive_residency_masks,
    mean_set_size,
    nested_anytime_masks,
    required_set_calibration_threshold,
    required_set_coverage,
    residency_tier_masks,
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
    assert 0.7 <= threshold <= 0.800001


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


def test_required_set_conformal_covers_all_required_pages() -> None:
    calibration = torch.tensor(
        [
            [0.55, 0.30, 0.10, 0.05],
            [0.50, 0.25, 0.20, 0.05],
            [0.45, 0.35, 0.15, 0.05],
            [0.60, 0.20, 0.15, 0.05],
        ]
    )
    required = torch.tensor(
        [
            [True, True, False, False],
            [True, False, True, False],
            [True, True, False, False],
            [True, False, True, False],
        ]
    )

    threshold = required_set_calibration_threshold(
        calibration,
        required,
        alpha=0.25,
    )
    prediction = aps_prediction_mask(calibration, threshold)

    assert required_set_coverage(prediction, required) >= 0.75


def test_required_set_calibration_supports_zero_fault_targets() -> None:
    probabilities = torch.tensor([[0.6, 0.4]])
    required = torch.tensor([[False, False]])

    threshold = required_set_calibration_threshold(
        probabilities,
        required,
        alpha=0.1,
    )
    prediction = aps_prediction_mask(probabilities, threshold)

    assert threshold == 0.0
    assert prediction.tolist() == [[False, False]]
    assert required_set_coverage(prediction, required) == 1.0


def test_risk_coded_memory_tiers_are_nested() -> None:
    calibration = torch.tensor(
        [
            [0.70, 0.20, 0.08, 0.02],
            [0.55, 0.25, 0.15, 0.05],
            [0.45, 0.30, 0.20, 0.05],
            [0.60, 0.20, 0.15, 0.05],
            [0.35, 0.30, 0.25, 0.10],
            [0.50, 0.25, 0.20, 0.05],
        ]
    )
    required = torch.tensor(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, False, True, False],
            [True, False, False, False],
            [True, True, True, False],
            [True, False, True, False],
        ]
    )

    tiers = calibrate_required_set_tiers(
        calibration,
        required,
        alphas=[0.30, 0.10, 0.01],
    )
    masks = residency_tier_masks(calibration, tiers)
    exclusive = exclusive_residency_masks(masks)

    assert tiers[0][1] <= tiers[1][1] <= tiers[2][1]
    assert torch.all(masks[0] <= masks[1])
    assert torch.all(masks[1] <= masks[2])
    assert torch.equal(exclusive[0] | exclusive[1] | exclusive[2], masks[2])


def test_distortion_prefix_calibration_targets_quality_not_page_identity() -> None:
    probabilities = torch.tensor(
        [
            [0.60, 0.30, 0.10],
            [0.55, 0.25, 0.20],
            [0.70, 0.20, 0.10],
            [0.50, 0.30, 0.20],
        ]
    )
    # Columns are k = 0, 1, 2, 3 selected pages.
    distortions = torch.tensor(
        [
            [0.20, 0.04, 0.01, 0.00],
            [0.20, 0.08, 0.03, 0.00],
            [0.20, 0.02, 0.01, 0.00],
            [0.20, 0.09, 0.02, 0.00],
        ]
    )

    threshold = distortion_prefix_calibration_threshold(
        probabilities,
        distortions,
        tolerance=0.05,
        alpha=0.25,
    )

    assert 0.5 <= threshold <= 1.0
    realized = torch.tensor([0.01, 0.02, 0.08, 0.03])
    assert distortion_coverage(realized, tolerance=0.05) == 0.75


def test_distortion_calibration_uses_last_violation_not_first_pass() -> None:
    probabilities = torch.tensor([[0.50, 0.30, 0.20]])
    # k=1 passes, k=2 becomes unsafe again, k=3 (dense) is safe.
    distortions = torch.tensor([[0.20, 0.04, 0.08, 0.00]])

    threshold = distortion_prefix_calibration_threshold(
        probabilities,
        distortions,
        tolerance=0.05,
        alpha=0.5,
    )

    # The valid monotone requirement is the full 3-page prefix, not the first
    # passing 1-page prefix.
    assert threshold == 1.0
