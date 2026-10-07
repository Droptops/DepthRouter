from __future__ import annotations

import math

import torch
from torch import Tensor


def _normalize(probabilities: Tensor) -> Tensor:
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [examples, pages]")
    if probabilities.shape[1] < 1:
        raise ValueError("probabilities must contain at least one page")
    if not torch.is_floating_point(probabilities):
        probabilities = probabilities.float()
    if bool((probabilities < 0).any()):
        raise ValueError("probabilities must be non-negative")
    total = probabilities.sum(dim=-1, keepdim=True)
    if bool((total <= 0).any()):
        raise ValueError("each row must have positive mass")
    return probabilities / total


def aps_calibration_threshold(
    probabilities: Tensor,
    true_pages: Tensor,
    *,
    alpha: float,
) -> float:
    """Calibrate an Adaptive Prediction Set threshold.

    The nonconformity score is the cumulative probability mass through the true
    page after sorting pages by descending predicted probability.

    Under the usual split-conformal exchangeability assumption, using the
    finite-sample quantile gives marginal coverage at least 1 - alpha.
    """

    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    p = _normalize(probabilities)
    if true_pages.ndim != 1 or true_pages.shape[0] != p.shape[0]:
        raise ValueError("true_pages must have shape [examples]")
    if bool((true_pages < 0).any()) or bool((true_pages >= p.shape[1]).any()):
        raise ValueError("true page ids are out of range")

    order = torch.argsort(p, dim=-1, descending=True)
    sorted_p = torch.gather(p, 1, order)
    cumulative = sorted_p.cumsum(dim=-1)
    is_true = order.eq(true_pages[:, None])
    scores = cumulative[is_true]

    n = int(scores.numel())
    rank = min(math.ceil((n + 1) * (1.0 - alpha)), n)
    rank = max(rank, 1)
    return float(torch.kthvalue(scores, rank).values.item())


def required_set_calibration_threshold(
    probabilities: Tensor,
    required_mask: Tensor,
    *,
    alpha: float,
) -> float:
    """Calibrate coverage for a *set* of required operator pages.

    The score is the cumulative predicted mass through the worst-ranked
    required page. A prediction set that reaches the calibrated threshold then
    contains every required page with split-conformal marginal coverage under
    exchangeability.
    """

    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    p = _normalize(probabilities)
    if required_mask.shape != p.shape:
        raise ValueError("required_mask must match probabilities")
    required = required_mask.to(dtype=torch.bool, device=p.device)
    order = torch.argsort(p, dim=-1, descending=True)
    sorted_p = torch.gather(p, 1, order)
    sorted_required = torch.gather(required, 1, order)
    cumulative = sorted_p.cumsum(dim=-1)

    # The candidate set must reach at least the least-favored required page.
    scores = torch.where(
        sorted_required,
        cumulative,
        torch.zeros_like(cumulative),
    ).max(dim=-1).values

    n = int(scores.numel())
    rank = min(math.ceil((n + 1) * (1.0 - alpha)), n)
    rank = max(rank, 1)
    return float(torch.kthvalue(scores, rank).values.item())


def required_set_coverage(
    prediction_mask: Tensor,
    required_mask: Tensor,
) -> float:
    """Fraction of examples for which every required page is resident."""

    if prediction_mask.ndim != 2:
        raise ValueError("prediction_mask must have shape [examples, pages]")
    if required_mask.shape != prediction_mask.shape:
        raise ValueError("required_mask must match prediction_mask")

    required = required_mask.to(
        dtype=torch.bool,
        device=prediction_mask.device,
    )
    covered = (prediction_mask | ~required).all(dim=-1)
    return float(covered.float().mean().item())


def aps_prediction_mask(
    probabilities: Tensor,
    threshold: float,
) -> Tensor:
    """Return the smallest top-probability set reaching the APS threshold."""

    if threshold < 0:
        raise ValueError("threshold must be non-negative")

    p = _normalize(probabilities)
    threshold = min(float(threshold), 1.0)
    if threshold == 0:
        return torch.zeros_like(p, dtype=torch.bool)

    order = torch.argsort(p, dim=-1, descending=True)
    sorted_p = torch.gather(p, 1, order)
    cumulative = sorted_p.cumsum(dim=-1)

    counts = (cumulative < threshold).sum(dim=-1) + 1
    counts = counts.clamp_max(p.shape[1])

    mask = torch.zeros_like(p, dtype=torch.bool)
    ranks = torch.arange(p.shape[1], device=p.device)[None, :]
    selected_sorted = ranks < counts[:, None]
    mask.scatter_(1, order, selected_sorted)
    return mask


def empirical_coverage(mask: Tensor, true_pages: Tensor) -> float:
    if mask.ndim != 2:
        raise ValueError("mask must have shape [examples, pages]")
    if true_pages.ndim != 1 or true_pages.shape[0] != mask.shape[0]:
        raise ValueError("true_pages must have shape [examples]")
    covered = mask[
        torch.arange(mask.shape[0], device=mask.device),
        true_pages.to(mask.device),
    ]
    return float(covered.float().mean().item())


def mean_set_size(mask: Tensor) -> float:
    if mask.ndim != 2:
        raise ValueError("mask must have shape [examples, pages]")
    return float(mask.sum(dim=-1).float().mean().item())


def anytime_bonferroni_thresholds(
    calibration_probabilities: list[Tensor],
    true_pages: Tensor,
    *,
    alpha: float,
) -> list[float]:
    """Calibrate a finite-horizon family of per-spin conformal sets.

    Alpha is split across K solver spins. By a union bound, if the split
    conformal assumptions hold at each spin, the probability that the true page
    is excluded at *any* of the K calibrated spins is at most alpha.

    This is conservative. It is intentionally the first implementation because
    it makes the optional-stopping claim explicit instead of silently assuming
    that marginal per-spin coverage remains valid under adaptive halting.
    """

    if not calibration_probabilities:
        raise ValueError("at least one spin is required")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    per_spin_alpha = alpha / len(calibration_probabilities)
    return [
        aps_calibration_threshold(p, true_pages, alpha=per_spin_alpha)
        for p in calibration_probabilities
    ]


def nested_anytime_masks(
    probabilities_by_spin: list[Tensor],
    thresholds: list[float],
) -> list[Tensor]:
    """Intersect calibrated page sets so the resident candidate set only shrinks."""

    if len(probabilities_by_spin) != len(thresholds):
        raise ValueError("one threshold is required per spin")
    if not probabilities_by_spin:
        raise ValueError("at least one spin is required")

    running: Tensor | None = None
    nested: list[Tensor] = []
    for probabilities, threshold in zip(
        probabilities_by_spin,
        thresholds,
        strict=True,
    ):
        current = aps_prediction_mask(probabilities, threshold)
        running = current if running is None else (running & current)
        nested.append(running.clone())

    return nested


def bytes_for_mask(mask: Tensor, *, page_bytes: int) -> Tensor:
    if page_bytes < 1:
        raise ValueError("page_bytes must be >= 1")
    if mask.ndim != 2:
        raise ValueError("mask must have shape [examples, pages]")
    return mask.sum(dim=-1) * int(page_bytes)
