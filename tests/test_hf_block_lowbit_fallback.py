# ruff: noqa: I001
import sys
from pathlib import Path

import pytest
import torch


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from hf_block_lowbit_fallback import (
    CandidateOutcomes,
    calibrate_threshold,
    candidate_confidence,
    fallback_metrics,
    text_set_sha256,
    wilson_interval,
)


def outcomes() -> CandidateOutcomes:
    return CandidateOutcomes(
        kl=torch.tensor([0.001, 0.002, 0.2, 0.3]),
        error=torch.tensor([False, False, True, True]),
        logit_margin=torch.tensor([4.0, 3.0, 2.0, 1.0]),
        probability_margin=torch.tensor([0.8, 0.7, 0.2, 0.1]),
        block_gap=torch.tensor([0.4, 0.3, 0.2, 0.1]),
    )


def test_fallback_metrics_charge_dense_reexecution():
    metrics = fallback_metrics(
        outcomes(),
        outcomes().logit_margin,
        threshold=3.0,
        sparse_execution_fraction=0.8,
        metadata_fraction=0.05,
    )
    assert metrics["sparse_acceptance_fraction"] == 0.5
    assert metrics["average_execution_fraction"] == 0.9
    assert metrics["metadata_plus_average_execution_fraction"] == pytest.approx(0.95)
    assert metrics["top1_agreement"] == 1.0
    assert metrics["top1_wilson_95_lower"] < 1.0
    assert metrics["p95_kl_nats"] >= metrics["mean_kl_nats"]
    assert metrics["passes_strict_gate"]


def test_calibration_selects_largest_quality_passing_acceptance():
    calibrated = calibrate_threshold(
        outcomes(),
        outcomes().logit_margin,
        sparse_execution_fraction=0.8,
        metadata_fraction=0.05,
    )
    assert calibrated is not None
    threshold, metrics = calibrated
    assert threshold == 3.0
    assert metrics["sparse_acceptance_fraction"] == 0.5


def test_stricter_calibration_guardrail_reduces_acceptance():
    guarded = calibrate_threshold(
        outcomes(),
        outcomes().logit_margin,
        sparse_execution_fraction=0.5,
        metadata_fraction=0.05,
        minimum_calibration_top1=1.0,
        maximum_calibration_kl=0.0005,
    )
    assert guarded is not None
    _threshold, metrics = guarded
    assert metrics["sparse_acceptance_fraction"] == 0.25


def test_candidate_confidence_modes_are_causal_scalars():
    candidate = torch.tensor([[4.0, 2.0, 1.0]])
    block_score = torch.tensor([[3.0, 2.0, 0.5]])
    logit = candidate_confidence(candidate, block_score, k=2, mode="logit_margin")
    probability = candidate_confidence(
        candidate,
        block_score,
        k=2,
        mode="probability_margin",
    )
    gap = candidate_confidence(candidate, block_score, k=2, mode="block_gap")
    assert logit.item() == 2.0
    assert 0.0 < probability.item() < 1.0
    assert gap.item() == 1.5


def test_wilson_interval_and_prompt_hash_are_deterministic():
    lower, upper = wilson_interval(600, 600)
    assert lower > 0.99
    assert upper == pytest.approx(1.0)
    assert text_set_sha256(["a", "b"]) == text_set_sha256(["a", "b"])
    assert text_set_sha256(["a", "b"]) != text_set_sha256(["b", "a"])
