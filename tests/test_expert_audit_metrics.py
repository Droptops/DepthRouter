import copy
import itertools
import json

import numpy as np
import pytest

from depth_router.expert_audit_metrics import (
    candidate_combinations,
    candidate_state_metrics,
    numerical_control_summary,
    paired_context_difference,
    portfolio_gate,
    select_fixed_candidate,
    summarize_candidates,
)


def _matrix(states=8):
    return np.full((states, 28), 0.1), np.zeros((states, 28), dtype=bool)


def _summary(kl, agreement, selection, split="heldout", control_agreement=None):
    states = len(kl)
    return summarize_candidates(
        kl, agreement, np.ones_like(kl), np.arange(states) // 4,
        np.linspace(0, 3, states), fixed_selection=selection, split=split,
        control_kl=np.zeros(states),
        control_agreement=np.ones(states, dtype=bool)
        if control_agreement is None else control_agreement,
        bootstrap_repetitions=100,
    )


def test_every_six_of_eight_candidate_is_enumerated_once():
    candidates = candidate_combinations()
    assert len(candidates) == len(set(candidates)) == 28
    assert candidates == tuple(itertools.combinations(range(8), 6))
    assert all(len(set(candidate)) == 6 for candidate in candidates)
    kl, agreement = _matrix()
    for bad in (candidates[:-1], candidates[:-1] + (candidates[0],), candidates[::-1]):
        with pytest.raises(ValueError, match="28 canonical"):
            select_fixed_candidate(kl, agreement, candidate_experts=bad)
    with pytest.raises(ValueError, match="complete candidate matrix"):
        select_fixed_candidate(kl[:, :27], agreement[:, :27])


def test_fixed_candidate_selection_uses_coverage_then_kl_then_index():
    kl, agreement = _matrix()
    kl[:, 1:4] = 0.005
    agreement[:, 1:4] = True
    kl[:, 1] = 0.006
    selection = select_fixed_candidate(kl, agreement)
    assert selection["index"] == 2
    assert selection["development_coverage"] == 1
    assert selection["experts"] == list(candidate_combinations()[2])
    # Coverage dominates lower mean divergence if greedy agreement is worse.
    kl[:, 0] = 0
    agreement[:-1, 0] = True
    assert select_fixed_candidate(kl, agreement)["index"] == 2


def test_fixed_candidate_is_not_reselected_using_heldout_outputs():
    development_kl, development_agreement = _matrix()
    development_kl[:, 2] = 0.001
    development_agreement[:, 2] = True
    frozen = select_fixed_candidate(development_kl, development_agreement)
    heldout_kl, heldout_agreement = _matrix()
    heldout_kl[:, 3] = 0.001
    heldout_agreement[:, 3] = True
    summary = _summary(heldout_kl, heldout_agreement, frozen)
    assert summary["fixed_selection"]["index"] == 2
    assert summary["fixed_cheap_coverage"] == 0
    assert summary["cheap_coverage"] == 1
    assert summary["best_fixed"]["greedy_agreement"] == 0
    assert summary["oracle_cheap_only"]["greedy_agreement"] == 1
    with pytest.raises(ValueError, match="development only"):
        select_fixed_candidate(heldout_kl, heldout_agreement, split="heldout")
    forged = {**frozen, "selected_on": "heldout"}
    with pytest.raises(ValueError, match="development-only"):
        _summary(heldout_kl, heldout_agreement, forged)
    mismatched = {**frozen, "index": 3}
    with pytest.raises(ValueError, match="frozen index"):
        _summary(heldout_kl, heldout_agreement, mismatched)


def test_local_acceptance_is_strict_and_oracle_prefers_agreement_then_kl():
    kl, agreement = _matrix(4)
    kl[0, 0] = 0.01
    agreement[0, 0] = True  # Strict KL boundary fails local acceptability.
    kl[1, 0] = 0
    kl[1, 1] = 0.02
    agreement[1, 1] = True  # Agreement is prioritized over minimum KL.
    kl[2, 0] = 0.005  # Low KL with a different greedy token also fails.
    kl[3, 2] = kl[3, 3] = 0.009
    agreement[3, 2:4] = True
    selection = select_fixed_candidate(kl, agreement)
    state = candidate_state_metrics(kl, agreement, fixed_selection=selection)
    np.testing.assert_equal(state["cheap_coverage"], [0, 0, 0, 1])
    np.testing.assert_equal(state["oracle_candidate_index"], [0, 1, 0, 2])
    summary = _summary(kl, agreement, selection)
    assert summary["oracle_cheap_only"]["greedy_agreement"] == 0.75
    assert summary["oracle_with_dense"]["greedy_agreement"] == 1
    assert summary["oracle_with_dense"]["mean_kl"] == pytest.approx(0.009 / 4)
    assert summary["oracle_with_dense"]["ideal_active_width"] == 0.9375
    assert summary["oracle_with_dense"]["dense_fraction"] == 0.75
    assert summary["oracle_with_dense"]["mean_hidden_mse"] == 0.25


def test_portfolio_advantage_equals_fallback_width_savings_and_is_not_runtime():
    kl, agreement = _matrix()
    kl[:4, 0] = kl[4:, 1] = 0.001
    agreement[:4, 0] = agreement[4:, 1] = True
    selection = select_fixed_candidate(kl, agreement)
    summary = _summary(kl, agreement, selection)
    assert summary["cheap_coverage"] == 1
    assert summary["fixed_cheap_coverage"] == 0.5
    assert summary["portfolio_advantage"] == 0.125
    assert summary["oracle_with_dense"]["ideal_active_width"] == 0.75
    assert summary["fixed_oracle_with_dense"]["ideal_active_width"] == 0.875
    assert summary["engineering_gate"]["point_pass"]
    assert not summary["engineering_gate"]["statistically_resolved_pass"]
    assert not summary["oracle_is_deployable"]
    for arm in ("oracle_cheap_only", "oracle_with_dense", "fixed_oracle_with_dense"):
        assert summary[arm]["deployable"] is False
    assert summary["measured_latency"] is None
    assert summary["measured_dram_bytes"] is None
    json.dumps(summary, allow_nan=False)


def test_context_bootstrap_does_not_treat_correlated_positions_as_independent():
    # Two contexts, four completely correlated states each. Complete-context
    # resampling can produce all-zero/all-one samples; a token bootstrap is
    # substantially narrower and would not reproduce these interval endpoints.
    left = np.array([0.0] * 4 + [1.0] * 4)
    result = paired_context_difference(
        left, np.zeros_like(left), np.repeat(["a", "b"], 4), bootstrap_repetitions=200
    )
    assert result["estimate"] == 0.5
    assert result["lower"] == 0
    assert result["upper"] == 1
    assert result["n_contexts"] == 2
    assert result["n_states"] == 8


def test_cluster_bootstrap_retains_state_weighting_for_unequal_context_lengths():
    result = paired_context_difference(
        [0, 1, 1, 1], [0, 0, 0, 0], ["short", "long", "long", "long"],
        bootstrap_repetitions=200,
    )
    assert result["estimate"] == 0.75
    assert result["lower"] == 0
    assert result["upper"] == 1


def test_paired_seed_aggregation_uses_the_same_context_draw_for_every_seed():
    left = np.array([[0.0] * 4 + [1.0] * 4, [1.0] * 4 + [0.0] * 4])
    right = np.zeros_like(left)
    result = paired_context_difference(
        left, right, np.repeat(["a", "b"], 4), bootstrap_repetitions=200
    )
    assert result["n_paired_seeds"] == 2
    assert result["estimate"] == result["lower"] == result["upper"] == 0.5
    reversed_result = paired_context_difference(
        right, left, np.repeat(["a", "b"], 4), bootstrap_repetitions=200
    )
    assert reversed_result["estimate"] == reversed_result["lower"] == -0.5


def test_all_eight_control_gate_blocks_interpretation_of_sparse_results():
    kl, agreement = _matrix()
    kl[:4, 0] = kl[4:, 1] = 0.001
    agreement[:4, 0] = agreement[4:, 1] = True
    selection = select_fixed_candidate(kl, agreement)
    summary = _summary(kl, agreement, selection, control_agreement=np.zeros(8, dtype=bool))
    assert summary["engineering_gate"]["coverage_point_pass"]
    assert summary["engineering_gate"]["portfolio_advantage_point_pass"]
    assert not summary["engineering_gate"]["numerically_interpretable"]
    assert not summary["engineering_gate"]["point_pass"]
    no_control = summarize_candidates(
        kl, agreement, np.zeros_like(kl), np.arange(8) // 4, np.ones(8),
        fixed_selection=selection, split="heldout", bootstrap_repetitions=10,
    )
    assert no_control["numerical_control"] is None
    assert not no_control["engineering_gate"]["point_pass"]
    control = numerical_control_summary(np.full(100, 0.01), np.ones(100), np.arange(100),
                                        bootstrap_repetitions=10)
    assert not control["point_pass"]


def test_cycle_gate_requires_every_split_and_reports_distinct_failure_causes():
    kl, agreement = _matrix()
    kl[:4, 0] = kl[4:, 1] = 0.001
    agreement[:4, 0] = agreement[4:, 1] = True
    selection = select_fixed_candidate(kl, agreement)
    summaries = {split: _summary(kl, agreement, selection, split)
                 for split in ("development", "heldout", "ood")}
    decision = portfolio_gate(summaries)
    assert decision["point_pass"]
    assert decision["router_training_allowed"]
    assert not decision["economic_savings_established"]
    assert not decision["statistically_resolved_pass"]
    del summaries["ood"]
    decision = portfolio_gate(summaries)
    assert "ood:missing_split" in decision["failure_reasons"]
    assert decision["stop_before_router_training"]
    bad_kl, bad_agreement = _matrix()
    summaries["ood"] = _summary(bad_kl, bad_agreement, selection, "ood")
    reasons = portfolio_gate(summaries)["failure_reasons"]
    assert "ood:insufficient_cheap_candidate_coverage" in reasons
    assert "ood:insufficient_advantage_over_fixed_subset" in reasons
    summaries["ood"] = copy.deepcopy(summaries["heldout"])
    with pytest.raises(ValueError, match="different split label"):
        portfolio_gate(summaries)


def test_gate_rejects_different_fixed_choices_between_splits():
    kl, agreement = _matrix()
    kl[:, 1] = 0.001
    agreement[:, 1] = True
    first = select_fixed_candidate(kl, agreement)
    summaries = {"development": _summary(kl, agreement, first, "development")}
    kl[:, 2] = 0.0005
    agreement[:, 2] = True
    second = select_fixed_candidate(kl, agreement)
    summaries["heldout"] = _summary(kl, agreement, second)
    with pytest.raises(ValueError, match="same development-frozen"):
        portfolio_gate(summaries)


def test_nonfinite_and_malformed_candidate_evidence_is_rejected():
    kl, agreement = _matrix()
    kl[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        select_fixed_candidate(kl, agreement)
    kl[0, 0] = -0.01
    with pytest.raises(ValueError, match="roundoff"):
        select_fixed_candidate(kl, agreement)
    kl[0, 0] = -1e-7  # Record harmless numerical cancellation without clipping.
    select_fixed_candidate(kl, agreement)
    invalid_agreement = agreement.astype(float)
    invalid_agreement[0, 0] = 0.5
    with pytest.raises(ValueError, match="boolean"):
        select_fixed_candidate(kl, invalid_agreement)
    with pytest.raises(ValueError, match="one entry per token state"):
        paired_context_difference([1, 2], [0, 0], ["only-one"])
    with pytest.raises(ValueError, match="positive"):
        paired_context_difference([1, 2], [0, 0], ["a", "b"], bootstrap_repetitions=0)
