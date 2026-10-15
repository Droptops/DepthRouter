import copy
import itertools
import json

import numpy as np
import pytest

from depth_router.expert_granularity import (
    array_sha256,
    candidate_combinations,
    candidate_schema_sha256,
    cycle_gate,
    matrix_arrays,
    original_candidate_indices,
    paired_coverage_difference,
    paired_portfolio_difference,
    partition_indices,
    portfolio_coverage,
    portfolio_state_metrics,
    select_calibration_portfolio,
    select_development_fixed,
    split_parent_experts,
    summarize_portfolio,
    validate_refinement,
    verify_frozen_selection,
)


def parent_labels():
    return np.arange(4864, dtype=np.int64) // 608


def matrix(states=8, *, ids=None, split="development"):
    ids = np.arange(1820, dtype=np.int64) if ids is None else np.asarray(ids, dtype=np.int64)
    child = split_parent_experts(parent_labels(), seed=17)
    return {
        "kl": np.full((states, len(ids)), 0.1),
        "agreement": np.zeros((states, len(ids)), dtype=bool),
        "candidate_ids": ids, "selected_experts": np.asarray(candidate_combinations())[ids],
        "partition": child, "partition_sha256": array_sha256(child),
        "context_ids": np.arange(states) // 4, "positions": np.tile([1, 3, 7, 11], states // 4),
        "control_kl": np.zeros(states), "control_agreement": np.ones(states, dtype=bool),
        "split": split,
    }


def make_acceptable(raw, rows, column, kl=0.001):
    raw["kl"][rows, column] = kl
    raw["agreement"][rows, column] = True


@pytest.mark.parametrize("seed", [17, 29, 43])
def test_balanced_halves_refine_frozen_parent_exactly(seed):
    parent = parent_labels()
    child = split_parent_experts(parent, seed=seed)
    np.testing.assert_array_equal(child // 2, parent)
    np.testing.assert_array_equal(np.bincount(child), np.full(16, 304))
    np.testing.assert_array_equal(child, split_parent_experts(parent, seed=seed))
    audit = validate_refinement(parent, child)
    assert audit["active_neurons"] == 3648
    assert audit["partition_sha256"] == array_sha256(child)
    assert audit["exact_parent_refinement"]


def test_refinement_seeds_are_distinct_and_invalid_partitions_fail():
    parent = parent_labels()
    assert len({array_sha256(split_parent_experts(parent, seed=seed))
                for seed in [17, 29, 43]}) == 3
    bad_parent = parent.copy()
    bad_parent[0] = 1
    with pytest.raises(ValueError, match="equal experts"):
        split_parent_experts(bad_parent, seed=17)
    with pytest.raises(ValueError, match="integer"):
        split_parent_experts(parent.astype(float), seed=17)
    child = split_parent_experts(parent, seed=17)
    child[[0, 609]] = child[[609, 0]]
    with pytest.raises(ValueError, match="refine their frozen parent"):
        validate_refinement(parent, child)


def test_every_12_of_16_combination_is_unique_lexicographic_and_75_percent():
    combos = candidate_combinations()
    assert len(combos) == len(set(combos)) == 1820
    assert combos == tuple(itertools.combinations(range(16), 12))
    assert len(candidate_schema_sha256()) == 64
    child = split_parent_experts(parent_labels(), seed=29)
    for experts in combos:
        indices = partition_indices(child, experts)
        assert len(indices) == 3648
        assert np.all(np.diff(indices) > 0)
        assert set(child[indices]) == set(experts)
    with pytest.raises(ValueError, match="unique"):
        partition_indices(child, [1, 1])


@pytest.mark.parametrize("seed", [17, 29, 43])
def test_old_28_masks_are_exact_physical_neuron_sets_in_each_refinement(seed):
    parent = parent_labels()
    child = split_parent_experts(parent, seed=seed)
    mapped = original_candidate_indices()
    assert mapped.shape == (28,) and len(set(mapped)) == 28
    for global_id, old_experts in zip(mapped, itertools.combinations(range(8), 6)):
        expected = np.flatnonzero(np.isin(parent, old_experts))
        actual = partition_indices(child, candidate_combinations()[global_id])
        np.testing.assert_array_equal(actual, expected)


def test_calibration_greedy_prioritizes_marginal_coverage_then_mean_kl_then_global_id():
    raw = matrix(8, split="calibration")
    make_acceptable(raw, slice(0, 4), 100, 0.005)
    raw["kl"][4:, 100] = 0.2
    make_acceptable(raw, slice(0, 4), 80, 0.001)
    make_acceptable(raw, slice(4, 8), 40, 0.001)
    raw["kl"][:4, 40] = 0.2  # Four-new-state tie loses to 80 on mean KL.
    make_acceptable(raw, slice(0, 3), 10, 0.0001)
    raw["kl"][3:, 10] = 0.0001  # Lower mean KL cannot beat four-state coverage.
    selected = select_calibration_portfolio(raw)
    assert selected["selection_order"][:4] == [80, 40, 10, 0]
    assert [row["newly_covered_states"] for row in selected["steps"][:4]] == [4, 4, 0, 0]
    assert selected["calibration_coverage"] == 1
    assert len(selected["candidate_ids"]) == len(set(selected["candidate_ids"])) == 28
    assert selected["candidate_ids"] == sorted(selected["selection_order"])
    assert selected["selected_on"] == "calibration"
    verify_frozen_selection(selected, selected_on="calibration")
    json.dumps(selected, allow_nan=False)


def test_calibration_still_selects_exactly_28_after_zero_gain_and_resolves_global_ties():
    raw = matrix(split="calibration")
    first = select_calibration_portfolio(raw)
    assert first["selection_order"] == list(range(28))
    assert first["calibration_coverage"] == 0
    permutation = np.random.default_rng(29).permutation(1820)
    shuffled = {**raw, "kl": raw["kl"][:, permutation],
                "agreement": raw["agreement"][:, permutation],
                "candidate_ids": raw["candidate_ids"][permutation],
                "selected_experts": raw["selected_experts"][permutation]}
    second = select_calibration_portfolio(shuffled)
    assert second["selection_order"] == first["selection_order"]
    assert second["matrix_candidate_ids_sha256"] != first["matrix_candidate_ids_sha256"]


def test_portfolio_selection_rejects_wrong_stage_incomplete_search_and_size_tuning():
    raw = matrix(split="calibration")
    with pytest.raises(ValueError, match="calibration only"):
        select_calibration_portfolio(raw, split="heldout_matched")
    raw["split"] = "heldout_ood"
    with pytest.raises(ValueError, match="split provenance"):
        select_calibration_portfolio(raw)
    with pytest.raises(ValueError, match="all 1820"):
        select_calibration_portfolio(matrix(ids=range(28), split="calibration"))
    with pytest.raises(ValueError, match="exactly 28"):
        select_calibration_portfolio(matrix(split="calibration"), size=27)


def test_local_acceptance_requires_same_candidate_to_meet_both_strict_conditions():
    raw = matrix(ids=[0, 1, 2])
    make_acceptable(raw, 0, 0, kl=0.01)  # Strict boundary fails.
    raw["kl"][1, 0] = 0.001  # Low KL but wrong token.
    make_acceptable(raw, 1, 1, kl=0.02)  # Same token but high KL.
    make_acceptable(raw, 2, 2, kl=0.009)
    np.testing.assert_array_equal(portfolio_coverage(raw, [0, 1, 2]),
                                  [False, False, True, False, False, False, False, False])


def test_fixed_candidate_is_chosen_on_fresh_development_and_not_rechosen_on_heldout():
    raw = matrix(ids=[0, 1, 2])
    make_acceptable(raw, slice(0, 4), 0, kl=0.009)
    make_acceptable(raw, slice(0, 4), 1, kl=0.001)
    make_acceptable(raw, slice(0, 4), 2, kl=0.001)
    fixed = select_development_fixed(raw, [2, 1, 0])
    assert fixed["candidate_id"] == 1
    assert fixed["portfolio_candidate_ids"] == [0, 1, 2]
    heldout = copy.deepcopy(raw)
    heldout["split"] = "heldout_matched"
    make_acceptable(heldout, slice(None), 0)
    states = portfolio_state_metrics(heldout, [0, 1, 2], fixed)
    assert states["cheap_coverage"].mean() == 1
    assert states["fixed_cheap_coverage"].mean() == 0.5
    with pytest.raises(ValueError, match="development only"):
        select_development_fixed(heldout, [0, 1, 2], split="heldout_matched")


def test_frozen_choices_are_sealed_and_cannot_move_between_portfolios_or_partitions():
    raw = matrix(ids=[0, 1, 2])
    fixed = select_development_fixed(raw, [0, 1])
    modified = copy.deepcopy(fixed)
    modified["candidate_id"] = 2
    with pytest.raises(ValueError, match="modified"):
        verify_frozen_selection(modified, selected_on="development")
    with pytest.raises(ValueError, match="different partition or frozen portfolio"):
        portfolio_state_metrics(raw, [1, 2], fixed)
    other = copy.deepcopy(raw)
    other["partition"] = split_parent_experts(parent_labels(), seed=29)
    other["partition_sha256"] = array_sha256(other["partition"])
    with pytest.raises(ValueError, match="different partition or frozen portfolio"):
        portfolio_state_metrics(other, [0, 1], fixed)


def test_matrix_schema_rejects_duplicate_ids_wrong_experts_or_wrong_partition_hash():
    raw = matrix(ids=[3, 7])
    raw["candidate_ids"] = np.array([3, 3])
    with pytest.raises(ValueError, match="unique canonical"):
        matrix_arrays(raw)
    raw = matrix(ids=[3, 7])
    raw["selected_experts"] = raw["selected_experts"][::-1]
    with pytest.raises(ValueError, match="selected_experts"):
        matrix_arrays(raw)
    raw = matrix(ids=[3, 7])
    raw["partition_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="partition hash"):
        matrix_arrays(raw)


def test_raw_token_matrix_matches_explicit_agreement():
    raw = matrix(ids=[0, 7])
    make_acceptable(raw, [1, 3], 1)
    converted = {key: value for key, value in raw.items() if key != "agreement"}
    converted["dense_token"] = np.ones(8, dtype=np.int64)
    converted["candidate_token"] = np.where(raw["agreement"], 1, 2)
    np.testing.assert_array_equal(portfolio_coverage(converted, [0, 7]),
                                  portfolio_coverage(raw, [0, 7]))


def test_incomplete_or_partially_measured_archive_is_never_selected():
    raw = matrix(split="calibration")
    raw["complete"] = np.array(False)
    with pytest.raises(ValueError, match="incomplete"):
        select_calibration_portfolio(raw)
    raw["complete"] = np.array(True)
    raw["measured"] = np.ones_like(raw["agreement"])
    raw["measured"][0, 0] = False
    with pytest.raises(ValueError, match="every declared measurement"):
        select_calibration_portfolio(raw)


def test_generic_summary_keeps_numerical_and_engineering_gates_separate():
    raw = matrix(ids=[0, 1])
    make_acceptable(raw, slice(0, 4), 0)
    make_acceptable(raw, slice(4, 8), 1)
    fixed = select_development_fixed(raw, [0, 1])
    result = summarize_portfolio(raw, [0, 1], fixed, split="development",
                                 bootstrap_repetitions=100)
    assert result["cheap_coverage"] == 1
    assert result["fixed_cheap_coverage"] == 0.5
    assert result["portfolio_advantage"] == 0.125
    assert result["ideal_active_width"] == 0.75
    assert result["engineering_gate"]["point_pass"]
    assert not result["engineering_gate"]["statistically_resolved_pass"]
    assert result["confidence_intervals"]["portfolio_advantage"]["lower"] == 0
    assert result["numerical_control"]["point_pass"]
    assert result["measured_latency"] is None and result["measured_dram_bytes"] is None
    assert not result["oracle_is_deployable"]
    json.dumps(result, allow_nan=False)
    raw["control_kl"][:] = 0.01
    bad = summarize_portfolio(raw, [0, 1], fixed, split="development",
                              bootstrap_repetitions=20)
    assert bad["engineering_gate"]["coverage_point_pass"]
    assert not bad["engineering_gate"]["point_pass"]


def test_missing_numerical_control_fails_closed_and_wrong_context_order_is_rejected():
    raw = matrix(ids=[0, 1])
    fixed = select_development_fixed(raw, [0, 1])
    del raw["control_kl"]
    result = summarize_portfolio(raw, [0, 1], fixed, split="development",
                                 bootstrap_repetitions=20)
    assert not result["numerical_control"]["point_pass"]
    with pytest.raises(ValueError, match="context IDs"):
        summarize_portfolio(raw, [0, 1], fixed, raw["context_ids"][::-1], split="development",
                             bootstrap_repetitions=20)


def test_rescue_loss_and_net_are_paired_by_context_not_independent_tokens():
    selected = np.array([True, True, False, False])
    original = np.array([True, False, True, False])
    result = paired_coverage_difference(selected, original, [0, 0, 1, 1],
                                        bootstrap_repetitions=200)
    assert result["state_counts"] == {"rescue": 1, "loss": 1, "net": 0, "states_per_seed": 4}
    assert result["rescue"]["estimate"] == result["loss"]["estimate"] == 0.25
    assert result["net_coverage"]["estimate"] == 0
    assert result["net_coverage"]["lower"] == -0.5
    assert result["net_coverage"]["upper"] == 0.5
    assert result["ideal_width_improvement"]["estimate"] == 0


def test_seedmean_is_taken_within_paired_whole_context_draws():
    selected = np.array([[1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 1, 1]])
    original = np.zeros_like(selected)
    result = paired_coverage_difference(selected, original, [0, 0, 1, 1],
                                        bootstrap_repetitions=100)
    interval = result["net_coverage"]
    assert interval["estimate"] == pytest.approx(2 / 3)
    assert interval["lower"] == pytest.approx(2 / 3)
    assert interval["upper"] == pytest.approx(2 / 3)
    assert interval["n_paired_seeds"] == 3
    assert result["state_counts"]["rescue"] == [2, 2, 4]


def test_paired_matrix_wrapper_checks_context_and_position_alignment():
    raw = matrix(ids=[0, 1])
    make_acceptable(raw, [0, 1], 0)
    make_acceptable(raw, [2, 3, 4], 1)
    result = paired_portfolio_difference(raw, [1], raw, [0], bootstrap_repetitions=20)
    assert result["state_counts"]["rescue"] == 3 and result["state_counts"]["loss"] == 2
    other = copy.deepcopy(raw)
    other["positions"] = other["positions"][::-1]
    with pytest.raises(ValueError, match="positions"):
        paired_portfolio_difference(raw, [1], other, [0], bootstrap_repetitions=20)


def test_full_universe_coverage_cannot_be_lower_than_its_original_28_subset():
    raw = matrix()
    old = original_candidate_indices()
    make_acceptable(raw, [0, 1, 2], int(old[0]))
    new_id = next(index for index in range(1820) if index not in old)
    make_acceptable(raw, [3, 4], new_id)
    original = portfolio_coverage(raw, old)
    full = portfolio_coverage(raw, np.arange(1820))
    assert np.all(full >= original) and int(full.sum() - original.sum()) == 2


def test_cycle_requires_all_selected_portfolios_and_resolved_seedmean_ood_gain():
    summary = {"candidate_count": 28,
               "engineering_gate": {"point_pass": True, "statistically_resolved_pass": True}}
    results = {str(seed): {split: copy.deepcopy(summary)
                          for split in ["development", "heldout_matched", "heldout_ood"]}
               for seed in [17, 29, 43]}
    paired = {"net_coverage": {"estimate": 0.05, "lower": 0.01, "upper": 0.09,
                               "n_paired_seeds": 3}}
    assert cycle_gate(results, paired)["point_pass"]
    results["43"]["heldout_ood"]["engineering_gate"]["point_pass"] = False
    stopped = cycle_gate(results, paired)
    assert not stopped["point_pass"]
    assert stopped["individual_selected_portfolios"]["17"]["point_pass"]
    results["43"]["heldout_ood"]["engineering_gate"]["point_pass"] = True
    paired["net_coverage"]["lower"] = 0
    assert not cycle_gate(results, paired)["point_pass"]
    results["17"]["heldout_ood"]["candidate_count"] = 1820
    with pytest.raises(ValueError, match="selected 28-mask"):
        cycle_gate(results, paired)


def test_cycle_rejects_missing_seeds_splits_and_wrong_aggregation():
    summary = {"candidate_count": 28,
               "engineering_gate": {"point_pass": True, "statistically_resolved_pass": False}}
    results = {seed: {split: copy.deepcopy(summary)
                     for split in ["development", "heldout_matched", "heldout_ood"]}
               for seed in [17, 29, 43]}
    paired = {"net_coverage": {"estimate": 0.05, "lower": 0.01, "upper": 0.09,
                               "n_paired_seeds": 1}}
    with pytest.raises(ValueError, match="all preregistered seeds"):
        cycle_gate(results, paired)
    with pytest.raises(ValueError, match="all three"):
        cycle_gate({17: results[17]}, paired)
    del results[29]["development"]
    with pytest.raises(ValueError, match="both held-out"):
        cycle_gate(results, paired)
