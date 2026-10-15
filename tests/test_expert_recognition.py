import copy
import json

import numpy as np
import pytest
import torch

from depth_router.expert_recognition import (
    ARCHITECTURES,
    DENSE_ACTION,
    Selector,
    acceptable_labels,
    actions_from_scores,
    array_sha256,
    load_checkpoint,
    matrix_arrays,
    oracle_actions,
    paired_policy_difference,
    policy_metrics,
    policy_state_metrics,
    predict_scores,
    random_actions,
    repeatability_gate,
    save_checkpoint,
    select_policy,
    state_sha256,
    train_selector_pair,
    verify_frozen_choice,
)


@pytest.fixture(autouse=True)
def small_cpu_thread_pool():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def matrix(states=8):
    return {"kl": np.full((states, 28), 0.1),
            "agreement": np.zeros((states, 28), dtype=bool)}


def features_and_labels():
    x = torch.arange(48, dtype=torch.float32).reshape(8, 6) / 48
    x[:, 0] = 1
    y = torch.zeros((8, 28), dtype=torch.bool)
    y[:4, :3] = True
    y[4:, 3:7] = True
    return x, y


def test_labels_keep_all_acceptable_and_strict_local_boundary():
    raw = matrix(4)
    raw["kl"][0, :3] = [0, 0.009, 0.01]
    raw["agreement"][0, :3] = True
    raw["kl"][1, 0] = 0.001
    labels = acceptable_labels(raw)
    np.testing.assert_array_equal(np.flatnonzero(labels[0]), [0, 1])
    assert not labels[1].any()
    converted = {"kl": raw["kl"], "dense_token": np.ones(4),
                 "candidate_token": np.where(raw["agreement"], 1, 2)}
    np.testing.assert_array_equal(acceptable_labels(converted), labels)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_pair_has_shared_initial_backbone_normalization_and_schedule(architecture):
    x, labels = features_and_labels()
    portfolio, binary, metadata = train_selector_pair(
        x, labels, 2, architecture=architecture, seed=17, steps=3, batch_size=4)
    assert portfolio.outputs == 28 and binary.outputs == 1
    torch.testing.assert_close(portfolio.mean, x.mean(0))
    torch.testing.assert_close(portfolio.scale, x.std(0, unbiased=False).clamp_min(1e-5))
    torch.testing.assert_close(portfolio.mean, binary.mean)
    assert portfolio.scale[0] == pytest.approx(1e-5)
    assert metadata["parameter_counts"]["portfolio"] > metadata["parameter_counts"]["binary"]
    assert metadata["final_state_sha256"]["portfolio"] == state_sha256(portfolio)
    assert len(metadata["losses"]["portfolio"]) == len(metadata["losses"]["binary"]) == 3
    assert metadata["selected_checkpoint"] == "final_iterate_no_early_stopping"
    assert portfolio.mean.dtype == torch.float32
    assert predict_scores(portfolio, x.to(torch.bfloat16)).shape == (8, 28)
    json.dumps(metadata, allow_nan=False)


def test_training_is_reproducible_and_architectures_share_schedule():
    x, labels = features_and_labels()
    records = []
    for architecture in ("linear", "linear", "mlp_64"):
        _, _, record = train_selector_pair(
            x, labels, 2, architecture=architecture, seed=29, steps=2, batch_size=4)
        records.append(record)
    assert records[0] == records[1]
    assert records[0]["shared_schedule_sha256"] == records[2]["shared_schedule_sha256"]
    assert records[0]["normalization_sha256"] == records[2]["normalization_sha256"]


def test_step_callback_bounds_both_models_without_hidden_extra_updates():
    x, labels = features_and_labels()
    seen = []
    train_selector_pair(x, labels, 0, architecture="linear", seed=17,
                        steps=3, batch_size=4, step_callback=seen.append)
    assert seen == [0, 1, 2]
    with pytest.raises(ValueError, match="positive"):
        train_selector_pair(x, labels, 0, architecture="linear", seed=17, learning_rate=np.nan)


def test_actions_dense_sentinel_and_exact_one_threshold():
    scores = np.zeros((4, 28))
    scores[:, 3] = [0.7, 0.9, 0.99, 1]
    np.testing.assert_array_equal(actions_from_scores(scores, 0.9), [28, 3, 3, 3])
    np.testing.assert_array_equal(actions_from_scores(scores, 1), np.full(4, 28))
    np.testing.assert_array_equal(actions_from_scores(scores[:, 3:4], 0.9, fixed_index=7),
                                  [28, 7, 7, 7])
    with pytest.raises(ValueError, match="binary"):
        actions_from_scores(scores[:, :1], 0.5)
    with pytest.raises(ValueError, match="portfolio"):
        actions_from_scores(scores, 0.5, fixed_index=7)
    with pytest.raises(ValueError, match="probabilities"):
        actions_from_scores(scores + 1, 0.5)


def test_runtime_is_x_only_and_matches_numpy_decisions(monkeypatch):
    model = Selector(torch.zeros(6), torch.ones(6), "mlp_64", 28).eval()
    x = torch.rand(5, 1, 6).to(torch.bfloat16)
    scores = model.scores(x).detach().numpy().reshape(5, 28)
    expected = actions_from_scores(scores, 0.5)
    monkeypatch.setattr("depth_router.expert_recognition.matrix_arrays",
                        lambda *args: pytest.fail("runtime consulted teacher matrix"))
    np.testing.assert_array_equal(model.actions(x, 0.5).detach().numpy().reshape(5), expected)
    monkeypatch.setattr(model, "scores", lambda *args: pytest.fail("dense sentinel ran router"))
    assert torch.all(model.actions(x, 1.0) == DENSE_ACTION)


def test_policy_metrics_use_dense_exact_and_actual_chosen_fidelity():
    raw = matrix(4)
    raw["kl"][:, 2] = [0.005, 0.015, 0.001, 0.02]
    raw["agreement"][:, 2] = [True, True, False, False]
    actions = np.array([2, 2, 2, 28])
    result = policy_metrics(actions, raw, ["a", "a", "b", "b"], bootstrap_repetitions=50)
    assert result["greedy_agreement"] == 0.75
    assert result["mean_kl"] == pytest.approx(0.021 / 4)
    assert result["cheap_fraction"] == 0.75 and result["dense_fraction"] == 0.25
    assert result["false_cheap_fraction"] == 0.5
    assert result["false_cheap_among_cheap"] == pytest.approx(2 / 3)
    assert result["ideal_active_width"] == 0.8125
    assert not result["quality_gate"]["point_pass"]
    assert result["measured_latency"] is None and result["measured_dram_bytes"] is None
    assert not result["confidence_is_certificate"]
    json.dumps(result, allow_nan=False)


def test_all_dense_quality_pass_is_not_cheap_recognition_success():
    result = policy_metrics(np.full(8, 28), matrix(), np.arange(8) // 4,
                            bootstrap_repetitions=30)
    assert result["quality_gate"]["point_pass"]
    assert not result["has_nonzero_cheap_use"]
    assert result["false_cheap_among_cheap"] is None
    conditional = result["intervals"]["false_cheap_among_cheap"]
    assert conditional["lower"] is None and conditional["upper"] is None
    assert conditional["undefined_zero_cheap_resamples"] == 30
    json.dumps(result, allow_nan=False)


def test_context_bootstrap_retains_all_positions_and_paired_alignment():
    raw = matrix()
    raw["kl"][:, 0] = 0.001
    raw["agreement"][:, 0] = True
    lhs = np.array([0] * 4 + [28] * 4)
    rhs = np.full(8, 28)
    result = paired_policy_difference(lhs, rhs, raw, ["a"] * 4 + ["b"] * 4,
                                      bootstrap_repetitions=200)
    cheap = result["cheap_fraction"]
    assert cheap["estimate"] == 0.5 and cheap["lower"] == 0 and cheap["upper"] == 1
    assert cheap["n_contexts"] == 2
    assert result["ideal_active_width"]["estimate"] == -0.125
    assert result["greedy_agreement"]["estimate"] == 0


def test_paired_difference_averages_fixed_seeds_within_each_context_draw():
    raw = matrix()
    raw["kl"][:, 0] = 0.001
    raw["agreement"][:, 0] = True
    left = np.array([[0] * 4 + [28] * 4, [28] * 4 + [0] * 4, [0] * 8])
    right = np.full((3, 8), 28)
    difference = paired_policy_difference(left, right, raw, np.arange(8) // 4,
                                          bootstrap_repetitions=100)
    cheap = difference["cheap_fraction"]
    assert cheap["estimate"] == pytest.approx(2 / 3)
    assert cheap["lower"] == pytest.approx(2 / 3)
    assert cheap["upper"] == pytest.approx(2 / 3)
    assert cheap["n_paired_seeds"] == 3
    assert cheap["seed_uncertainty"] == "conditional_on_the_evaluated_seed_set"


def test_selection_maximizes_development_cheap_rate_then_small_arch_then_threshold():
    raw = matrix()
    raw["kl"][:, 3] = 0.001
    raw["agreement"][:, 3] = True
    scores = np.zeros((8, 28))
    scores[:, 3] = 0.99
    result = select_policy({"linear": scores, "mlp_64": scores}, raw, np.arange(8))
    assert result["architecture"] == "linear" and result["threshold"] == 0.99
    assert result["development_metrics"]["cheap_fraction"] == 1
    assert len(result["grid"]) == 18
    verify_frozen_choice(result)
    modified = copy.deepcopy(result)
    modified["threshold"] = 0.5
    with pytest.raises(ValueError, match="modified"):
        verify_frozen_choice(modified)
    with pytest.raises(ValueError, match="development only"):
        select_policy({"linear": scores, "mlp_64": scores}, raw, np.arange(8), split="heldout")


def test_selection_never_chooses_bad_quality_despite_cheaper_execution():
    raw = matrix(100)
    raw["kl"][:, 0] = 0.001
    raw["agreement"][:, 0] = True
    raw["agreement"][0:2, 0] = False
    linear = np.zeros((100, 28))
    linear[:, 0] = 0.99
    mlp = linear.copy()
    mlp[0:2, 0] = 0.51
    choice = select_policy({"linear": linear, "mlp_64": mlp}, raw, np.arange(100))
    assert choice["architecture"] == "mlp_64"
    assert choice["threshold"] == 0.99
    assert choice["development_metrics"]["cheap_fraction"] == 0.98


def test_binary_selection_uses_frozen_subset_without_portfolio_search():
    raw = matrix()
    raw["kl"][:, 4] = 0.001
    raw["agreement"][:, 4] = True
    scores = np.full((8, 1), 0.98)
    choice = select_policy({name: scores for name in ARCHITECTURES}, raw,
                           np.arange(8), fixed_index=4)
    assert choice["fixed_index"] == 4 and choice["threshold"] == 0.98
    with pytest.raises(ValueError, match="preregistered"):
        select_policy({name: scores for name in ARCHITECTURES}, raw,
                       np.arange(8), fixed_index=4, thresholds=(0.5, 1.0))


def test_random_diagnostic_has_exactly_matched_cheap_count_and_optional_positions():
    reference = np.array([28, 1, 2, 28, 28, 3] * 10)
    first = random_actions(reference, seed=17)
    np.testing.assert_array_equal(first == 28, reference == 28)
    np.testing.assert_array_equal(first, random_actions(reference, seed=17))
    shuffled = random_actions(reference, seed=17, match_positions=False)
    assert np.sum(shuffled != 28) == np.sum(reference != 28)
    assert not np.array_equal(shuffled == 28, reference == 28)


def test_oracles_are_locally_acceptable_and_dense_when_none_exists():
    raw = matrix(4)
    raw["kl"][0, :2] = [0.008, 0.002]
    raw["agreement"][0, :2] = True
    raw["kl"][1, 2] = 0.01
    raw["agreement"][1, 2] = True
    raw["kl"][2, 1] = 0.001  # Agreement false: not acceptable.
    raw["kl"][3, 0] = 0.009
    raw["agreement"][3, 0] = True
    np.testing.assert_array_equal(oracle_actions(raw), [1, 28, 28, 0])
    np.testing.assert_array_equal(oracle_actions(raw, fixed_index=0), [0, 28, 28, 0])


def test_checkpoint_roundtrip_hash_and_overwrite_protection(tmp_path):
    model = Selector(torch.zeros(3), torch.ones(3), "linear", 28)
    destination = tmp_path / "frozen.pt"
    save_checkpoint(destination, model, {"source": "calibration"})
    restored = load_checkpoint(destination)
    assert state_sha256(restored) == state_sha256(model)
    with pytest.raises(FileExistsError, match="overwrite"):
        save_checkpoint(destination, model, {})
    payload = torch.load(destination, weights_only=True)
    payload["state_dict"]["head.bias"][0] += 1
    torch.save(payload, destination)
    with pytest.raises(ValueError, match="hash"):
        load_checkpoint(destination)


def test_repeatability_distinguishes_one_seed_failure_and_all_dense():
    passing = {"quality_gate": {"point_pass": True, "statistically_resolved_pass": True},
               "has_nonzero_cheap_use": True}
    records = {seed: {name: copy.deepcopy(passing)
                      for name in ("development", "heldout_matched", "heldout_ood")}
               for seed in (17, 29, 43)}
    assert repeatability_gate(records)["all_seed_quality_point_pass"]
    records[43]["heldout_ood"]["quality_gate"]["point_pass"] = False
    result = repeatability_gate(records)
    assert not result["all_seed_quality_point_pass"]
    assert result["individual_seeds"]["17"]["quality_point_pass"]
    records[43]["heldout_ood"]["quality_gate"]["point_pass"] = True
    records[29]["development"]["has_nonzero_cheap_use"] = False
    result = repeatability_gate(records)
    assert result["all_seed_quality_point_pass"] and not result["all_seed_nonzero_cheap_use"]
    with pytest.raises(ValueError, match="every preregistered seed"):
        repeatability_gate({17: records[17]})


def test_matrix_and_action_validation_and_reproducible_object_hash():
    raw = matrix()
    with pytest.raises(ValueError, match="integer"):
        policy_state_metrics(np.full(8, 28.0), raw)
    raw["kl"][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        matrix_arrays(raw)
    assert array_sha256(np.array(["a", "b"], dtype=object)) == array_sha256(
        np.array(["a", "b"], dtype=object))
    assert array_sha256(np.array(["a", "b"])) != array_sha256(np.array(["b", "a"]))
