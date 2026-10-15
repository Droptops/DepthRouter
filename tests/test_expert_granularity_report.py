"""CPU-only synthetic evidence: post-run reporting must not invent promotion."""

import copy
import hashlib
import importlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from depth_router.expert_granularity import (
    array_sha256,
    candidate_combinations,
    cycle_gate,
    original_candidate_indices,
    paired_coverage_difference,
    paired_portfolio_difference,
    partition_indices,
    portfolio_coverage,
    select_calibration_portfolio,
    select_development_fixed,
    summarize_portfolio,
)

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("calibration", "development", "heldout_matched", "heldout_ood")
SEEDS = (17, 29, 43)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, sort_keys=True, indent=2) + "\n").encode())


def read_json(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def altered(path, transform):
    original = path.read_bytes()
    try:
        write_json(path, transform(read_json(path)))
        yield
    finally:
        path.write_bytes(original)


def synthetic_matrix(records, labels, ids, control_order, split):
    combinations = np.asarray(candidate_combinations(), dtype=np.int64)
    ids = np.asarray(ids, dtype=np.int64)
    count = len(records)
    matrix = {
        "kl": np.full((count, len(ids)), 0.1, dtype=np.float32),
        "hidden_mse": np.zeros((count, len(ids)), dtype=np.float32),
        "candidate_margin": np.ones((count, len(ids)), dtype=np.float32),
        "max_abs_logit_error": np.zeros((count, len(ids)), dtype=np.float32),
        "candidate_token": np.zeros((count, len(ids)), dtype=np.int32),
        "measured": np.ones((count, len(ids)), dtype=bool),
        "control_measured": np.ones(count, dtype=bool),
        "control_kl": np.zeros(count, dtype=np.float32),
        "control_token": np.zeros(count, dtype=np.int32),
        "permuted_control_kl": np.full(count, 0.0002, dtype=np.float32),
        "permuted_control_token": np.zeros(count, dtype=np.int32),
        "complete": np.asarray(True), "split": np.asarray(split),
        "candidate_ids": ids, "selected_experts": combinations[ids],
        "partition": labels, "partition_sha256": np.asarray(array_sha256(labels)),
        "control_order": control_order,
        "neuron_indices": np.stack([partition_indices(labels, combinations[index]) for index in ids]),
    }
    for field, record_name in (("context_ids", "id"), ("positions", "position"),
                               ("groups", "group"), ("slots", "slot"),
                               ("dense_token", "dense_token"), ("dense_margin", "dense_margin")):
        matrix[field] = np.asarray([row[record_name] for row in records])
    old_ids = original_candidate_indices()
    new_id = np.setdiff1d(np.arange(1820), old_ids)[0]
    # Same paired states: original covers even positions; one new mask rescues odd positions.
    for candidate_id, rows in ((old_ids[0], slice(0, None, 2)), (new_id, slice(1, None, 2))):
        column = np.flatnonzero(ids == candidate_id)
        if len(column):
            matrix["kl"][rows, column[0]] = 0.001
    return matrix


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory):
    # These are explicitly fabricated scalar measurements, never Qwen inference.
    paths = [str(ROOT / "scripts"), str(ROOT / "experiments")]
    sys.path[:0] = paths
    verifier = importlib.import_module("verify_expert_granularity")
    report = importlib.import_module("expert_granularity_report")
    experiment = importlib.import_module("expert_granularity")
    evaluator = importlib.import_module("expert_granularity_evaluation")
    directory = tmp_path_factory.mktemp("granularity-postprocessing")
    repo, run = directory / "repo", directory / "run"
    run.mkdir()
    config = read_json(ROOT / "configs/expert_granularity.json")
    sources = verifier.REQUIRED_SOURCES | {config["manifest"], config["frozen_partition"]}
    for name in sources:
        for destination in (repo / name, run / "registered_sources" / name):
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((ROOT / name).read_bytes())
    parent = np.asarray(read_json(repo / config["frozen_partition"]), dtype=np.int64)
    partitions = experiment.frozen_partitions(parent, list(SEEDS))
    write_json(run / "partitions.json", partitions)
    write_json(run / "verified_data_audit.json", {"synthetic_test_only": True})
    write_json(run / "quarantined_confirmation.json", read_json(
        ROOT / "data/expert_granularity/quarantined_confirmation.json"))
    registration = {
        "utc": "synthetic-test", "config": config,
        "source_sha256": {name: sha(repo / name) for name in sources},
        "frozen_artifact_sha256": {name: sha(run / name) for name in
            ("partitions.json", "verified_data_audit.json", "quarantined_confirmation.json")},
        "before_model_load_fit_and_fresh_outputs": True, "prior_negative_results_unchanged": True,
    }
    write_json(run / "preregistration.json", registration)
    manifest = [json.loads(line) for line in (repo / config["manifest"]).read_text().splitlines()]
    old_ids, all_ids = original_candidate_indices(), np.arange(1820)
    control_order = np.concatenate([np.flatnonzero(parent == index) for index in range(8)])
    metrics = {"original": {}, "selected": {str(seed): {} for seed in SEEDS},
               "full_oracle": {str(seed): {} for seed in SEEDS}}
    paired, inherited, portfolios = {"calibration": {"per_seed": {}}}, {}, {}
    fixed = {"original": None, "selected": {}, "full_oracle": {}}
    for split in SPLITS:
        sources = [row for row in manifest if row["split"] == split]
        records = []
        for index, source in enumerate(sources[:4]):
            for offset, position in enumerate((1, 3)):
                records.append({"id": source["id"], "task": source["task"],
                    "stratum": source["stratum"], "state_index": len(records), "split": split,
                    "position": position, "group": offset, "slot": index,
                    "prefix_token_ids": [1, 2, 3], "dense_token": 0, "dense_margin": 1.0})
        (run / f"states-{split}.jsonl").write_bytes(
            ("\n".join(json.dumps(row) for row in records) + "\n").encode())
        write_json(run / f"capture-{split}.json", {
            "contexts_requested": len(sources), "contexts_evaluated": 4, "states": 8,
            "zero_state_context_ids": [row["id"] for row in sources[4:]], "prompt_truncations": 0})
        original = synthetic_matrix(records, np.asarray(partitions["children"]["17"]),
                                    old_ids, control_order, split)
        evaluator.write_candidate_matrix(run / f"matrix-{split}-original.json", original)
        inherited[split], coverage = {}, []
        if split == "development":
            fixed["original"] = select_development_fixed(original, old_ids)
        if split != "calibration":
            metrics["original"][split] = summarize_portfolio(original, old_ids, fixed["original"], split=split)
            paired[split] = {"per_seed": {}}
        for seed in SEEDS:
            key = str(seed)
            matrix = synthetic_matrix(records, np.asarray(partitions["children"][key]),
                                      all_ids, control_order, split)
            evaluator.write_candidate_matrix(run / f"matrix-{split}-seed-{seed}.json", matrix)
            inherited[split][key] = {"exact_masks_and_all_raw_measurements": True,
                                     "original_coverage": 0.5, "full_oracle_coverage": 1.0}
            if split == "calibration":
                portfolios[key] = select_calibration_portfolio(matrix)
            else:
                ids = portfolios[key]["candidate_ids"]
                if split == "development":
                    fixed["selected"][key] = select_development_fixed(matrix, ids)
                    fixed["full_oracle"][key] = select_development_fixed(matrix, all_ids)
                metrics["selected"][key][split] = summarize_portfolio(
                    matrix, ids, fixed["selected"][key], split=split)
                metrics["full_oracle"][key][split] = summarize_portfolio(
                    matrix, all_ids, fixed["full_oracle"][key], split=split)
                paired[split]["per_seed"][key] = paired_portfolio_difference(matrix, ids, original, old_ids)
                coverage.append(portfolio_coverage(matrix, ids))
        if split != "calibration":
            selected = np.stack(coverage)
            paired[split]["aggregate"] = paired_coverage_difference(selected,
                np.broadcast_to(portfolio_coverage(original, old_ids), selected.shape), original["context_ids"])
    for name, value in (("metrics", metrics), ("paired", paired), ("portfolio_selection", portfolios),
                        ("fixed_selection", fixed), ("inherited_mask_validation", inherited)):
        write_json(run / f"{name}.json", value)
    cal_seal = {name: sha(run / name) for name in
                ("partitions.json", "preregistration.json", "portfolio_selection.json")}
    write_json(run / "calibration_seal.json", cal_seal)
    heldout_seal = {name: sha(run / name) for name in
                    (*cal_seal, "calibration_seal.json", "fixed_selection.json")}
    write_json(run / "heldout_seal.json", heldout_seal)
    parity = {str(seed): [{"candidate_id": candidate_id, "states": 16,
        "kind": "cheap" if candidate_id >= 0 else "control" if candidate_id == -1 else "permuted_control",
        "exact_logits": True, "max_abs_logit_error": 0}
        for candidate_id in partitions["parity_ids"][str(seed)] + [-1, -2]] for seed in SEEDS}
    write_json(run / "full_model_cached_validation.json", parity)
    weights = {"synthetic.tensor": "a" * 64}
    write_json(run / "environment.json", {"gpu": "synthetic CPU fixture, no GPU", "torch": "test",
        "cuda": "none", "transformers": "test", "original_weight_sha256": weights})
    write_json(run / "final_weight_sha256.json", weights)
    gate = cycle_gate(metrics["selected"], paired["heldout_ood"]["aggregate"])
    write_json(run / "promotion_decision.json", {
        **gate, "status": "completed", "capability_pass": gate["point_pass"], "promote": False,
        "original_weights_unchanged": True, "prior_negative_results_unchanged": True,
        "runtime_training": "not_run_out_of_scope", "runtime_timing": "not_run_out_of_scope",
        "global_bandwidth_solver_promotion": False, "single_gpu_wall_elapsed_seconds": 12.0,
        "gpu_budget_ceiling_seconds": 14400,
    })
    events = [
        {"event": "capture_progress", "split": "calibration"},
        {"event": "portfolio_choices_sealed_before_development", "seal": cal_seal},
        {"event": "capture_progress", "split": "development"},
        {"event": "fixed_comparators_sealed_before_heldouts", "seal": heldout_seal},
        {"event": "capture_progress", "split": "heldout_matched"},
        {"event": "capture_progress", "split": "heldout_ood"}, {"event": "cycle_complete"}]
    (run / "events.jsonl").write_bytes(("\n".join(map(json.dumps, events)) + "\n").encode())
    yield run, repo, verifier, report
    for path in paths:
        sys.path.remove(path)


def test_full_cpu_verifier_recomputes_all_artifacts(synthetic_run):
    run, repo, verifier, _report = synthetic_run
    result = verifier.verify_run(run, repo=repo)
    assert result["pass"] and result["capability_pass"]
    assert result["matrix_manifests_checked"] == 16
    assert result["candidate_cells_checked"] == 8 * 4 * (28 + 3 * 1820)
    assert result["registered_sources_checked"] == 29
    assert result["registered_wall_budget_respected"]
    assert result["inherited_all_scalar_fields_masks_and_controls_equal"]
    assert result["no_router_or_economic_promotion"]
    assert not result["language_model_loaded"] and not result["gpu_used"]


def test_report_primary_equal_budget_and_unmeasured_economics(synthetic_run):
    run, repo, _verifier, report = synthetic_run
    text = report.build_report(run, repo=repo)
    assert "capability gate passed" in text
    assert text.count("| Selected 28 |") == 10  # Scope row plus all nine evaluation cells.
    assert "Paired selected-28 minus original-28" in text
    assert "population failure" in text and "threshold remains unresolved" in text
    assert "about 65 times as many choices" in text
    assert "Parent-permuted KL" in text and "0.000199" in text
    assert "**unmeasured**, not zero" in text and "not a latency benchmark" in text
    assert "75.78" not in text


def test_prereg_only_does_not_open_output_matrices(synthetic_run):
    run, repo, verifier, _report = synthetic_run
    path = run / "matrix-calibration-original.json"
    with altered(path, lambda _value: {"deliberately": "unreadable archive"}):
        result = verifier.verify_run(run, repo=repo, prereg_only=True)
    assert result["pass"] and not result["heldout_artifacts_opened"]


def test_full_verification_requires_terminal_completion(synthetic_run):
    run, repo, verifier, _report = synthetic_run
    with (altered(run / "promotion_decision.json", lambda value: {**value, "status": "running"}),
          pytest.raises(ValueError, match="completed granularity")):
        verifier.verify_run(run, repo=repo)


@pytest.mark.parametrize("elapsed", [-1, float("nan"), float("inf"), 14401, True])
def test_completed_run_must_respect_registered_budget(synthetic_run, elapsed):
    run, repo, verifier, _report = synthetic_run
    with (altered(run / "promotion_decision.json", lambda value: {
            **value, "single_gpu_wall_elapsed_seconds": elapsed}),
          pytest.raises(ValueError, match="registered wall budget")):
        verifier.verify_run(run, repo=repo)


def test_missing_registered_dependency_is_rejected(synthetic_run):
    run, repo, verifier, _report = synthetic_run
    def remove_dependency(value):
        del value["source_sha256"]["experiments/expert_granularity_evaluation.py"]
        return value
    with (altered(run / "preregistration.json", remove_dependency),
          pytest.raises(ValueError, match="required scientific inputs")):
        verifier.verify_run(run, repo=repo, prereg_only=True)


def test_complete_recipe_is_checked_after_matching_hashes(synthetic_run):
    run, repo, verifier, _report = synthetic_run
    name = "configs/expert_granularity.json"
    original = (repo / name).read_bytes()
    paths = [repo / name, run / "registered_sources" / name]
    try:
        changed = read_json(repo / name)
        changed["bootstrap_repetitions"] = 40
        for path in paths:
            write_json(path, changed)
        def modified_registration(value):
            value["config"] = changed
            value["source_sha256"][name] = sha(repo / name)
            return value
        with (altered(run / "preregistration.json", modified_registration),
              pytest.raises(ValueError, match="prospectively bounded protocol")):
            verifier.verify_run(run, repo=repo, prereg_only=True)
    finally:
        for path in paths:
            path.write_bytes(original)


def test_archive_hash_tampering_is_rejected(synthetic_run):
    run, repo, verifier, _report = synthetic_run
    descriptor = read_json(run / "matrix-calibration-original.json")
    path = run / descriptor["chunks"][0]["file"]
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"tampered")
        with pytest.raises(AssertionError, match="archive hash mismatch"):
            verifier.verify_run(run, repo=repo)
    finally:
        path.write_bytes(original)


def test_selection_tampering_cannot_bypass_seal(synthetic_run):
    run, repo, verifier, _report = synthetic_run
    with (altered(run / "portfolio_selection.json", lambda value: {**value, "extra_seed": {}}),
          pytest.raises(ValueError, match="calibration seal hash mismatch")):
        verifier.verify_run(run, repo=repo)


def test_reported_metrics_are_recomputed_not_trusted(synthetic_run):
    run, repo, verifier, _report = synthetic_run
    def changed_metric(value):
        value["selected"]["17"]["heldout_ood"]["cheap_coverage"] = 0.25
        return value
    with (altered(run / "metrics.json", changed_metric), pytest.raises(ValueError, match="metrics")):
        verifier.verify_run(run, repo=repo)


def test_capability_failure_does_not_promote_full_oracle(synthetic_run):
    run, repo, _verifier, report = synthetic_run
    with altered(run / "promotion_decision.json", lambda value: {**value, "capability_pass": False}):
        text = report.build_report(run, repo=repo)
    assert "capability gate did not pass" in text
    assert "Router recognition and inference economics were not tested" in text
    assert "not substitute the larger 1,820-choice diagnostic oracle" in text


def test_source_snapshot_must_match_registered_hash(synthetic_run):
    run, repo, verifier, _report = synthetic_run
    path = run / "registered_sources/configs/expert_granularity.json"
    with (altered(path, lambda value: {**value, "batch_size": 8}),
          pytest.raises(ValueError, match="registered snapshot changed")):
        verifier.verify_run(run, repo=repo, prereg_only=True)


def test_incomplete_run_is_not_a_negative_capability_result(synthetic_run, tmp_path):
    run, repo, _verifier, report = synthetic_run
    write_json(tmp_path / "preregistration.json", read_json(run / "preregistration.json"))
    write_json(tmp_path / "incomplete.json", {"error_type": "TimeoutError", "error": "fixture timeout",
        "single_gpu_wall_elapsed_seconds": 14401})
    text = report.build_report(tmp_path, repo=repo)
    assert "**Incomplete.**" in text
    assert "No positive or negative capability conclusion" in text
    assert "Primary equal-budget result" not in text
    assert "capability gate did not pass" not in text


def test_report_requires_terminal_artifact(synthetic_run, tmp_path):
    run, repo, _verifier, report = synthetic_run
    write_json(tmp_path / "preregistration.json", copy.deepcopy(read_json(run / "preregistration.json")))
    with pytest.raises(ValueError, match="terminal completed or explicitly incomplete"):
        report.build_report(tmp_path, repo=repo)
