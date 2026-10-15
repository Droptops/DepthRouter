"""Synthetic report/integrity tests; no language model or GPU is involved."""

import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from depth_router.expert_audit_metrics import candidate_combinations, select_fixed_candidate

ROOT = Path(__file__).resolve().parents[1]


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, sort_keys=True) + "\n").encode())


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "experiments"))
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return (importlib.import_module("expert_recognition_report"),
            importlib.import_module("verify_expert_recognition"),
            importlib.import_module("expert_recognition"))


@pytest.fixture
def stored_run(tmp_path, modules):
    repo, run = tmp_path / "repo", tmp_path / "repo/artifacts/test"
    run.mkdir(parents=True)
    config = json.loads((ROOT / "configs/expert_recognition.json").read_text())
    config["bootstrap_repetitions"] = 40
    labels = np.repeat(np.arange(8), 608)
    fixed = select_fixed_candidate(np.full((4, 28), 0.1), np.ones((4, 28), dtype=bool))
    config["fixed_candidate_index"] = fixed["index"]
    manifest = [{"id": f"context-{index}", "split": "confirmation", "task": "creative_writing"
                 if index % 2 else "general_qa", "stratum": "general_natural_language",
                 "confirmation_stratum": "ood" if index % 2 else "matched"} for index in range(4)]
    for name, value in ((config["frozen_partition"], labels.tolist()),
                        (config["old_fixed_selection"], {config["partition_name"]: fixed})):
        _write(repo / name, value)
    config["frozen_partition_sha256"] = _sha(repo / config["frozen_partition"])
    config["old_fixed_selection_sha256"] = _sha(repo / config["old_fixed_selection"])
    _write(repo / "configs/expert_recognition.json", config)
    _write(repo / "EXPERT_RECOGNITION_PREREGISTRATION.md", "Protocol placeholder")
    manifest_path = repo / config["manifest"]
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_bytes(("\n".join(json.dumps(row) for row in manifest) + "\n").encode())
    sources = ["configs/expert_recognition.json", "EXPERT_RECOGNITION_PREREGISTRATION.md",
               config["manifest"], config["frozen_partition"], config["old_fixed_selection"]]
    for name in sources:
        destination = run / "registered_sources" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((repo / name).read_bytes())
    _write(run / "preregistration.json", {
        "utc": "2026-10-15T00:00:00Z", "config": config,
        "before_model_load_fit_and_fresh_outputs": True,
        "source_sha256": {name: _sha(repo / name) for name in sources},
    })
    _write(run / "frozen_partition.json", labels.tolist())
    _write(run / "frozen_fixed_selection.json", fixed)
    records = [{**row, "state_index": index, "position": 1, "group": 0, "slot": index,
                "dense_token": 0, "dense_margin": 1.0, "prefix_token_ids": [1, 2, 3]}
               for index, row in enumerate(manifest)]
    (run / "states-confirmation.jsonl").write_bytes(
        ("\n".join(json.dumps(row) for row in records) + "\n").encode())
    matrix = {"kl": np.full((4, 28), 0.1), "candidate_token": np.zeros((4, 28), dtype=np.int32),
              "dense_token": np.zeros(4, dtype=np.int32), "dense_margin": np.ones(4),
              "hidden_mse": np.ones((4, 28)), "candidate_margin": np.ones((4, 28)),
              "max_abs_logit_error": np.ones((4, 28)), "control_kl": np.zeros(4),
              "permuted_control_kl": np.zeros(4), "control_token": np.zeros(4, dtype=np.int32),
              "permuted_control_token": np.zeros(4, dtype=np.int32),
              "context_ids": np.array([row["id"] for row in records]), "positions": np.ones(4, dtype=np.int64),
              "selected_experts": np.array(candidate_combinations()), "partition": labels}
    np.savez_compressed(run / "matrix-confirmation.npz", **matrix)
    _write(run / "capture-confirmation.json", {"contexts_requested": 4, "contexts_evaluated": 4,
        "states": 4, "zero_state_context_ids": [], "prompt_truncations": 0})
    confirmation = modules[2].confirmation_decision(matrix, records, manifest, fixed, config)
    _write(run / "confirmation_decision.json", confirmation)
    _write(run / "full_model_cached_validation.json", [
        {"experts": list(combo), "states": config["batch_size"], "exact_logits": True,
         "max_abs_logit_error": 0.0} for combo in [*candidate_combinations(), tuple(range(8))]])
    _write(run / "promotion_decision.json", {
        "confirmation_pass": False, "recognition": "not_run_confirmation_failed",
        "runtime": "not_run_confirmation_failed", "promote": False,
        "original_weights_unchanged": True, "prior_construction_negative_result_unchanged": True,
        "single_gpu_wall_elapsed_seconds": 10.0, "gpu_budget_ceiling_seconds": 14400,
    })
    return repo, run


def test_confirmation_failure_report_preserves_stop_and_unknown_costs(stored_run, modules):
    repo, run = stored_run
    result = modules[0].build_report(run, repo=repo)
    assert "Independent capability confirmation failed" in result
    assert "stopped before selector fitting" in result
    assert "**unknown**, not zero" in result
    assert "all-three-construction-seed failure" in result
    assert "at most about 1.04%" in result


def test_preregistration_hashes_and_complete_confirmation_stop_verify(stored_run, modules):
    repo, run = stored_run
    verifier = modules[1]
    prereg = verifier.verify_run(run, repo=repo, prereg_only=True)
    assert prereg["pass"] and not prereg["heldout_artifacts_opened"]
    full = verifier.verify_run(run, repo=repo)
    assert full["confirmation_failure_stop_verified"]
    assert full["full_model_parity_checks"] == 29
    assert not full["language_model_loaded"] and not full["gpu_used"]


def test_registered_snapshot_tampering_is_rejected(stored_run, modules):
    repo, run = stored_run
    path = run / "registered_sources/EXPERT_RECOGNITION_PREREGISTRATION.md"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="registered snapshot changed"):
        modules[1].verify_run(run, repo=repo, prereg_only=True)


def test_live_newline_only_difference_requires_explicit_permission(stored_run, modules):
    repo, run = stored_run
    path = repo / "EXPERT_RECOGNITION_PREREGISTRATION.md"
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    with pytest.raises(ValueError, match="current source changed"):
        modules[1].verify_run(run, repo=repo, prereg_only=True)
    result = modules[1].verify_run(run, repo=repo, prereg_only=True, allow_current_newline_only=True)
    assert result["current_source_newline_only_differences"] == ["EXPERT_RECOGNITION_PREREGISTRATION.md"]


def test_full_verification_does_not_read_unfinished_outputs(stored_run, modules):
    repo, run = stored_run
    (run / "promotion_decision.json").unlink()
    with pytest.raises(ValueError, match="requires terminal promotion_decision"):
        modules[1].verify_run(run, repo=repo)
    with pytest.raises(ValueError, match="terminal decision or explicit incomplete"):
        modules[0].build_report(run, repo=repo)


def test_stop_violation_is_not_recast_as_success(stored_run, modules):
    repo, run = stored_run
    np.savez_compressed(run / "matrix-heldout_ood.npz", unexpected=np.array(1))
    with pytest.raises(ValueError, match="work continued after failed confirmation"):
        modules[1].verify_run(run, repo=repo)


def test_incomplete_report_is_not_negative_quality_evidence(stored_run, modules):
    repo, run = stored_run
    (run / "promotion_decision.json").unlink()
    _write(run / "incomplete.json", {"error_type": "TimeoutError", "error": "bounded budget reached",
                                    "single_gpu_wall_elapsed_seconds": 14401})
    text = modules[0].build_report(run, repo=repo)
    assert "Incomplete: no quality or economic conclusion is established" in text
    assert "TimeoutError" in text


def test_metric_comparison_rejects_material_mismatch(modules):
    modules[1]._same({"value": [0.1]}, {"value": [0.100000000001]})
    with pytest.raises(ValueError, match="numeric result differs"):
        modules[1]._same({"value": [0.1]}, {"value": [0.101]})
