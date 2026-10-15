"""Tamper and completeness tests for the offline evidence verifier; no model calls."""

import importlib
import json
from pathlib import Path

import numpy as np
import pytest


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, sort_keys=True) + "\n").encode())


@pytest.fixture
def verifier(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("verify_expert_audit")


@pytest.fixture
def stored_run(tmp_path, verifier):
    repo = tmp_path / "repo"
    run = repo / "artifacts/expert_construction/test-integrity"
    run.mkdir(parents=True)
    config = {"seeds": [17, 29, 43], "assignment": {"steps": 100, "batch_size": 16},
              "batch_size": 16, "manifest": "data/expert_audit/evaluation_manifest.jsonl"}
    write_json(repo / "configs/expert_construction.json", config)
    write_json(repo / config["manifest"], {"prompt": "Fixture data only."})
    code = repo / "src/depth_router/expert_fixture.py"
    code.parent.mkdir(parents=True)
    code.write_bytes(b"# Fixture source\nvalue = 17\n")
    sources = {}
    for name in ("configs/expert_construction.json", config["manifest"],
                 "src/depth_router/expert_fixture.py"):
        source = repo / name
        snapshot = run / "registered_sources" / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(source.read_bytes())
        sources[name] = verifier._sha(source)
    write_json(run / "preregistration.json", {
        "config": config, "frozen_assignment_recipe": config["assignment"],
        "before_model_load_fit_and_heldout": True, "source_sha256": sources,
    })
    labels = np.repeat(np.arange(8), 608)
    keys = [f"{arm}-{seed}" for seed in verifier.SEEDS for arm in "ABCD"]
    for key in keys:
        write_json(run / f"partition-{key}.json", labels.tolist())
    fixed = {key: {"index": 0, "experts": list(verifier.COMBINATIONS[0]),
                   "selected_on": "development"} for key in keys}
    write_json(run / "fixed_selection.json", fixed)
    write_json(run / "family_selection.json", {"arm": "A", "selected_on": "development"})
    seal_names = [f"partition-{key}.json" for key in keys] + [
        "preregistration.json", "fixed_selection.json", "family_selection.json"]
    write_json(run / "heldout_seal.json", {name: verifier._sha(run / name) for name in seal_names})
    for seed in verifier.SEEDS:
        objective = {"loss_by_step": [0.1] * 100, "gradient_norm_by_step": [0.01] * 100,
                     "final_partition_sha256": verifier._partition_hash(labels),
                     **{name: "a" * 64 for name in verifier.SHARED_HASHES}}
        write_json(run / f"construction-{seed}.json", {
            "seed": seed, "recipe": config["assignment"], "parameter_count": 4864 * 8,
            "effective_batch_size": 16, "initial_partition_sha256": verifier._partition_hash(labels),
            "initialization_equal": True, "minibatch_order_equal": True, "candidate_order_equal": True,
            "objectives": {"hidden": objective, "output": objective},
        })
    for split in verifier.SPLITS:
        state = {"id": f"context-{split}", "position": 1, "dense_token": 10,
                 "split": split, "state_index": 0}
        write_json(run / f"states-{split}.jsonl", state)
        matrix = {"partition": labels, "selected_experts": verifier.COMBINATIONS,
                  "context_ids": np.array([state["id"]]), "positions": np.array([1]),
                  "dense_token": np.array([10]), "dense_margin": np.array([0.5]),
                  "candidate_token": np.full((1, 28), 10, dtype=np.int32),
                  "control_token": np.array([10]), "permuted_control_token": np.array([10])}
        matrix.update({field: np.zeros((1, 28)) for field in
                       ("kl", "hidden_mse", "candidate_margin", "max_abs_logit_error")})
        matrix.update({field: np.zeros(1) for field in ("control_kl", "permuted_control_kl")})
        for key in keys:
            np.savez_compressed(run / f"matrix-{split}-{key}.npz", **matrix)
    checks = [{"experts": list(combo), "exact_logits": True, "max_abs_logit_error": 0,
               "states": 16} for combo in (*verifier.COMBINATIONS, tuple(range(8)))]
    write_json(run / "full_model_cached_validation.json", {key: checks for key in keys})
    write_json(run / "promotion_decision.json", {
        "phase1_complete": True, "original_weights_unchanged": True,
    })
    return repo, run


def test_complete_run_verifies_all_raw_evidence(verifier, stored_run):
    repo, run = stored_run
    audit = verifier.verify_run(run, repo=repo)
    assert audit["pass"] and audit["matrix_files_checked"] == 48
    assert audit["partitions_checked"] == 12
    assert audit["heldout_seal_entries_checked"] == 15
    assert audit["full_model_parity_checks"] == 348
    assert audit["current_source_hashes_match_exactly"]
    assert not audit["quality_or_economic_promotion_inferred"]
    assert not audit["model_loaded"]


def test_partial_verification_does_not_open_heldout_or_decision(verifier, stored_run):
    repo, run = stored_run
    (run / "promotion_decision.json").write_bytes(b"not valid JSON")
    (run / "states-heldout_matched.jsonl").write_bytes(b"not valid JSON")
    audit = verifier.verify_run(run, repo=repo, prereg_only=True)
    assert audit["scope"] == "preregistration_only"
    assert not audit["heldout_artifacts_opened"]


def test_full_verification_rejects_incomplete_run_before_reading_heldout(verifier, stored_run):
    repo, run = stored_run
    write_json(run / "promotion_decision.json", {"phase1_complete": False})
    (run / "states-heldout_matched.jsonl").write_bytes(b"not valid JSON")
    with pytest.raises(ValueError, match="phase1_complete"):
        verifier.verify_run(run, repo=repo)


def test_snapshot_hashes_are_strict_even_with_newline_option(verifier, stored_run):
    repo, run = stored_run
    path = run / "registered_sources/src/depth_router/expert_fixture.py"
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    with pytest.raises(ValueError, match="registered source copy hash mismatch"):
        verifier.verify_run(run, repo=repo, prereg_only=True, allow_current_newline_only=True)


def test_current_newline_normalization_requires_explicit_option(verifier, stored_run):
    repo, run = stored_run
    path = repo / "src/depth_router/expert_fixture.py"
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    with pytest.raises(ValueError, match="newline_only=True"):
        verifier.verify_run(run, repo=repo, prereg_only=True)
    audit = verifier.verify_run(run, repo=repo, prereg_only=True, allow_current_newline_only=True)
    assert not audit["current_source_hashes_match_exactly"]
    assert audit["current_source_newline_only_differences"] == ["src/depth_router/expert_fixture.py"]


def test_missing_seal_entry_is_rejected(verifier, stored_run):
    repo, run = stored_run
    seal = verifier._json(run / "heldout_seal.json")
    del seal["partition-D-43.json"]
    write_json(run / "heldout_seal.json", seal)
    with pytest.raises(ValueError, match="missing required frozen"):
        verifier.verify_run(run, repo=repo)


def test_unbalanced_partition_is_rejected(verifier, stored_run):
    repo, run = stored_run
    labels = verifier._json(run / "partition-C-29.json")
    labels[0] = 1
    write_json(run / "partition-C-29.json", labels)
    with pytest.raises(ValueError, match="exactly 608"):
        verifier.verify_run(run, repo=repo)


@pytest.mark.parametrize("tamper", ["shared_hash", "short_fit", "nonfinite_gradient"])
def test_construction_tampering_is_rejected(verifier, stored_run, tamper):
    repo, run = stored_run
    path = run / "construction-17.json"
    record = verifier._json(path)
    output = record["objectives"]["output"]
    if tamper == "shared_hash":
        output["candidate_indices_sha256"] = "b" * 64
    elif tamper == "short_fit":
        output["loss_by_step"].pop()
    else:
        output["gradient_norm_by_step"][3] = float("nan")
    write_json(path, record)
    with pytest.raises(ValueError, match="shared construction hash|100 finite"):
        verifier.verify_run(run, repo=repo)


def test_missing_candidate_matrix_is_rejected(verifier, stored_run):
    repo, run = stored_run
    (run / "matrix-heldout_ood-D-43.npz").unlink()
    with pytest.raises(ValueError, match="exactly 48"):
        verifier.verify_run(run, repo=repo)


@pytest.mark.parametrize("tamper", ["candidate_order", "state_alignment"])
def test_matrix_schema_or_alignment_tampering_is_rejected(verifier, stored_run, tamper):
    repo, run = stored_run
    path = run / "matrix-heldout_matched-A-17.npz"
    with np.load(path, allow_pickle=False) as stored:
        matrix = {name: stored[name] for name in stored.files}
    if tamper == "candidate_order":
        matrix["selected_experts"] = matrix["selected_experts"][::-1]
    else:
        matrix["positions"][0] = 7
    np.savez_compressed(path, **matrix)
    with pytest.raises(ValueError, match="candidate enumeration|matrix/state alignment"):
        verifier.verify_run(run, repo=repo)


def test_unchanged_original_weights_attestation_is_required(verifier, stored_run):
    repo, run = stored_run
    write_json(run / "promotion_decision.json", {
        "phase1_complete": True, "original_weights_unchanged": False,
    })
    with pytest.raises(ValueError, match="unchanged original weights"):
        verifier.verify_run(run, repo=repo)


def test_fullmodel_parity_failure_is_rejected(verifier, stored_run):
    repo, run = stored_run
    path = run / "full_model_cached_validation.json"
    parity = verifier._json(path)
    parity["D-43"][0]["exact_logits"] = False
    write_json(path, parity)
    with pytest.raises(ValueError, match="full-model parity failed"):
        verifier.verify_run(run, repo=repo)


def test_sealed_path_cannot_escape_run_directory(verifier, stored_run):
    repo, run = stored_run
    path = run / "heldout_seal.json"
    seal = verifier._json(path)
    seal["../unrelated-file"] = "a" * 64
    write_json(path, seal)
    with pytest.raises(ValueError, match="unsafe artifact path"):
        verifier.verify_run(run, repo=repo)
