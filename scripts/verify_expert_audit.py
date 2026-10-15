"""Verify frozen expert-audit evidence offline, without loading a language model.

Use --prereg-only while a run is in progress. Full verification requires its
phase1_complete marker before opening any held-out state or matrix artifact.
Reports and this post-run verifier are deliberately outside the scientific seal.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path, PurePosixPath

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("calibration", "development", "heldout_matched", "heldout_ood")
SEEDS = (17, 29, 43)
COMBINATIONS = tuple(itertools.combinations(range(8), 6))
SHARED_HASHES = ("initial_logits_sha256", "minibatch_indices_sha256", "candidate_indices_sha256")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _child(root, name):
    path = PurePosixPath(name)
    _require(not path.is_absolute() and bool(path.parts)
             and all(part not in (".", "..") and ":" not in part and "\\" not in part
                     for part in path.parts), f"unsafe artifact path: {name}")
    resolved = (root / name).resolve()
    _require(resolved.is_relative_to(root.resolve()), f"artifact escapes root: {name}")
    return resolved


def _valid_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _partition_hash(labels):
    # Reproduce output_gap.tensor_sha256(torch.int64) without importing PyTorch.
    digest = hashlib.sha256()
    digest.update(json.dumps({"shape": [len(labels)], "dtype": "torch.int64"}).encode())
    digest.update(np.asarray(labels, dtype="<i8").tobytes())
    return digest.hexdigest()


def _partition(value, label):
    values = np.asarray(value)
    _require(values.shape == (4864,) and np.issubdtype(values.dtype, np.integer)
             and not np.issubdtype(values.dtype, np.bool_), f"invalid partition array: {label}")
    _require(np.all((values >= 0) & (values < 8)), f"invalid expert labels: {label}")
    _require(np.array_equal(np.bincount(values, minlength=8), np.full(8, 608)),
             f"partition must have exactly 608 neurons per expert: {label}")
    return values


def _verify_preregistration(run, repo, allow_current_newline_only):
    prereg = _json(run / "preregistration.json")
    sources = prereg.get("source_sha256", {})
    _require(bool(sources), "preregistration must contain source hashes")
    _require(prereg.get("before_model_load_fit_and_heldout") is True,
             "preregistration lacks its before-execution assertion")
    newline_only = []
    for name, expected in sources.items():
        _require(_valid_hash(expected), f"invalid registered hash: {name}")
        _require(_sha(_child(run / "registered_sources", name)) == expected,
                 f"registered source copy hash mismatch: {name}")
        current = _child(repo, name)
        if _sha(current) != expected:
            snapshot = _child(run / "registered_sources", name)
            differs_only_in_newlines = current.read_bytes().replace(b"\r\n", b"\n") == (
                snapshot.read_bytes().replace(b"\r\n", b"\n")
            )
            _require(allow_current_newline_only and differs_only_in_newlines,
                     f"current scientific source hash mismatch: {name}; "
                     f"newline_only={differs_only_in_newlines}")
            newline_only.append(name)
    config = prereg["config"]
    config_name = "configs/expert_construction.json"
    _require(config_name in sources and config["manifest"] in sources,
             "config and manifest must both be registered")
    _require(_json(repo / config_name) == config, "registered config payload mismatch")
    _require(config["assignment"] == prereg["frozen_assignment_recipe"],
             "assignment recipe differs from preregistration")
    _require(tuple(config["seeds"]) == SEEDS, "this bounded cycle requires seeds 17, 29, and 43")
    _require(config["assignment"]["steps"] == 100, "this bounded cycle requires 100 construction steps")
    return prereg, {
        "pass": True, "scope": "preregistration_only", "run_dir": str(run),
        "registered_utc": prereg.get("utc"), "registered_files_checked": len(sources),
        "snapshot_source_hashes_match": True,
        "current_source_hashes_match_exactly": not newline_only,
        "current_source_newline_only_differences": newline_only,
        "current_newline_only_differences_explicitly_allowed": allow_current_newline_only,
        "config_and_assignment_recipe_match": True,
        "manifest_sha256": sources[config["manifest"]],
        "model_loaded": False, "heldout_artifacts_opened": False,
        "report_or_verifier_files_part_of_scientific_seal": False,
    }


def _verify_construction(run, config, partitions):
    result = {}
    for seed in SEEDS:
        audit = _json(run / f"construction-{seed}.json")
        _require(audit["seed"] == seed and audit["recipe"] == config["assignment"],
                 f"construction recipe/seed mismatch: {seed}")
        _require(audit["parameter_count"] == 4864 * 8, f"assignment parameter count mismatch: {seed}")
        _require(audit["effective_batch_size"] == config["assignment"]["batch_size"],
                 f"construction batch size mismatch: {seed}")
        _require(audit["initial_partition_sha256"] == _partition_hash(partitions[f"A-{seed}"]),
                 f"initial partition hash mismatch: {seed}")
        _require(all(audit[field] is True for field in
                     ("initialization_equal", "minibatch_order_equal", "candidate_order_equal")),
                 f"paired construction assertion missing: {seed}")
        objectives = audit["objectives"]
        _require(set(objectives) == {"hidden", "output"}, f"missing paired objective: {seed}")
        for name in SHARED_HASHES:
            _require(_valid_hash(objectives["hidden"][name])
                     and objectives["hidden"][name] == objectives["output"][name],
                     f"C/D shared construction hash mismatch: {seed} {name}")
        for objective, arm in (("hidden", "C"), ("output", "D")):
            record = objectives[objective]
            for field in ("loss_by_step", "gradient_norm_by_step"):
                values = np.asarray(record[field], dtype=float)
                _require(values.shape == (100,) and np.isfinite(values).all(),
                         f"construction needs 100 finite {field}: {seed} {objective}")
                if field == "gradient_norm_by_step":
                    _require(np.all(values >= 0), f"negative gradient norm: {seed} {objective}")
            _require(record["final_partition_sha256"] == _partition_hash(partitions[f"{arm}-{seed}"]),
                     f"final construction partition hash mismatch: {seed} {objective}")
        result[str(seed)] = {"steps_per_objective": 100, "paired_hashes_match": True,
                             "losses_and_gradient_norms_finite": True}
    return result


def _verify_matrix(path, labels, states):
    count = len(states)
    _require(count > 0, f"empty state file: {path.name}")
    with np.load(path, allow_pickle=False) as matrix:
        _require(np.array_equal(matrix["partition"], labels), f"matrix partition mismatch: {path.name}")
        _require(np.array_equal(matrix["selected_experts"], np.asarray(COMBINATIONS)),
                 f"candidate enumeration mismatch: {path.name}")
        for field, expected in (("context_ids", [row["id"] for row in states]),
                                ("positions", [row["position"] for row in states]),
                                ("dense_token", [row["dense_token"] for row in states])):
            _require(np.array_equal(matrix[field], np.asarray(expected)),
                     f"matrix/state alignment mismatch: {path.name} {field}")
        for field in ("kl", "hidden_mse", "candidate_margin", "max_abs_logit_error",
                      "control_kl", "permuted_control_kl", "dense_margin"):
            values = matrix[field]
            expected_shape = (count, 28) if field in {
                "kl", "hidden_mse", "candidate_margin", "max_abs_logit_error"} else (count,)
            _require(values.shape == expected_shape and np.isfinite(values).all(),
                     f"invalid matrix shape/nonfinite values: {path.name} {field}")
            _require(np.all(values >= (-1e-5 if "kl" in field else 0)),
                     f"negative metric outside tolerance: {path.name} {field}")
        for field in ("candidate_token", "dense_token", "control_token", "permuted_control_token"):
            values = matrix[field]
            expected_shape = (count, 28) if field == "candidate_token" else (count,)
            _require(values.shape == expected_shape and np.issubdtype(values.dtype, np.integer)
                     and np.all((values >= 0) & (values < 151936)),
                     f"invalid token array: {path.name} {field}")
    return {"sha256": _sha(path), "states": count, "candidates": 28}


def verify_run(run_dir, *, repo=ROOT, prereg_only=False, allow_current_newline_only=False):
    """Raise on inconsistent evidence; never execute fitting or model inference."""
    run, repo = Path(run_dir).resolve(), Path(repo).resolve()
    prereg, result = _verify_preregistration(run, repo, allow_current_newline_only)
    if prereg_only:
        return result
    decision = _json(run / "promotion_decision.json")
    _require(decision.get("phase1_complete") is True,
             "full verification requires phase1_complete; use --prereg-only while running")
    _require(decision.get("original_weights_unchanged") is True,
             "promotion decision does not attest unchanged original weights")
    config = prereg["config"]
    keys = [f"{arm}-{seed}" for seed in SEEDS for arm in "ABCD"]
    partitions = {key: _partition(_json(run / f"partition-{key}.json"), key) for key in keys}
    seal = _json(run / "heldout_seal.json")
    required_seal = {f"partition-{key}.json" for key in keys}
    required_seal |= {"preregistration.json", "fixed_selection.json", "family_selection.json"}
    _require(required_seal <= seal.keys(), "held-out seal is missing required frozen artifacts")
    for name, expected in seal.items():
        _require(_valid_hash(expected) and _sha(_child(run, name)) == expected,
                 f"held-out seal hash mismatch: {name}")
    fixed = _json(run / "fixed_selection.json")
    _require(set(fixed) == set(keys), "fixed choices do not cover every partition")
    for key, choice in fixed.items():
        index = choice["index"]
        _require(type(index) is int and 0 <= index < 28 and choice["selected_on"] == "development"
                 and choice["experts"] == list(COMBINATIONS[index]),
                 f"invalid development-frozen fixed choice: {key}")
    construction = _verify_construction(run, config, partitions)
    expected_matrices = {f"matrix-{split}-{key}.npz" for split in SPLITS for key in keys}
    actual_matrices = {path.name for path in run.glob("matrix-*.npz")}
    _require(actual_matrices == expected_matrices,
             f"complete run needs exactly 48 canonical matrices; missing="
             f"{sorted(expected_matrices - actual_matrices)}, extra="
             f"{sorted(actual_matrices - expected_matrices)}")
    matrices = {}
    for split in SPLITS:
        states = [json.loads(line) for line in (run / f"states-{split}.jsonl").read_text().splitlines()]
        _require(all(row["split"] == split and row["state_index"] == index
                     for index, row in enumerate(states)), f"state split/index mismatch: {split}")
        for key in keys:
            name = f"matrix-{split}-{key}.npz"
            matrices[name] = _verify_matrix(run / name, partitions[key], states)
    parity = _json(run / "full_model_cached_validation.json")
    _require(set(parity) == set(keys), "full-model parity does not cover every partition")
    for key, checks in parity.items():
        expected = [list(combo) for combo in COMBINATIONS] + [list(range(8))]
        _require([check["experts"] for check in checks] == expected,
                 f"full-model parity misses candidates: {key}")
        _require(all(check["exact_logits"] is True and check["max_abs_logit_error"] == 0
                     and check["states"] == config["batch_size"] for check in checks),
                 f"full-model parity failed: {key}")
    result.update({
        "scope": "complete_phase1_integrity", "heldout_artifacts_opened": True,
        "heldout_seal_entries_checked": len(seal), "partitions_checked": len(partitions),
        "neurons_per_expert": 608, "paired_construction": construction,
        "matrix_files_checked": len(matrices), "matrices": matrices,
        "full_model_parity_checks": sum(len(checks) for checks in parity.values()),
        "original_weights_unchanged_from_execution_record": True,
        "weight_claim_scope": "verifies saved execution assertion; no model weights are reloaded",
        "quality_or_economic_promotion_inferred": False,
    })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--prereg-only", action="store_true")
    parser.add_argument("--allow-current-newline-only", action="store_true",
                        help="Allow and explicitly report CRLF/LF-only changes in live source; "
                        "saved source snapshots must always match their exact byte hashes.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = verify_run(args.run_dir, repo=args.repo, prereg_only=args.prereg_only,
                            allow_current_newline_only=args.allow_current_newline_only)
    except (ValueError, KeyError, OSError, TypeError) as error:
        result = {"pass": False, "error_type": type(error).__name__, "error": str(error),
                  "model_loaded": False}
    serialized = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    if args.output:
        _require(not args.output.exists(), "refusing to overwrite an existing audit artifact")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(serialized)
    sys.stdout.write(serialized.decode())
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
