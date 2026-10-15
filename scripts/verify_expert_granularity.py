"""Verify frozen granularity artifacts on CPU, without model inference.

Full verification requires a completed terminal decision. During execution use
--prereg-only, which never opens fresh candidate matrices or held-out outputs.
This checks and recomputes saved evidence, not a new physical GPU measurement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from verify_expert_audit import _child, _json, _require, _sha, _valid_hash
from verify_expert_recognition import _same

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

SPLITS = ("calibration", "development", "heldout_matched", "heldout_ood")
SEEDS = (17, 29, 43)
REQUIRED_SOURCES = {
    "Dockerfile.photonic", "requirements-photonic.txt",
    "configs/expert_granularity.json", "EXPERT_GRANULARITY_PREREGISTRATION.md",
    "experiments/expert_granularity.py", "experiments/expert_granularity_evaluation.py",
    "experiments/expert_construction.py", "scripts/run_expert_granularity.ps1",
    "scripts/prepare_granularity_data.py", "src/depth_router/expert_granularity.py",
    "src/depth_router/expert_partitions.py", "src/depth_router/expert_audit_metrics.py",
    "src/depth_router/expert_audit_data.py", "src/depth_router/expert_recognition.py",
    "src/depth_router/granularity_data.py", "src/depth_router/recognition_data.py",
    "src/depth_router/output_gap.py", "src/depth_router/output_gap_data.py",
    "data/expert_granularity/README.md", "data/expert_granularity/data_audit.json",
    "data/expert_granularity/source_provenance.json",
    "data/expert_granularity/quarantined_confirmation.json",
    "data/expert_recognition/evaluation_manifest.jsonl", "data/expert_recognition/data_audit.json",
    "data/expert_audit/evaluation_manifest.jsonl", "data/expert_audit/historical_alpaca_prompts.jsonl",
    "data/expert_audit/historical_alpaca_provenance.json",
}


def _verify_seal(run, seal, required, label):
    _require(required <= seal.keys(), f"{label} misses required artifacts")
    for name, expected in seal.items():
        _require(_valid_hash(expected) and _sha(_child(run, name)) == expected,
                 f"{label} hash mismatch: {name}")


def _preregistration(run, repo, allow_current_newline_only):
    registration = _json(run / "preregistration.json")
    _require(registration["before_model_load_fit_and_fresh_outputs"] is True
             and registration["prior_negative_results_unchanged"] is True,
             "registration lacks prospective/preserved-prior-result assertions")
    sources = registration["source_sha256"]
    _require(bool(sources), "registration has no source hashes")
    newline_only = []
    for name, expected in sources.items():
        snapshot, current = _child(run / "registered_sources", name), _child(repo, name)
        _require(_valid_hash(expected) and _sha(snapshot) == expected, f"registered snapshot changed: {name}")
        if _sha(current) != expected:
            equivalent = (current.read_bytes().replace(b"\r\n", b"\n")
                          == snapshot.read_bytes().replace(b"\r\n", b"\n"))
            _require(allow_current_newline_only and equivalent,
                     f"current registered source changed: {name}; newline_only={equivalent}")
            newline_only.append(name)
    config = registration["config"]
    required_sources = REQUIRED_SOURCES | {config["manifest"], config["frozen_partition"]}
    _require(required_sources <= sources.keys(), "registration misses required scientific inputs")
    _same(_json(run / "registered_sources/configs/expert_granularity.json"), config, "config")
    from expert_granularity import validate_config

    validate_config(config)
    _require(sources[config["frozen_partition"]] == config["frozen_partition_sha256"],
             "frozen parent differs from its discovery hash")
    _verify_seal(run, registration["frozen_artifact_sha256"],
                 {"partitions.json", "verified_data_audit.json", "quarantined_confirmation.json"},
                 "preregistered artifact seal")
    return registration, {
        "pass": True, "scope": "preregistration_only", "registered_utc": registration["utc"],
        "registered_sources_checked": len(sources), "snapshot_source_hashes_match": True,
        "current_source_hashes_match_exactly": not newline_only,
        "current_source_newline_only_differences": newline_only,
        "current_newline_only_differences_explicitly_allowed": allow_current_newline_only,
        "language_model_loaded": False, "gpu_used": False, "heldout_artifacts_opened": False,
    }


def _partitions(run, config):
    from depth_router.expert_granularity import (
        array_sha256,
        candidate_schema_sha256,
        original_candidate_indices,
        split_parent_experts,
        validate_refinement,
    )

    stored = _json(run / "partitions.json")
    parent = np.asarray(_json(run / "registered_sources" / config["frozen_partition"]), dtype=np.int64)
    old_ids = original_candidate_indices()
    new_ids = np.setdiff1d(np.arange(1820), old_ids)
    expected = {"parent": parent.tolist(), "parent_array_sha256": array_sha256(parent),
                "candidate_schema_sha256": candidate_schema_sha256(),
                "original_candidate_ids": old_ids.tolist(), "children": {}, "audits": {}, "parity_ids": {}}
    for seed in SEEDS:
        child = split_parent_experts(parent, seed=seed)
        expected["children"][str(seed)] = child.tolist()
        expected["audits"][str(seed)] = validate_refinement(parent, child)
        chosen = np.random.default_rng(seed).choice(new_ids, 16, replace=False)
        expected["parity_ids"][str(seed)] = np.unique(
            np.concatenate((old_ids, [new_ids[0], new_ids[-1]], chosen))).tolist()
    _same(expected, stored, "prospective partitions/parity sample")
    return stored


def _records(run, split, manifest, config):
    records = [json.loads(line) for line in (run / f"states-{split}.jsonl").read_text().splitlines()]
    sources = {row["id"]: row for row in manifest if row["split"] == split}
    identities = []
    for index, row in enumerate(records):
        _require(row["id"] in sources and row["split"] == split and row["state_index"] == index,
                 f"invalid state identity/order: {split}/{index}")
        _require(row["task"] == sources[row["id"]]["task"]
                 and row["stratum"] == sources[row["id"]]["stratum"]
                 and row["position"] in config["continuation_positions"],
                 f"state/source field mismatch: {split}/{index}")
        _require(row["prefix_token_ids"] and all(type(token) is int and 0 <= token < 151936
                  for token in row["prefix_token_ids"]), f"invalid captured prefix: {split}/{index}")
        identities.append((row["id"], row["position"]))
    _require(records and len(set(identities)) == len(records), f"empty or duplicate states: {split}")
    capture = _json(run / f"capture-{split}.json")
    observed = {row["id"] for row in records}
    _require(capture["states"] == len(records) and capture["contexts_requested"] == len(sources)
             and capture["contexts_evaluated"] == len(observed)
             and set(capture["zero_state_context_ids"]) == sources.keys() - observed,
             f"captured-state counts/exclusions differ: {split}")
    cache = run / f"cache-{split}.pt"
    if cache.exists():
        _require(_sha(cache) == capture["cache_sha256"], f"activation cache hash changed: {split}")
    return records, {"states": len(records), "contexts_evaluated": len(observed),
                     "capture_metadata_sha256": _sha(run / f"capture-{split}.json"),
                     "state_metadata_sha256": _sha(run / f"states-{split}.jsonl"),
                     "activation_cache_present_and_hash_checked": cache.exists()}


def _matrix(path, records, labels, candidate_ids, control_order, split):
    from expert_granularity_evaluation import FLOAT_FIELDS, read_candidate_matrix

    from depth_router.expert_granularity import (
        array_sha256,
        candidate_combinations,
        matrix_arrays,
        partition_indices,
    )

    matrix = read_candidate_matrix(path)
    kl, _agreement, ids = matrix_arrays(matrix)
    count = len(records)
    _require(bool(matrix["complete"]) and len(kl) == count and np.array_equal(ids, candidate_ids),
             f"matrix incomplete or candidate/state count differs: {path.name}")
    _require(str(matrix["split"].item()) == split and np.array_equal(matrix["partition"], labels)
             and matrix["partition_sha256"].item() == array_sha256(labels),
             f"matrix split/partition differs: {path.name}")
    for field, record_name in (("context_ids", "id"), ("positions", "position"),
                                ("groups", "group"), ("slots", "slot"),
                                ("dense_token", "dense_token"), ("dense_margin", "dense_margin")):
        expected = np.asarray([row[record_name] for row in records], dtype=matrix[field].dtype)
        _require(np.array_equal(matrix[field], expected), f"matrix/state alignment differs: {path.name}/{field}")
    for name in (*FLOAT_FIELDS, "control_kl", "permuted_control_kl"):
        values = matrix[name]
        shape = (count, len(candidate_ids)) if name in FLOAT_FIELDS else (count,)
        _require(values.shape == shape and np.isfinite(values).all() and np.all(values >= 0),
                 f"invalid numeric matrix values: {path.name}/{name}")
    for name in ("candidate_token", "control_token", "permuted_control_token"):
        values = matrix[name]
        shape = (count, len(candidate_ids)) if name == "candidate_token" else (count,)
        _require(values.shape == shape and values.dtype.kind in "iu"
                 and np.all((values >= 0) & (values < 151936)), f"invalid token array: {path.name}/{name}")
    combinations = np.asarray(candidate_combinations())
    expected_indices = np.stack([partition_indices(labels, combinations[index]) for index in candidate_ids])
    _require(np.array_equal(matrix["neuron_indices"], expected_indices)
             and expected_indices.shape[1] == 3648
             and np.array_equal(matrix["control_order"], control_order),
             f"actual packed neuron layout differs: {path.name}")
    _require(np.all(matrix["control_kl"] == 0)
             and np.array_equal(matrix["control_token"], matrix["dense_token"]),
             f"original-order dense numerical control failed: {path.name}")
    descriptor = _json(path)
    return matrix, {"archive_manifest_sha256": _sha(path), "chunks_checked": len(descriptor["chunks"]),
                    "states": count, "candidates": len(candidate_ids), "complete": True,
                    "all_archive_payload_hashes_checked": True, "canonical_neuron_layout_recomputed": True}


def _chronology(run):
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    names = [row["event"] for row in events]
    _require(names[-1] == "cycle_complete", "event ledger lacks terminal completion")
    cal = names.index("portfolio_choices_sealed_before_development")
    dev = names.index("fixed_comparators_sealed_before_heldouts")
    _require(cal < dev, "selection seals appear in the wrong order")
    for index, event in enumerate(events):
        if event.get("split") == "development":
            _require(index > cal, "development work preceded calibration seal")
        if str(event.get("split", "")).startswith("heldout"):
            _require(index > dev, "heldout work preceded fixed-comparator seal")
    _same(events[cal]["seal"], _json(run / "calibration_seal.json"), "calibration seal event")
    _same(events[dev]["seal"], _json(run / "heldout_seal.json"), "heldout seal event")
    return {"calibration_seal_precedes_development_events": True,
            "fixed_seal_precedes_heldout_events": True, "events_checked": len(events),
            "scope": "saved ledger consistency, not cryptographic proof of wall-clock chronology"}


def verify_run(run_dir, *, repo=ROOT, prereg_only=False, allow_current_newline_only=False,
               validate_data=False):
    run, repo = Path(run_dir).resolve(), Path(repo).resolve()
    registration, audit = _preregistration(run, repo, allow_current_newline_only)
    config = registration["config"]
    partitions = _partitions(run, config)
    audit["prospective_partitions_and_parity_ids_recomputed"] = True
    if prereg_only:
        return audit
    _require((run / "promotion_decision.json").exists(),
             "full verification requires completed promotion_decision; use --prereg-only while running")
    decision = _json(run / "promotion_decision.json")
    _require(decision.get("status") == "completed", "run is not a completed granularity study")
    elapsed = decision["single_gpu_wall_elapsed_seconds"]
    _require(isinstance(elapsed, (float, int)) and not isinstance(elapsed, bool)
             and np.isfinite(elapsed) and 0 <= elapsed <= config["gpu_wall_budget_seconds"]
             and decision["gpu_budget_ceiling_seconds"] == config["gpu_wall_budget_seconds"],
             "completed execution exceeded or changed its registered wall budget")
    from expert_granularity_evaluation import assert_inherited_equal

    from depth_router.expert_granularity import (
        cycle_gate,
        original_candidate_indices,
        paired_coverage_difference,
        paired_portfolio_difference,
        portfolio_coverage,
        select_calibration_portfolio,
        select_development_fixed,
        summarize_portfolio,
    )

    manifest = [json.loads(line) for line in
                (run / "registered_sources" / config["manifest"]).read_text().splitlines()]
    _require(len(manifest) == 1600 and not any(row["split"] == "confirmation" for row in manifest),
             "derived manifest must exclude all prior confirmation contexts")
    if validate_data:
        from depth_router.granularity_data import derive_granularity_manifest
        rows, data_audit, quarantine = derive_granularity_manifest(repo)
        _same(rows, manifest, "derived manifest")
        _same(data_audit, _json(run / "verified_data_audit.json"), "prospective data audit")
        _same(quarantine, _json(run / "quarantined_confirmation.json"), "quarantine")
    calibration_seal, heldout_seal = _json(run / "calibration_seal.json"), _json(run / "heldout_seal.json")
    _verify_seal(run, calibration_seal, {"partitions.json", "preregistration.json", "portfolio_selection.json"},
                 "calibration seal")
    _verify_seal(run, heldout_seal, {*calibration_seal, "calibration_seal.json", "fixed_selection.json"},
                 "heldout seal")
    portfolios, fixed = _json(run / "portfolio_selection.json"), _json(run / "fixed_selection.json")
    _require(set(portfolios) == set(map(str, SEEDS)), "all three calibration portfolios are required")
    metrics = {"original": {}, "selected": {str(seed): {} for seed in SEEDS},
               "full_oracle": {str(seed): {} for seed in SEEDS}}
    paired = {"calibration": {"per_seed": {}}}
    expected_fixed = {"original": None, "selected": {}, "full_oracle": {}}
    inherited_results, matrices, captures = {}, {}, {}
    old_ids, all_ids = original_candidate_indices(), np.arange(1820)
    parent = np.asarray(partitions["parent"])
    labels = {seed: np.asarray(partitions["children"][str(seed)], dtype=np.int64) for seed in SEEDS}
    control_order = np.concatenate([np.flatnonzero(parent == index) for index in range(8)])
    bootstrap = {"bootstrap_repetitions": config["bootstrap_repetitions"],
                 "bootstrap_seed": config["bootstrap_seed"]}
    expected_paths = {f"matrix-{split}-{suffix}.json" for split in SPLITS
                      for suffix in ("original", *(f"seed-{seed}" for seed in SEEDS))}
    actual_paths = {path.name for path in run.glob("matrix-*.json")}
    _require(actual_paths == expected_paths, "completed study must contain exactly 16 matrix manifests")
    for split in SPLITS:
        records, captures[split] = _records(run, split, manifest, config)
        name = f"matrix-{split}-original.json"
        original, matrices[name] = _matrix(run / name, records, labels[17], old_ids, control_order, split)
        if split == "development":
            expected_fixed["original"] = select_development_fixed(original, old_ids)
        if split != "calibration":
            metrics["original"][split] = summarize_portfolio(
                original, old_ids, expected_fixed["original"], split=split, **bootstrap)
            paired[split] = {"per_seed": {}}
        inherited_results[split], selected_coverage = {}, []
        for seed in SEEDS:
            key, name = str(seed), f"matrix-{split}-seed-{seed}.json"
            matrix, matrices[name] = _matrix(run / name, records, labels[seed], all_ids, control_order, split)
            assert_inherited_equal(matrix, original, old_ids)
            inherited_results[split][key] = {
                "exact_masks_and_all_raw_measurements": True,
                "original_coverage": float(portfolio_coverage(original, old_ids).mean()),
                "full_oracle_coverage": float(portfolio_coverage(matrix, all_ids).mean())}
            if split == "calibration":
                _same(select_calibration_portfolio(matrix), portfolios[key], f"calibration portfolio/{key}")
            else:
                ids = portfolios[key]["candidate_ids"]
                if split == "development":
                    expected_fixed["selected"][key] = select_development_fixed(matrix, ids)
                    expected_fixed["full_oracle"][key] = select_development_fixed(matrix, all_ids)
                metrics["selected"][key][split] = summarize_portfolio(
                    matrix, ids, expected_fixed["selected"][key], split=split, **bootstrap)
                metrics["full_oracle"][key][split] = summarize_portfolio(
                    matrix, all_ids, expected_fixed["full_oracle"][key], split=split, **bootstrap)
                paired[split]["per_seed"][key] = paired_portfolio_difference(
                    matrix, ids, original, old_ids, **bootstrap)
                selected_coverage.append(portfolio_coverage(matrix, ids))
            del matrix
        if split != "calibration":
            coverage = np.stack(selected_coverage)
            paired[split]["aggregate"] = paired_coverage_difference(
                coverage, np.broadcast_to(portfolio_coverage(original, old_ids), coverage.shape),
                original["context_ids"], **bootstrap)
        del original
    _same(expected_fixed, fixed, "development fixed comparators")
    _same(metrics, _json(run / "metrics.json"), "all original/selected/full-oracle metrics")
    _same(paired, _json(run / "paired.json"), "paired rescue/loss/net intervals")
    _same(inherited_results, _json(run / "inherited_mask_validation.json"), "inherited exact raw parity")
    parity = _json(run / "full_model_cached_validation.json")
    _require(set(parity) == set(map(str, SEEDS)), "parity checks miss a refinement seed")
    for seed in SEEDS:
        expected_ids = partitions["parity_ids"][str(seed)] + [-1, -2]
        checks = parity[str(seed)]
        _require([row["candidate_id"] for row in checks] == expected_ids
                 and [row["kind"] for row in checks] == ["cheap"] * (len(checks) - 2) + ["control", "permuted_control"]
                 and all(row["states"] == config["batch_size"] and row["exact_logits"] is True
                         and row["max_abs_logit_error"] == 0 for row in checks),
                 f"full-model/cache parity failed or lacks preregistered masks/controls: {seed}")
    gate = cycle_gate(metrics["selected"], paired["heldout_ood"]["aggregate"])
    _same(gate, {key: decision[key] for key in gate}, "capability gate")
    _require(decision["capability_pass"] == gate["point_pass"] and decision["promote"] is False
             and decision["global_bandwidth_solver_promotion"] is False
             and decision["runtime_training"] == "not_run_out_of_scope"
             and decision["runtime_timing"] == "not_run_out_of_scope"
             and decision["measured_latency"] is None and decision["measured_dram_bytes"] is None,
             "terminal scope or capability decision differs")
    weights_before = _json(run / "environment.json")["original_weight_sha256"]
    _require(isinstance(weights_before, dict) and weights_before
             and all(isinstance(name, str) and name and _valid_hash(digest)
                     for name, digest in weights_before.items()),
             "original tensor hash inventory is empty or invalid")
    _same(_json(run / "final_weight_sha256.json"), weights_before,
          "unchanged original model tensor hashes")
    _require(decision["original_weights_unchanged"] is True
             and decision["prior_negative_results_unchanged"] is True, "weight/prior-result assertion missing")
    audit.update({"scope": "complete_granularity_integrity", "heldout_artifacts_opened": True,
                  "fresh_data_exclusions_and_quarantine_recomputed": validate_data,
                  "calibration_portfolios_and_development_fixed_choices_recomputed": True,
                  "all_metrics_paired_intervals_and_capability_gate_recomputed": True,
                  "inherited_all_scalar_fields_masks_and_controls_equal": True,
                  "matrices": matrices, "captures": captures, "matrix_manifests_checked": len(matrices),
                  "candidate_cells_checked": sum(value["states"] * value["candidates"] for value in matrices.values()),
                  "full_model_parity_checks": sum(map(len, parity.values())),
                  "original_weights_before_after_hashes_equal": True,
                  "weight_claim_scope": "saved complete model tensor hashes; no language-model weight reload",
                  "registered_wall_budget_respected": True,
                  "chronology": _chronology(run), "capability_pass": gate["point_pass"],
                  "no_router_or_economic_promotion": True})
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--prereg-only", action="store_true")
    parser.add_argument("--allow-current-newline-only", action="store_true")
    parser.add_argument("--validate-data", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = verify_run(args.run_dir, repo=args.repo, prereg_only=args.prereg_only,
                            allow_current_newline_only=args.allow_current_newline_only,
                            validate_data=args.validate_data)
    except (AssertionError, ValueError, KeyError, OSError, TypeError, RuntimeError) as error:
        result = {"pass": False, "error_type": type(error).__name__, "error": str(error),
                  "language_model_loaded": False, "gpu_used": False}
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output and result["pass"]:
        _require(not args.output.exists(), "refusing to overwrite an existing verification artifact")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(serialized.encode())
    sys.stdout.write(serialized)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
