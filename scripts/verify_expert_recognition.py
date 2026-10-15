"""Read-only, CPU-only verification of frozen recognition-study artifacts.

Full verification requires a terminal promotion decision. --prereg-only is safe
during execution and never opens fresh model-output artifacts. No language model
is loaded; optional local activation caches allow CPU selector-score replay.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from verify_expert_audit import (  # noqa: E402
    COMBINATIONS,
    _child,
    _json,
    _partition,
    _require,
    _sha,
    _valid_hash,
    _verify_matrix,
)


def _same(actual, expected, where="result"):
    """Compare independently recalculated JSON, tolerating only float roundoff."""
    if isinstance(actual, dict):
        _require(isinstance(expected, dict) and actual.keys() == expected.keys(),
                 f"object fields differ: {where}")
        for name in actual:
            _same(actual[name], expected[name], f"{where}.{name}")
    elif isinstance(actual, list):
        _require(isinstance(expected, list) and len(actual) == len(expected),
                 f"list length differs: {where}")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            _same(left, right, f"{where}[{index}]")
    elif isinstance(actual, (float, np.floating)):
        _require(isinstance(expected, (int, float))
                 and np.isclose(actual, expected, rtol=1e-10, atol=1e-12),
                 f"recomputed numeric result differs: {where}: {actual} != {expected}")
    else:
        _require(actual == expected, f"recomputed result differs: {where}")


def _npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _registration(run, repo, allow_current_newline_only):
    registration = _json(run / "preregistration.json")
    _require(registration.get("before_model_load_fit_and_fresh_outputs") is True,
             "registration lacks its prospective assertion")
    sources = registration["source_sha256"]
    _require(bool(sources), "registered sources are missing")
    newline_only = []
    for name, expected in sources.items():
        _require(_valid_hash(expected), f"invalid registered SHA256: {name}")
        snapshot = _child(run / "registered_sources", name)
        current = _child(repo, name)
        _require(_sha(snapshot) == expected, f"registered snapshot changed: {name}")
        if _sha(current) != expected:
            equivalent = (snapshot.read_bytes().replace(b"\r\n", b"\n")
                          == current.read_bytes().replace(b"\r\n", b"\n"))
            _require(allow_current_newline_only and equivalent,
                     f"current source changed: {name}; newline_only={equivalent}")
            newline_only.append(name)
    config = registration["config"]
    required = {"configs/expert_recognition.json", config["manifest"],
                config["frozen_partition"], config["old_fixed_selection"],
                "EXPERT_RECOGNITION_PREREGISTRATION.md"}
    _require(required <= sources.keys(), "preregistration misses required inputs")
    _same(_json(run / "registered_sources/configs/expert_recognition.json"), config, "config")
    _require(config["seeds"] == [17, 29, 43] and config["primary_runtime_seed"] == 17,
             "router seeds or prospective runtime seed changed")
    for key in ("frozen_partition", "old_fixed_selection"):
        _require(sources[config[key]] == config[key + "_sha256"], f"discovery hash changed: {key}")
    return registration, {
        "pass": True, "scope": "preregistration_only", "registered_files_checked": len(sources),
        "registered_utc": registration["utc"], "snapshot_source_hashes_match": True,
        "current_source_hashes_match_exactly": not newline_only,
        "current_source_newline_only_differences": newline_only,
        "current_newline_only_differences_explicitly_allowed": allow_current_newline_only,
        "heldout_artifacts_opened": False, "language_model_loaded": False, "gpu_used": False,
        "report_or_verifier_part_of_scientific_seal": False,
    }


def _states_and_matrix(run, split, labels, manifest, config):
    records = [json.loads(line) for line in (run / f"states-{split}.jsonl").read_text().splitlines()]
    expected_rows = {row["id"]: row for row in manifest if row["split"] == split}
    keys = []
    for index, row in enumerate(records):
        _require(row["id"] in expected_rows and row["state_index"] == index and row["split"] == split,
                 f"state identity/order differs: {split} {index}")
        _require(row["position"] in config["continuation_positions"], f"unexpected position: {split}")
        _require(row["task"] == expected_rows[row["id"]]["task"]
                 and row["stratum"] == expected_rows[row["id"]]["stratum"],
                 f"state/source metadata differs: {split} {index}")
        _require(row["prefix_token_ids"] and all(type(token) is int and 0 <= token < 151936
                  for token in row["prefix_token_ids"]), f"invalid prefix tokens: {split}")
        keys.append((row["id"], row["position"]))
    _require(len(set(keys)) == len(keys), f"duplicate context-position: {split}")
    audit = _verify_matrix(run / f"matrix-{split}.npz", labels, records)
    matrix = _npz(run / f"matrix-{split}.npz")
    _require(np.array_equal(matrix["control_token"], matrix["dense_token"])
             and np.all(matrix["control_kl"] == 0), f"sorted dense control failed: {split}")
    capture = _json(run / f"capture-{split}.json")
    observed = {row["id"] for row in records}
    _require(capture["states"] == len(records) and capture["contexts_requested"] == len(expected_rows)
             and capture["contexts_evaluated"] == len(observed)
             and set(capture["zero_state_context_ids"]) == expected_rows.keys() - observed,
             f"capture count/exclusion mismatch: {split}")
    cache_path = run / f"cache-{split}.pt"
    if cache_path.exists():
        _require(_sha(cache_path) == capture["cache_sha256"], f"activation cache hash changed: {split}")
    audit.update({"state_metadata_sha256": _sha(run / f"states-{split}.jsonl"),
                  "activation_cache_available_and_hash_checked": cache_path.exists(),
                  "contexts_evaluated": len(observed), "zero_state_contexts": len(expected_rows) - len(observed)})
    return records, matrix, audit


def _verify_policies(run, split, matrix, choices, config, cache_available):
    import torch
    from expert_recognition import slice_matrix

    from depth_router.expert_recognition import (
        actions_from_scores,
        load_checkpoint,
        oracle_actions,
        paired_policy_difference,
        policy_metrics,
        predict_scores,
        random_actions,
        select_policy,
        verify_frozen_choice,
    )

    raw, saved = _npz(run / f"policies-{split}.npz"), _json(run / f"policies-{split}.json")
    for field in ("context_ids", "positions"):
        _require(np.array_equal(raw[field], matrix[field]), f"policy row alignment differs: {split}/{field}")
    params = {"bootstrap_repetitions": config["bootstrap_repetitions"],
              "bootstrap_seed": config["bootstrap_seed"]}
    records = [json.loads(line) for line in (run / f"states-{split}.jsonl").read_text().splitlines()]
    strata = {}
    for field in ("task", "stratum"):
        for value in sorted({row[field] for row in records}):
            strata[f"{field}:{value}"] = np.array([row[field] == value for row in records])
    for lower, upper in itertools.pairwise((0.0, 0.1, 0.5, 1.0, 2.0, float("inf"))):
        strata[f"dense_margin:[{lower},{upper})"] = (
            (matrix["dense_margin"] >= lower) & (matrix["dense_margin"] < upper))
    metrics, actions, by_stratum, score_replays = {}, {}, {}, 0
    x = None
    if cache_available:
        groups = torch.load(run / f"cache-{split}.pt", map_location="cpu", weights_only=True)
        x = torch.cat([group["x"][group["valid"]] for group in groups])[:, 0]
        _require(len(x) == len(matrix["kl"]), f"cache state count differs: {split}")
    for family in ("portfolio", "binary"):
        metrics[family], actions[family], by_stratum[family] = {}, {}, {}
        for seed in config["seeds"]:
            key, choice = str(seed), choices[family][str(seed)]
            verify_frozen_choice(choice)
            scores, selected = raw[f"{family}_{seed}_scores"], raw[f"{family}_{seed}_actions"]
            _require(scores.shape == (len(matrix["kl"]), 28 if family == "portfolio" else 1),
                     f"score shape differs: {split}/{family}/{seed}")
            recomputed_actions = actions_from_scores(scores, choice["threshold"],
                                                     fixed_index=choice["fixed_index"])
            _require(np.array_equal(selected, recomputed_actions), f"chosen actions differ: {split}/{family}/{seed}")
            actions[family][key] = selected
            metrics[family][key] = policy_metrics(selected, matrix, matrix["context_ids"], **params)
            by_stratum[family][key] = {
                name: policy_metrics(selected[selection], slice_matrix(matrix, selection),
                                     matrix["context_ids"][selection], bootstrap_repetitions=0)
                for name, selection in strata.items() if selection.any()}
            if x is not None:
                model = load_checkpoint(run / f"selector-{family}-{choice['architecture']}-{seed}.pt")
                replayed = predict_scores(model, x)
                _require(np.allclose(replayed, scores, rtol=1e-5, atol=2e-6),
                         f"CPU checkpoint score replay differs: {split}/{family}/{seed}")
                score_replays += 1
            if split == "development":
                all_scores = _npz(run / f"dev-scores-{family}-{seed}.npz")
                recomputed_choice = select_policy(all_scores, matrix, matrix["context_ids"],
                                                  fixed_index=choice["fixed_index"])
                _same(recomputed_choice, choice, f"frozen choice/{family}/{seed}")
                _require(np.array_equal(scores, all_scores[choice["architecture"]]),
                         f"selected development scores differ: {family}/{seed}")
    diagnostics = {
        "original_dense": np.full(len(matrix["kl"]), 28, dtype=np.int64),
        "always_fixed": np.full(len(matrix["kl"]), config["fixed_candidate_index"], dtype=np.int64),
        "teacher_portfolio_oracle": oracle_actions(matrix),
        "teacher_fixed_oracle": oracle_actions(matrix, fixed_index=config["fixed_candidate_index"]),
    }
    per_seed = {}
    for seed in config["seeds"]:
        key = str(seed)
        for name, match in (("random_subset_same_positions", True), ("random_subset_random_positions", False)):
            diagnostics[f"{name}_{seed}"] = random_actions(actions["portfolio"][key], seed=seed,
                                                           match_positions=match)
        per_seed[key] = paired_policy_difference(actions["portfolio"][key], actions["binary"][key],
                                                 matrix, matrix["context_ids"], **params)
    diagnostic_metrics = {}
    for name, selected in diagnostics.items():
        _require(np.array_equal(raw[name + "_actions"], selected), f"diagnostic actions differ: {split}/{name}")
        diagnostic_metrics[name] = policy_metrics(selected, matrix, matrix["context_ids"], **params)
    paired = paired_policy_difference(
        np.stack([actions["portfolio"][str(seed)] for seed in config["seeds"]]),
        np.stack([actions["binary"][str(seed)] for seed in config["seeds"]]),
        matrix, matrix["context_ids"], **params)
    _same({"metrics": metrics, "diagnostics": diagnostic_metrics,
           "paired_portfolio_minus_binary_by_seed": per_seed,
           "paired_portfolio_minus_binary_seed_mean": paired,
           "descriptive_task_enterprise_and_dense_margin_strata": by_stratum,
           "teacher_oracles_are_deployable": False},
          saved, f"policies/{split}")
    return metrics, paired, score_replays


def _verify_runtime(run, runtime, config):
    from expert_recognition_timing import _summaries

    from depth_router.expert_recognition import policy_metrics

    if not isinstance(runtime, dict) or not runtime.get("policies"):
        return {"policies_checked": 0, "timing_scopes_checked": 0}
    scopes_checked = 0
    for family, result in runtime["policies"].items():
        for split, saved in result["quality"].items():
            raw = _npz(run / "runtime" / f"quality-{split}-{family}.npz")
            states = [json.loads(line) for line in
                      (run / "runtime" / f"states-{split}.jsonl").read_text().splitlines()]
            _require(np.array_equal(raw["context_ids"], [row["id"] for row in states])
                     and np.array_equal(raw["positions"], [row["position"] for row in states]),
                     f"runtime state alignment differs: {family}/{split}")
            matrix = {"kl": np.repeat(raw["kl"][:, None], 28, 1),
                      "agreement": np.repeat(raw["agreement"][:, None], 28, 1)}
            summary = policy_metrics(raw["actions"], matrix, raw["context_ids"],
                bootstrap_repetitions=config["bootstrap_repetitions"], bootstrap_seed=config["bootstrap_seed"])
            summary.update({"batch_size": 1, "genuine_batch1_dense_prefixes": True})
            _same(summary, saved, f"runtime metrics/{family}/{split}")
        if result.get("timing_status") != "measured":
            continue
        for scope, filename in (("isolated", "timing_blocks.csv"),
                                ("cached_decode", "cached_decode_blocks.csv")):
            if scope == "cached_decode" and result[scope].get("failure") is not None:
                _require(not result[scope]["rows"], "failed full-model timing must not claim savings")
                continue
            directory = run / "runtime" / family / scope
            with (directory / filename).open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            for row in rows:
                row["wall_ms_per_state"] = float(row["wall_ms_per_state"])
                row["block"] = int(row["block"])
            if scope == "cached_decode":
                rows = [{"method": method, "block": block,
                         "scope": "complete_model_cached_decode_batch1",
                         "wall_ms_per_state": float(np.mean([row["wall_ms_per_state"] for row in rows
                            if row["method"] == method and row["block"] == block]))}
                        for method in sorted({row["method"] for row in rows})
                        for block in sorted({row["block"] for row in rows})]
            recomputed = _summaries(rows, seed=config["runtime"]["seed"],
                                   resamples=config["runtime"]["bootstrap_resamples"])
            _same(recomputed, result[scope]["rows"], f"timing/{family}/{scope}")
            scopes_checked += 1
    return {"policies_checked": len(runtime["policies"]), "timing_scopes_checked": scopes_checked,
            "scope": "raw numerical/timing artifact recomputation; no new GPU measurements"}


def verify_run(run_dir, *, repo=ROOT, prereg_only=False, allow_current_newline_only=False,
               validate_data=False):
    run, repo = Path(run_dir).resolve(), Path(repo).resolve()
    registration, audit = _registration(run, repo, allow_current_newline_only)
    if prereg_only:
        return audit
    _require((run / "promotion_decision.json").exists(),
             "full verification requires terminal promotion_decision; use --prereg-only while running")
    from expert_construction import summarize
    from expert_recognition import confirmation_decision, recognition_decision

    from depth_router.expert_recognition import load_checkpoint

    config, decision = registration["config"], _json(run / "promotion_decision.json")
    _require(decision["original_weights_unchanged"] is True
             and decision["prior_construction_negative_result_unchanged"] is True,
             "terminal decision lacks unchanged-weight/prior-result attestations")
    labels = _partition(_json(run / "frozen_partition.json"), "C-29")
    _require(np.array_equal(labels, _json(run / "registered_sources" / config["frozen_partition"])),
             "frozen discovery partition differs from its registered snapshot")
    fixed = _json(run / "frozen_fixed_selection.json")
    _same(fixed, _json(run / "registered_sources" / config["old_fixed_selection"])[config["partition_name"]],
          "discovery fixed choice")
    manifest = [json.loads(line) for line in
                (run / "registered_sources" / config["manifest"]).read_text().splitlines()]
    if validate_data:
        from depth_router.recognition_data import (
            load_recognition_exclusions,
            validate_recognition_manifest,
        )
        historical, prior, _ = load_recognition_exclusions(repo)
        data = validate_recognition_manifest(manifest, historical_prompts=historical,
                                              prior_expert_rows=prior)
        _require(data["manifest_sha256"] == _json(repo / "data/expert_recognition/data_audit.json")["manifest_sha256"],
                 "fresh-data exclusion audit changed")
    records, matrix, matrix_audit = _states_and_matrix(run, "confirmation", labels, manifest, config)
    confirmation = confirmation_decision(matrix, records,
        [row for row in manifest if row["split"] == "confirmation"], fixed, config)
    _same(confirmation, _json(run / "confirmation_decision.json"), "confirmation")
    _require(decision["confirmation_pass"] == confirmation["point_pass"], "confirmation decision differs")
    parity = _json(run / "full_model_cached_validation.json")
    _require([row["experts"] for row in parity] == [list(combo) for combo in COMBINATIONS] + [list(range(8))]
             and all(row["exact_logits"] is True and row["max_abs_logit_error"] == 0
                     and row["states"] == config["batch_size"] for row in parity),
             "full-model/cache parity control failed or is incomplete")
    audit.update({"scope": "complete_recognition_integrity", "matrices": {"confirmation": matrix_audit},
                  "confirmation_recomputed": True, "full_model_parity_checks": len(parity),
                  "fresh_data_exclusions_recomputed": validate_data, "selector_checkpoint_cpu_replays": 0,
                  "original_weight_claim_scope": "saved execution assertion; language-model weights not reloaded"})
    if not confirmation["point_pass"]:
        forbidden = [path.name for pattern in ("selector-*.pt", "policies-*.npz", "matrix-*.npz")
                     for path in run.glob(pattern) if path.name != "matrix-confirmation.npz"]
        _require(not forbidden and not decision["promote"], f"work continued after failed confirmation: {forbidden}")
        audit["confirmation_failure_stop_verified"] = True
        return audit
    seal = _json(run / "heldout_seal.json")
    required = {"frozen_policy_choices.json", "frozen_partition.json", "frozen_fixed_selection.json",
                "preregistration.json"}
    required |= {f"selector-{family}-{architecture}-{seed}.pt" for family in ("portfolio", "binary")
                 for architecture in config["selector"]["architectures"] for seed in config["seeds"]}
    _require(required <= seal.keys(), "heldout seal misses frozen inputs/checkpoints")
    for name, expected in seal.items():
        _require(_valid_hash(expected) and _sha(_child(run, name)) == expected, f"heldout seal changed: {name}")
        if name.startswith("selector-"):
            load_checkpoint(_child(run, name))
    choices = _json(run / "frozen_policy_choices.json")
    metrics = {family: {str(seed): {} for seed in config["seeds"]} for family in ("portfolio", "binary")}
    paired = {}
    for split in ("calibration", "development", "heldout_matched", "heldout_ood"):
        _, matrix, checked = _states_and_matrix(run, split, labels, manifest, config)
        audit["matrices"][split] = checked
        if split == "calibration":
            continue
        _same(summarize(matrix, fixed, split, config), _json(run / f"capability-{split}.json"),
              f"capability/{split}")
        current, paired[split], replays = _verify_policies(
            run, split, matrix, choices, config, checked["activation_cache_available_and_hash_checked"])
        audit["selector_checkpoint_cpu_replays"] += replays
        for family in metrics:
            for seed in config["seeds"]:
                metrics[family][str(seed)][split] = current[family][str(seed)]
    _same(metrics, _json(run / "recognition_metrics.json"), "aggregate metrics")
    _same(paired, _json(run / "paired_portfolio_minus_binary.json"), "aggregate paired contrast")
    decisions = recognition_decision(metrics, paired, config)
    _same(decisions, _json(run / "recognition_decision.json"), "recognition gate")
    _same(decisions, decision["recognition"], "terminal recognition gate")
    _require(all(decisions[family]["eligible_for_runtime_quality"]
                 for family in decision["runtime"]["policies"]),
             "runtime was attempted for a recognition-ineligible policy family")
    audit["runtime"] = _verify_runtime(run, decision["runtime"], config)
    promoted = [family for family, result in decision["runtime"]["policies"].items()
                if result.get("timing_status") == "measured" and all(
                    any(row["resolved_savings"] for row in result[scope].get("rows", []))
                    for scope in ("isolated", "cached_decode"))]
    _require(decision["promote"] == bool(promoted)
             and decision["promoted_policy_families"] == promoted, "economic promotion differs")
    audit.update({"heldout_artifacts_opened": True, "heldout_seal_entries_checked": len(seal),
                  "policy_metrics_gates_and_paired_intervals_recomputed": True,
                  "cpu_score_replay_tolerance": {"rtol": 1e-5, "atol": 2e-6},
                  "promotion_decision_recomputed": True})
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
    except (ValueError, KeyError, OSError, TypeError, RuntimeError) as error:
        result = {"pass": False, "error_type": type(error).__name__, "error": str(error),
                  "language_model_loaded": False, "gpu_used": False}
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        _require(not args.output.exists(), "refusing to overwrite an audit artifact")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(serialized.encode())
    sys.stdout.write(serialized)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
