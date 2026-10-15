"""One preregistered causal recognition cycle for the frozen C-29 partition."""

from __future__ import annotations

import argparse
import inspect
import itertools
import json
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from expert_construction import (
    Budget,
    Log,
    collect_states,
    evaluate_partition,
    final_logits,
    sha_file,
    summarize,
    tensor_sha256,
    validate_cached_candidates,
    validate_partition,
    verify_seal,
    write_json,
)
from torch import nn
from torch.nn import functional as F

from depth_router.expert_recognition import (
    DENSE_ACTION,
    acceptable_labels,
    actions_from_scores,
    oracle_actions,
    paired_policy_difference,
    policy_metrics,
    predict_scores,
    random_actions,
    repeatability_gate,
    save_checkpoint,
    select_policy,
    train_selector_pair,
    verify_frozen_choice,
)

ROOT = Path(__file__).resolve().parents[1]
EVALUATION_SPLITS = ("development", "heldout_matched", "heldout_ood")
FAMILIES = ("portfolio", "binary")


def slice_matrix(matrix, selected):
    """Select states, preserving non-state candidate/partition metadata."""
    count = len(matrix["kl"])
    return {key: value[selected] if key not in ("selected_experts", "partition")
            and value.ndim > 0 and value.shape[0] == count else value
            for key, value in matrix.items()}


def confirmation_decision(matrix, records, rows, fixed, config):
    strata = {row["id"]: row["confirmation_stratum"] for row in rows}
    summaries = {}
    for stratum in ("matched", "ood"):
        selected = np.array([strata[row["id"]] == stratum for row in records])
        if not selected.any():
            raise ValueError(f"confirmation missing {stratum} states")
        summaries[stratum] = summarize(slice_matrix(matrix, selected), fixed,
                                       f"confirmation_{stratum}", config)
    passed = all(value["engineering_gate"]["point_pass"] for value in summaries.values())
    return {"point_pass": passed, "strata": summaries,
            "statistically_resolved_pass": all(
                value["engineering_gate"]["statistically_resolved_pass"]
                for value in summaries.values()),
            "confirmation_is_training_data": False,
            "failure_action": "stop_without_switching_partition_or_consuming_other_splits"}


def recognition_decision(metrics, paired, config):
    decisions = {}
    advantage = all(paired[split]["cheap_fraction"]["lower"] > 0
                    for split in ("heldout_matched", "heldout_ood"))
    for family in FAMILIES:
        repeated = repeatability_gate(
            {seed: metrics[family][str(seed)] for seed in config["seeds"]})
        quality = (repeated["all_seed_quality_point_pass"]
                   and repeated["all_seed_nonzero_cheap_use"])
        decisions[family] = {
            **repeated,
            "portfolio_advantage_required": family == "portfolio",
            "portfolio_advantage_pass": advantage if family == "portfolio" else None,
            "eligible_for_runtime_quality": quality and (family == "binary" or advantage),
            "primary_runtime_seed": config["primary_runtime_seed"],
        }
    return decisions


def verify_sources(output):
    registration = json.loads((output / "preregistration.json").read_text())
    for name, expected in registration["source_sha256"].items():
        if sha_file(ROOT / name) != expected:
            raise RuntimeError(f"registered execution source changed: {name}")
        if sha_file(output / "registered_sources" / name) != expected:
            raise RuntimeError(f"registered snapshot changed: {name}")


def register(config_path, output):
    config = json.loads(config_path.read_text())
    if config["seeds"] != [17, 29, 43] or config["primary_runtime_seed"] != 17:
        raise ValueError("this registered cycle fixes all three router seeds and runtime seed17")
    if config["confirmation_gates"] != {
        "coverage": 0.8, "portfolio_advantage": 0.05, "each_stratum": True,
    } or config["recognition_gates"] != {
        "agreement": 0.99, "mean_kl": 0.01, "all_three_seeds": True,
        "positive_cheap_fraction_each_split": True,
        "portfolio_minus_binary_cheap_fraction_ci_lower_strictly_above": 0.0,
        "advantage_splits": ["heldout_matched", "heldout_ood"],
    }:
        raise ValueError("config gates do not match this preregistered implementation")
    if config["selector"]["thresholds"] != [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 1.0]:
        raise ValueError("config thresholds do not match the frozen grid")
    for key, hash_key in (("frozen_partition", "frozen_partition_sha256"),
                          ("old_fixed_selection", "old_fixed_selection_sha256")):
        if sha_file(ROOT / config[key]) != config[hash_key]:
            raise ValueError(f"discovery artifact changed: {key}")
    rows = [json.loads(line) for line in (ROOT / config["manifest"]).read_text().splitlines()]
    if dict(Counter(row["split"] for row in rows)) != config["split_sizes"]:
        raise ValueError("manifest split counts differ from the prospective config")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("manifest context identifiers must be unique")
    files = [config_path, Path(__file__), ROOT / config["manifest"],
             ROOT / config["frozen_partition"], ROOT / config["old_fixed_selection"],
             ROOT / "EXPERT_RECOGNITION_PREREGISTRATION.md",
             ROOT / "experiments/expert_construction.py",
             ROOT / "experiments/expert_recognition_timing.py",
             ROOT / "scripts/run_expert_recognition.ps1",
             ROOT / "scripts/prepare_recognition_data.py",
             ROOT / "requirements-photonic.txt", ROOT / "Dockerfile.photonic",
             ROOT / "data/expert_audit/evaluation_manifest.jsonl"]
    files += sorted((ROOT / "src/depth_router").glob("expert_*.py"))
    files += [ROOT / f"src/depth_router/{name}.py" for name in
              ("recognition_data", "output_gap", "output_gap_data", "fused_swiglu")]
    files += sorted((ROOT / "data/expert_recognition").glob("*.json"))
    files += sorted((ROOT / "data/expert_audit").glob("historical_*.json*"))
    # Output does not exist until every required input has been checked.
    source_hashes = {source.relative_to(ROOT).as_posix(): sha_file(source) for source in files}
    output.mkdir(parents=True, exist_ok=False)
    for source in files:
        snapshot = output / "registered_sources" / source.relative_to(ROOT)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(source.read_bytes())
    write_json(output / "preregistration.json", {
        "utc": datetime.now(timezone.utc).isoformat(), "config": config,
        "source_sha256": source_hashes, "before_model_load_fit_and_fresh_outputs": True,
        "prior_construction_negative_result_unchanged": True,
    })
    fixed = json.loads((ROOT / config["old_fixed_selection"]).read_text())[config["partition_name"]]
    if fixed["index"] != config["fixed_candidate_index"]:
        raise ValueError("frozen fixed candidate changed")
    write_json(output / "frozen_fixed_selection.json", fixed)
    write_json(output / "frozen_partition.json",
               json.loads((ROOT / config["frozen_partition"]).read_text()))
    return config


def features(groups):
    return torch.cat([group["x"][group["valid"]] for group in groups])[:, 0]


def evaluate_split(model, tokenizer, dense, labels, rows, split, config, output, log, budget):
    groups, records, replay = collect_states(
        model, tokenizer, dense, [row for row in rows if row["split"] == split],
        config, output, log, budget)
    matrix = evaluate_partition(model, dense, labels, groups, records,
                                output / f"matrix-{split}.npz", budget, log)
    # Numerical corruption is not an ordinary recognition-negative result.
    if not np.array_equal(matrix["control_token"], matrix["dense_token"]):
        raise AssertionError("sorted all-eight control is not exactly dense in greedy tokens")
    if float(matrix["control_kl"].max()) != 0.0:
        raise AssertionError("sorted all-eight control has nonzero KL")
    return groups, records, replay, matrix


def metrics_for(actions, matrix, config):
    return policy_metrics(actions, matrix, matrix["context_ids"],
                          bootstrap_repetitions=config["bootstrap_repetitions"],
                          bootstrap_seed=config["bootstrap_seed"])


def report_policies(split, x, matrix, models, choices, fixed, config, output):
    metrics, actions = {name: {} for name in FAMILIES}, {name: {} for name in FAMILIES}
    records = [json.loads(line) for line in
               (output / f"states-{split}.jsonl").read_text().splitlines()]
    if len(records) != len(x):
        raise AssertionError("policy reporting state alignment failed")
    strata = {}
    for field in ("task", "stratum"):
        for value in sorted({row[field] for row in records}):
            strata[f"{field}:{value}"] = np.array([row[field] == value for row in records])
    boundaries = (0.0, 0.1, 0.5, 1.0, 2.0, float("inf"))
    for lower, upper in itertools.pairwise(boundaries):
        strata[f"dense_margin:[{lower},{upper})"] = (
            (matrix["dense_margin"] >= lower) & (matrix["dense_margin"] < upper))
    by_stratum = {name: {} for name in FAMILIES}
    raw = {"context_ids": matrix["context_ids"], "positions": matrix["positions"]}
    for family in FAMILIES:
        for seed in config["seeds"]:
            key = str(seed)
            choice = choices[family][key]
            verify_frozen_choice(choice)
            scores = predict_scores(models[family][key][choice["architecture"]], x)
            chosen = actions_from_scores(scores, choice["threshold"],
                                         fixed_index=choice["fixed_index"])
            actions[family][key] = chosen
            metrics[family][key] = metrics_for(chosen, matrix, config)
            by_stratum[family][key] = {
                name: policy_metrics(chosen[selection], slice_matrix(matrix, selection),
                                     matrix["context_ids"][selection], bootstrap_repetitions=0)
                for name, selection in strata.items() if selection.any()}
            raw[f"{family}_{seed}_scores"] = scores
            raw[f"{family}_{seed}_actions"] = chosen
    diagnostics = {}
    for name, chosen in {
        "original_dense": np.full(len(x), DENSE_ACTION, dtype=np.int64),
        "always_fixed": np.full(len(x), fixed["index"], dtype=np.int64),
        "teacher_portfolio_oracle": oracle_actions(matrix),
        "teacher_fixed_oracle": oracle_actions(matrix, fixed_index=fixed["index"]),
    }.items():
        diagnostics[name] = metrics_for(chosen, matrix, config)
        raw[name + "_actions"] = chosen
    paired_by_seed = {}
    for seed in config["seeds"]:
        key = str(seed)
        for name, match in (("random_subset_same_positions", True),
                            ("random_subset_random_positions", False)):
            chosen = random_actions(actions["portfolio"][key], seed=seed, match_positions=match)
            diagnostics[f"{name}_{seed}"] = metrics_for(chosen, matrix, config)
            raw[f"{name}_{seed}_actions"] = chosen
        paired_by_seed[key] = paired_policy_difference(
            actions["portfolio"][key], actions["binary"][key], matrix, matrix["context_ids"],
            bootstrap_repetitions=config["bootstrap_repetitions"],
            bootstrap_seed=config["bootstrap_seed"])
    paired = paired_policy_difference(
        np.stack([actions["portfolio"][str(seed)] for seed in config["seeds"]]),
        np.stack([actions["binary"][str(seed)] for seed in config["seeds"]]),
        matrix, matrix["context_ids"], bootstrap_repetitions=config["bootstrap_repetitions"],
        bootstrap_seed=config["bootstrap_seed"])
    np.savez_compressed(output / f"policies-{split}.npz", **raw)
    write_json(output / f"policies-{split}.json", {
        "metrics": metrics, "diagnostics": diagnostics,
        "paired_portfolio_minus_binary_by_seed": paired_by_seed,
        "paired_portfolio_minus_binary_seed_mean": paired,
        "descriptive_task_enterprise_and_dense_margin_strata": by_stratum,
        "teacher_oracles_are_deployable": False,
    })
    return metrics, paired


class FrozenDecision(nn.Module):
    """Deployment adapter: x only, never receives a teacher or cached label."""

    def __init__(self, selector, choice):
        super().__init__()
        verify_frozen_choice(choice)
        self.selector = selector
        self.threshold = choice["threshold"]
        self.fixed_index = choice["fixed_index"]

    def forward(self, x):
        return self.selector.actions(x, self.threshold, fixed_index=self.fixed_index)


@torch.no_grad()
def runtime_metrics(model, dense, routed, groups, records, config, path, budget):
    """Actual batch1 policy execution, not lookup in batch16 candidate matrices."""
    divergence, agreement, routes = [], [], []
    for group in groups:
        budget.check()
        if not bool(group["valid"][0]):
            continue
        x, base = group["x"].cuda(), group["base"].cuda()
        if x.shape[:2] != (1, 1):
            raise ValueError("runtime quality must use genuinely captured batch1 states")
        reference = final_logits(model, base, group["output"].cuda())
        if not torch.equal(reference, final_logits(model, base, dense(x))):
            raise AssertionError("runtime dense capture changed numerically")
        candidate = final_logits(model, base, routed(x))
        logp, logq = F.log_softmax(reference, -1), F.log_softmax(candidate, -1)
        divergence.append(float((logp.exp() * (logp - logq)).sum(-1).clamp_min(0)))
        agreement.append(bool(reference.argmax(-1) == candidate.argmax(-1)))
        routes.append(int(routed.routes(x).item()))
    if len(routes) != len(records):
        raise AssertionError("runtime metric/state alignment failed")
    chosen = np.asarray(routes, dtype=np.int64)
    # Replicate the actually observed chosen result only for metric-helper access;
    # this is NOT a measured 28-candidate matrix and is never exported as one.
    matrix = {"kl": np.repeat(np.asarray(divergence)[:, None], 28, axis=1),
              "agreement": np.repeat(np.asarray(agreement)[:, None], 28, axis=1),
              "context_ids": np.asarray([row["id"] for row in records])}
    if any(divergence[i] != 0 or not agreement[i] for i in np.flatnonzero(chosen == DENSE_ACTION)):
        raise AssertionError("actual dense fallback must match original dense")
    summary = metrics_for(chosen, matrix, config)
    summary.update({"batch_size": 1, "genuine_batch1_dense_prefixes": True,
                    "source": "fresh_batch1_fullmodel_dense_prefix_capture"})
    np.savez_compressed(path, actions=chosen, kl=np.asarray(divergence),
                        agreement=np.asarray(agreement), context_ids=matrix["context_ids"],
                        positions=np.asarray([row["position"] for row in records]))
    return summary


def run_runtime(model, tokenizer, dense, labels, rows, models, choices, metrics,
                paired, decisions, config, output, log, budget):
    from expert_recognition_timing import (
        RoutedPackedMLP,
        attempt_physical_counters,
        run_cached_decode_timing,
        run_isolated_timing,
        validate_fullmodel_replay,
    )

    eligible = [name for name in FAMILIES if decisions[name]["eligible_for_runtime_quality"]]
    if not eligible:
        return {"status": "skipped_recognition_gate_failed", "policies": {},
                "physical_counter_attempt": "not_attempted_prerequisite_quality_gate"}
    directory = output / "runtime"
    directory.mkdir()
    primary = str(config["primary_runtime_seed"])
    routed = {family: RoutedPackedMLP(
        dense, labels, FrozenDecision(
            models[family][primary][choices[family][primary]["architecture"]],
            choices[family][primary]),
        fixed_candidate=config["fixed_candidate_index"] if family == "binary" else None,
    ).eval() for family in eligible}
    results = {family: {"quality": {}, "parity": {}} for family in eligible}
    matched = None
    for split in EVALUATION_SPLITS:
        verify_sources(output)
        groups, records, replay = collect_states(
            model, tokenizer, dense, [row for row in rows if row["split"] == split],
            {**config, "batch_size": 1}, directory, log, budget)
        for family in eligible:
            results[family]["parity"][split] = validate_fullmodel_replay(
                model, dense, routed[family], replay)
            results[family]["quality"][split] = runtime_metrics(
                model, dense, routed[family], groups, records, config,
                directory / f"quality-{split}-{family}.npz", budget)
            write_json(directory / "runtime_quality.json", results)
        if split == "heldout_matched":
            matched = (groups, records)
        del replay
    for family in eligible:
        quality = results[family]["quality"]
        passed = all(value["quality_gate"]["point_pass"] and value["has_nonzero_cheap_use"]
                     for value in quality.values())
        results[family]["runtime_quality_point_pass"] = passed
        if not passed:
            results[family]["timing_status"] = "skipped_actual_runtime_quality_failed"
            continue
        evidence = {
            "seed_metrics": metrics[family], "runtime_metrics": quality,
            "policy_family": family,
            "portfolio_advantage_ci_low": {
                split: paired[split]["cheap_fraction"]["lower"]
                for split in ("heldout_matched", "heldout_ood")},
            "budget": budget, "config": config["runtime"],
        }
        groups, records = matched
        results[family]["isolated"] = run_isolated_timing(
            dense, routed[family], groups, records, directory / family / "isolated", **evidence)
        results[family]["cached_decode"] = run_cached_decode_timing(
            model, dense, routed[family], records, directory / family / "cached_decode", **evidence)
        results[family]["physical_counters"] = attempt_physical_counters(
            [sys.executable, str(ROOT / "experiments/expert_recognition_timing.py"),
             "--profile-run", str(output), "--family", family], directory / family / "counters",
            timeout_seconds=min(120, max(1, int(budget.seconds - budget.elapsed()))))
        results[family]["timing_status"] = "measured"
        log("runtime_timing_complete", family=family, elapsed_seconds=budget.elapsed())
    write_json(directory / "results.json", results)
    return {"status": "runtime_evaluated", "policies": results}


def run(config, output):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import __version__ as transformers_version

    budget, log = Budget(config["gpu_wall_budget_seconds"]), Log(output)
    try:
        verify_sources(output)
        torch.manual_seed(17)
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = AutoModelForCausalLM.from_pretrained(
            config["model"], revision=config["revision"], torch_dtype=torch.bfloat16,
            attn_implementation=config["attention"], local_files_only=True).cuda().eval()
        model.requires_grad_(False)
        tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["revision"],
                                                 local_files_only=True, padding_side="left")
        tokenizer.truncation_side = "left"
        dense = model.model.layers[-1].mlp
        if dense.gate_proj.out_features != 4864 or len(model.model.layers) != 24:
            raise AssertionError("wrong frozen model architecture")
        original = {name: tensor_sha256(value) for name, value in model.state_dict().items()}
        write_json(output / "environment.json", {
            "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
            "transformers": transformers_version, "cuda": torch.version.cuda,
            "numpy": np.__version__, "python": platform.python_version(),
            "platform": platform.platform(), "attention": model.config._attn_implementation,
            "dtype": str(next(model.parameters()).dtype), "allow_tf32": False,
            "torch_num_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "nvidia_smi": subprocess.run(["nvidia-smi"], capture_output=True, text=True,
                                         check=False).stdout,
            "model_class": type(model).__qualname__,
            "model_source_sha256": sha_file(inspect.getfile(type(model))),
            "original_weight_sha256": original,
        })
        labels = torch.tensor(json.loads((output / "frozen_partition.json").read_text()))
        validate_partition(labels)
        fixed = json.loads((output / "frozen_fixed_selection.json").read_text())
        rows = [json.loads(line) for line in (ROOT / config["manifest"]).read_text().splitlines()]

        def finish(decision):
            verify_sources(output)
            unchanged = all(tensor_sha256(value) == original[name]
                            for name, value in model.state_dict().items())
            if not unchanged:
                raise AssertionError("original model weights changed")
            budget.check()
            decision.update({
                "original_weights_unchanged": unchanged,
                "gpu_budget_ceiling_seconds": budget.seconds,
                "single_gpu_wall_elapsed_seconds": budget.elapsed(),
                "prior_construction_negative_result_unchanged": True,
                "promotion_scope": "this_frozen_recognition_study_only_not_global_bandwidth_solver",
                "global_bandwidth_solver_promotion": False,
                "measured_dram_bytes": None, "free_generation_task_quality": None,
                "cost_note": "cycle wall time is not inference latency or measured bandwidth",
            })
            write_json(output / "promotion_decision.json", decision)
            log("cycle_complete", promote=decision["promote"], elapsed_seconds=budget.elapsed())

        groups, records, replay, matrix = evaluate_split(
            model, tokenizer, dense, labels, rows, "confirmation", config, output, log, budget)
        parity = validate_cached_candidates(model, dense, labels, replay, budget)
        write_json(output / "full_model_cached_validation.json", parity)
        confirmation = confirmation_decision(matrix, records,
            [row for row in rows if row["split"] == "confirmation"], fixed, config)
        write_json(output / "confirmation_decision.json", confirmation)
        log("confirmation_complete", point_pass=confirmation["point_pass"],
            elapsed_seconds=budget.elapsed())
        del groups, records, replay, matrix
        if not confirmation["point_pass"]:
            finish({"confirmation_pass": False, "recognition": "not_run_confirmation_failed",
                    "runtime": "not_run_confirmation_failed", "promote": False,
                    "measured_latency": None,
                    "physical_counter_attempt": "not_attempted_prerequisite_quality_gate"})
            return

        verify_sources(output)
        cal, _, replay, cal_matrix = evaluate_split(
            model, tokenizer, dense, labels, rows, "calibration", config, output, log, budget)
        cal_x, cal_labels = features(cal), acceptable_labels(cal_matrix)
        del cal, replay, cal_matrix
        models = {family: {} for family in FAMILIES}
        checkpoints = []
        recipe = config["selector"]
        for seed in config["seeds"]:
            for family in FAMILIES:
                models[family][str(seed)] = {}
            for architecture in recipe["architectures"]:
                pair = train_selector_pair(
                    cal_x, cal_labels, fixed["index"], architecture=architecture, seed=seed,
                    steps=recipe["steps"], batch_size=recipe["batch_size"],
                    learning_rate=recipe["learning_rate"], device="cuda",
                    step_callback=lambda _step: budget.check())
                for family, selector in zip(FAMILIES, pair[:2]):
                    models[family][str(seed)][architecture] = selector
                    name = f"selector-{family}-{architecture}-{seed}.pt"
                    save_checkpoint(output / name, selector, {"family": family, **pair[2]})
                    checkpoints.append(name)
                write_json(output / f"training-{architecture}-{seed}.json", pair[2])
                log("selector_pair_fitted", seed=seed, architecture=architecture,
                    elapsed_seconds=budget.elapsed())
        del cal_x, cal_labels
        metrics = {family: {str(seed): {} for seed in config["seeds"]} for family in FAMILIES}
        choices, paired, seal = {family: {} for family in FAMILIES}, {}, None
        for split in EVALUATION_SPLITS:
            verify_sources(output)
            if split.startswith("heldout"):
                verify_seal(seal, output)
            groups, _, replay, matrix = evaluate_split(
                model, tokenizer, dense, labels, rows, split, config, output, log, budget)
            x = features(groups)
            write_json(output / f"capability-{split}.json", summarize(matrix, fixed, split, config))
            if split == "development":
                for family in FAMILIES:
                    for seed in config["seeds"]:
                        scores = {architecture: predict_scores(selector, x)
                                  for architecture, selector in models[family][str(seed)].items()}
                        choice = select_policy(scores, matrix, matrix["context_ids"],
                            fixed_index=fixed["index"] if family == "binary" else None)
                        choices[family][str(seed)] = choice
                        np.savez_compressed(output / f"dev-scores-{family}-{seed}.npz", **scores)
                write_json(output / "frozen_policy_choices.json", choices)
                names = ["frozen_policy_choices.json", "frozen_partition.json",
                         "frozen_fixed_selection.json", "preregistration.json", *checkpoints]
                seal = {name: sha_file(output / name) for name in names}
                write_json(output / "heldout_seal.json", seal)
                log("heldout_sealed", seal=seal)
            current, paired[split] = report_policies(
                split, x, matrix, models, choices, fixed, config, output)
            for family in FAMILIES:
                for seed in config["seeds"]:
                    metrics[family][str(seed)][split] = current[family][str(seed)]
            log("recognition_split_complete", split=split, elapsed_seconds=budget.elapsed())
            del groups, x, matrix, replay
        decisions = recognition_decision(metrics, paired, config)
        write_json(output / "recognition_metrics.json", metrics)
        write_json(output / "paired_portfolio_minus_binary.json", paired)
        write_json(output / "recognition_decision.json", decisions)
        verify_seal(seal, output)
        runtime = run_runtime(model, tokenizer, dense, labels, rows, models, choices,
                              metrics, paired, decisions, config, output, log, budget)
        # A measured speedup must clear both scope-specific comparisons to graph dense.
        promoted = []
        for family, result in runtime["policies"].items():
            if result.get("timing_status") != "measured":
                continue
            scopes = [result["isolated"], result["cached_decode"]]
            if all(any(row["resolved_savings"] for row in scope.get("rows", []))
                   for scope in scopes):
                promoted.append(family)
        finish({"confirmation_pass": True, "recognition": decisions, "runtime": runtime,
                "promote": bool(promoted), "promoted_policy_families": promoted,
                "measured_latency": "see runtime timing artifacts" if any(
                    result.get("timing_status") == "measured" and result.get("isolated", {}).get("rows")
                    for result in runtime["policies"].values()) else None})
    except Exception as error:
        write_json(output / "incomplete.json", {
            "error_type": type(error).__name__, "error": str(error), "promote": False,
            "single_gpu_wall_elapsed_seconds": budget.elapsed(),
        })
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/expert_recognition.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    config = register(args.config.resolve(), output)
    run(config, output)


if __name__ == "__main__":
    main()
