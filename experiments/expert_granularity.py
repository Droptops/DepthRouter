"""One prospective equal-28-mask-budget, frozen-parent granularity study."""

from __future__ import annotations

import argparse
import inspect
import json
import platform
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from expert_construction import (
    Budget,
    Log,
    collect_states,
    sha_file,
    tensor_sha256,
    verify_seal,
    write_json,
)
from expert_granularity_evaluation import evaluate_candidates, validate_cached_candidates

from depth_router.expert_granularity import (
    array_sha256,
    candidate_combinations,
    candidate_schema_sha256,
    cycle_gate,
    original_candidate_indices,
    paired_coverage_difference,
    paired_portfolio_difference,
    partition_indices,
    portfolio_coverage,
    select_calibration_portfolio,
    select_development_fixed,
    split_parent_experts,
    summarize_portfolio,
    validate_refinement,
)
from depth_router.granularity_data import derive_granularity_manifest, serialized_manifest_bytes

ROOT = Path(__file__).resolve().parents[1]
EVALUATION_SPLITS = ("development", "heldout_matched", "heldout_ood")


def validate_config(config):
    expected = {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        "base_commit": "1be53fefd37c344dbc4cdc44bb708e3e226ae0e9",
        "manifest": "data/expert_granularity/evaluation_manifest.jsonl",
        "frozen_partition": "artifacts/expert_construction/20261014-bounded-v1/partition-C-29.json",
        "frozen_partition_sha256": "f827d04b841f1d1997cd700f6ee95e9c5d86423c4d5048fb0bb71c5f103b4a5d",
        "dtype": "bfloat16", "attention": "sdpa", "batch_size": 16,
        "max_prompt_tokens": 512, "continuation_positions": [1, 3, 7, 11],
        "split_sizes": {"calibration": 768, "development": 128,
                        "heldout_matched": 512, "heldout_ood": 192},
        "fit": "calibration_only_greedy_28_mask_selection_no_gradients",
        "seeds": [17, 29, 43], "parents": 8, "children": 16, "active_children": 12,
        "portfolio_size": 28, "candidate_count": 1820, "candidate_tile_size": 28,
        "archive_candidate_chunk_size": 256, "parity_new_masks_per_seed": 16,
        "gpu_wall_budget_seconds": 14400, "bootstrap_repetitions": 2000,
        "bootstrap_seed": 17, "runtime_training_or_timing": False,
        "gates": {"same_greedy_token": True, "local_kl_strictly_less_than": 0.01,
                  "coverage": 0.8, "portfolio_advantage": 0.05,
                  "all_three_seeds_all_evaluation_splits": True,
                  "seed_mean_paired_ood_coverage_ci_lower_strictly_above": 0.0},
    }
    if config != expected:
        raise ValueError("config differs from the prospectively bounded protocol")


def frozen_partitions(parent, seeds):
    old_ids = np.asarray(original_candidate_indices(), dtype=np.int64)
    new_ids = np.setdiff1d(np.arange(1820), old_ids)
    result = {"parent": parent.tolist(), "parent_array_sha256": array_sha256(parent),
              "candidate_schema_sha256": candidate_schema_sha256(),
              "original_candidate_ids": old_ids.tolist(), "children": {},
              "audits": {}, "parity_ids": {}}
    for seed in seeds:
        child = split_parent_experts(parent, seed=seed)
        result["children"][str(seed)] = child.tolist()
        result["audits"][str(seed)] = validate_refinement(parent, child)
        chosen = np.random.default_rng(seed).choice(new_ids, 16, replace=False)
        result["parity_ids"][str(seed)] = np.unique(
            np.concatenate((old_ids, [new_ids[0], new_ids[-1]], chosen))).tolist()
    return result


def verify_sources(output):
    registration = json.loads((output / "preregistration.json").read_text())
    for name, expected in registration["source_sha256"].items():
        if (sha_file(ROOT / name) != expected
                or sha_file(output / "registered_sources" / name) != expected):
            raise RuntimeError(f"registered execution source changed: {name}")
    verify_seal(registration["frozen_artifact_sha256"], output)


def register(config_path, output):
    config = json.loads(config_path.read_text())
    validate_config(config)
    if sha_file(ROOT / config["frozen_partition"]) != config["frozen_partition_sha256"]:
        raise ValueError("frozen C-29 parent changed")
    rows, audit, quarantine = derive_granularity_manifest(ROOT)
    if serialized_manifest_bytes(ROOT, rows) != (ROOT / config["manifest"]).read_bytes():
        raise ValueError("manifest is not the exact unopened allocation")
    if dict(Counter(row["split"] for row in rows)) != config["split_sizes"]:
        raise ValueError("prospective split counts changed")
    parent = np.asarray(json.loads((ROOT / config["frozen_partition"]).read_text()), dtype=np.int64)
    partitions = frozen_partitions(parent, config["seeds"])
    files = {config_path, Path(__file__), ROOT / config["frozen_partition"],
             ROOT / "EXPERT_GRANULARITY_PREREGISTRATION.md",
             ROOT / "experiments/expert_construction.py",
             ROOT / "experiments/expert_granularity_evaluation.py",
             ROOT / "scripts/run_expert_granularity.ps1",
             ROOT / "scripts/prepare_granularity_data.py",
             ROOT / "requirements-photonic.txt", ROOT / "Dockerfile.photonic"}
    files.update((ROOT / "src/depth_router").glob("expert_*.py"))
    files.update(ROOT / f"src/depth_router/{name}.py" for name in
                 ("granularity_data", "recognition_data", "output_gap", "output_gap_data"))
    files.update(path for path in (ROOT / "data/expert_granularity").iterdir() if path.is_file())
    files.update(ROOT / name for name in ("data/expert_recognition/evaluation_manifest.jsonl",
                 "data/expert_recognition/data_audit.json", "data/expert_audit/evaluation_manifest.jsonl"))
    files.update((ROOT / "data/expert_audit").glob("historical_*.json*"))
    hashes = {path.relative_to(ROOT).as_posix(): sha_file(path) for path in sorted(files)}
    output.mkdir(parents=True, exist_ok=False)
    for path in sorted(files):
        target = output / "registered_sources" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    write_json(output / "partitions.json", partitions)
    write_json(output / "verified_data_audit.json", audit)
    write_json(output / "quarantined_confirmation.json", quarantine)
    frozen = {name: sha_file(output / name) for name in
              ("partitions.json", "verified_data_audit.json", "quarantined_confirmation.json")}
    write_json(output / "preregistration.json", {
        "utc": datetime.now(timezone.utc).isoformat(), "config": config,
        "source_sha256": hashes, "frozen_artifact_sha256": frozen,
        "before_model_load_fit_and_fresh_outputs": True,
        "prior_negative_results_unchanged": True,
    })
    return config


def seal_files(output, names, filename):
    seal = {name: sha_file(output / name) for name in names}
    write_json(output / filename, seal)
    return seal


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
            raise AssertionError("wrong model architecture")
        original_weights = {name: tensor_sha256(value) for name, value in model.state_dict().items()}
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
            "original_weight_sha256": original_weights,
        })
        partitions = json.loads((output / "partitions.json").read_text())
        parent = np.asarray(partitions["parent"])
        labels = {seed: np.asarray(partitions["children"][str(seed)], dtype=np.int64)
                  for seed in config["seeds"]}
        combinations = np.asarray(candidate_combinations(), dtype=np.int64)
        all_ids = np.arange(len(combinations), dtype=np.int64)
        old_ids = np.asarray(original_candidate_indices(), dtype=np.int64)
        indices = {seed: torch.from_numpy(np.stack([
            partition_indices(labels[seed], mask) for mask in combinations]))
            for seed in config["seeds"]}
        control_order = torch.from_numpy(np.concatenate([
            np.flatnonzero(parent == index) for index in range(8)]))
        rows = [json.loads(line) for line in (ROOT / config["manifest"]).read_text().splitlines()]
        portfolio, fixed, paired = {}, {"original": None, "selected": {}, "full_oracle": {}}, {}
        metrics = {"original": {}, "selected": {str(s): {} for s in config["seeds"]},
                   "full_oracle": {str(s): {} for s in config["seeds"]}}
        bootstrap = {"bootstrap_repetitions": config["bootstrap_repetitions"],
                     "bootstrap_seed": config["bootstrap_seed"]}
        cal_seal, heldout_seal = None, None
        inherited_checks, parity_checks = {}, {}
        for split in ("calibration", *EVALUATION_SPLITS):
            budget.check()
            verify_sources(output)
            if cal_seal is not None:
                verify_seal(cal_seal, output)
            if split.startswith("heldout"):
                verify_seal(heldout_seal, output)
            groups, records, replay = collect_states(
                model, tokenizer, dense, [row for row in rows if row["split"] == split],
                config, output, log, budget)
            baseline = evaluate_candidates(
                model, dense, indices[17][old_ids], groups, records,
                output / f"matrix-{split}-original.json", budget, log,
                tile_size=config["candidate_tile_size"], control_order=control_order,
                candidate_ids=old_ids, partition=labels[17],
                partition_sha256=array_sha256(labels[17]), selected_experts=combinations[old_ids],
                split=split)
            if split == "development":
                fixed["original"] = select_development_fixed(baseline, old_ids)
            if split != "calibration":
                metrics["original"][split] = summarize_portfolio(
                    baseline, old_ids, fixed["original"], split=split, **bootstrap)
            selected_coverages = []
            paired[split] = {"per_seed": {}}
            inherited_checks[split] = {}
            for seed in config["seeds"]:
                if split == "calibration":
                    parity_checks[str(seed)] = validate_cached_candidates(
                        model, dense, indices[seed], replay, budget,
                        candidate_ids=partitions["parity_ids"][str(seed)],
                        control_order=control_order)
                    write_json(output / "full_model_cached_validation.json", parity_checks)
                matrix = evaluate_candidates(
                    model, dense, indices[seed], groups, records,
                    output / f"matrix-{split}-seed-{seed}.json", budget, log,
                    tile_size=config["candidate_tile_size"], control_order=control_order,
                    inherited_matrix=baseline, inherited_candidate_ids=old_ids,
                    candidate_ids=all_ids, partition=labels[seed],
                    partition_sha256=array_sha256(labels[seed]), selected_experts=combinations,
                    split=split)
                inherited_checks[split][str(seed)] = {
                    "exact_masks_and_all_raw_measurements": True,
                    "original_coverage": float(portfolio_coverage(baseline, old_ids).mean()),
                    "full_oracle_coverage": float(portfolio_coverage(matrix, all_ids).mean()),
                }
                write_json(output / "inherited_mask_validation.json", inherited_checks)
                if split == "calibration":
                    portfolio[str(seed)] = select_calibration_portfolio(matrix)
                    write_json(output / "portfolio_selection.json", portfolio)
                else:
                    ids = portfolio[str(seed)]["candidate_ids"]
                    if split == "development":
                        fixed["selected"][str(seed)] = select_development_fixed(matrix, ids)
                        fixed["full_oracle"][str(seed)] = select_development_fixed(matrix, all_ids)
                    metrics["selected"][str(seed)][split] = summarize_portfolio(
                        matrix, ids, fixed["selected"][str(seed)], split=split, **bootstrap)
                    metrics["full_oracle"][str(seed)][split] = summarize_portfolio(
                        matrix, all_ids, fixed["full_oracle"][str(seed)], split=split, **bootstrap)
                    paired[split]["per_seed"][str(seed)] = paired_portfolio_difference(
                        matrix, ids, baseline, old_ids, **bootstrap)
                    selected_coverages.append(portfolio_coverage(matrix, ids))
                    write_json(output / "metrics.json", metrics)
                log("seed_split_complete", split=split, seed=seed, elapsed_seconds=budget.elapsed())
                del matrix
            if split == "calibration":
                cal_seal = seal_files(output, ["partitions.json", "preregistration.json",
                                              "portfolio_selection.json"], "calibration_seal.json")
                log("portfolio_choices_sealed_before_development", seal=cal_seal)
            else:
                coverage = np.stack(selected_coverages)
                paired[split]["aggregate"] = paired_coverage_difference(
                    coverage, np.broadcast_to(portfolio_coverage(baseline, old_ids), coverage.shape),
                    baseline["context_ids"], **bootstrap)
                write_json(output / "paired.json", paired)
                if split == "development":
                    write_json(output / "fixed_selection.json", fixed)
                    heldout_seal = seal_files(output, [*cal_seal, "calibration_seal.json",
                                                       "fixed_selection.json"], "heldout_seal.json")
                    log("fixed_comparators_sealed_before_heldouts", seal=heldout_seal)
            del groups, records, replay, baseline
            budget.check()
        gate = cycle_gate(metrics["selected"], paired["heldout_ood"]["aggregate"])
        verify_sources(output)
        verify_seal(heldout_seal, output)
        final_weights = {name: tensor_sha256(value) for name, value in model.state_dict().items()}
        write_json(output / "final_weight_sha256.json", final_weights)
        if final_weights != original_weights:
            raise AssertionError("original model weights changed")
        budget.check()
        write_json(output / "promotion_decision.json", {
            **gate, "status": "completed", "promote": False,
            "capability_pass": gate["point_pass"], "original_weights_unchanged": True,
            "next_recognition_study_requires_separate_authorization": True,
            "runtime_training": "not_run_out_of_scope", "runtime_timing": "not_run_out_of_scope",
            "global_bandwidth_solver_promotion": False,
            "prior_negative_results_unchanged": True,
            "gpu_budget_ceiling_seconds": budget.seconds,
            "single_gpu_wall_elapsed_seconds": budget.elapsed(),
            "free_generation_task_quality": None,
            "cost_note": "cycle wall time is not inference latency or measured bandwidth",
        })
        log("cycle_complete", capability_pass=gate["point_pass"], promote=False,
            elapsed_seconds=budget.elapsed())
    except Exception as error:
        write_json(output / "incomplete.json", {
            "error_type": type(error).__name__, "error": str(error), "promote": False,
            "single_gpu_wall_elapsed_seconds": budget.elapsed(),
        })
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/expert_granularity.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    run(register(args.config.resolve(), output), output)


if __name__ == "__main__":
    main()
