"""Render the completed controlled run without fitting or selecting anything.

All numerical evidence comes from the supplied run directory. Missing outcome
files are an error; partial runs are never rendered as completed experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SPLITS = ("development", "heldout_matched", "heldout_ood")
SPLIT_LABELS = ("Development", "Matched", "OOD")


def read_json(run: Path, name: str) -> Any:
    path = run / name
    if not path.is_file():
        raise ValueError(f"Incomplete run: required artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def interval_pp(row: dict[str, Any]) -> str:
    lower, upper = row["ci95"]
    return (
        f"{100 * row['gap_minus_hidden_agreement']:+.3f} [{100 * lower:+.3f}, {100 * upper:+.3f}]"
    )


def table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    def line(values: list[Any]) -> str:
        return "| " + " | ".join(str(value).replace("|", "\\|") for value in values) + " |"

    return [line(headers), line(["---"] * len(headers)), *(line(row) for row in rows)]


def arm_label(name: str) -> str:
    if name == "dense":
        return "Full Qwen"
    if name == "full_width_control":
        return "Full-width packed control"
    if name == "sparse75":
        return "75% retained, uncorrected"
    objective, rank, seed = name.split("_")
    return f"{'Hidden MSE' if objective == 'hidden' else 'Output gap'}, {rank}, {seed}"


def render_report(run: Path) -> str:
    prereg = read_json(run, "preregistration.json")
    config = prereg["config"]
    decision = read_json(run, "decision.json")
    development = read_json(run, "development_quality.json")
    heldout = read_json(run, "heldout_quality.json")
    comparison = read_json(run, "paired_objective_comparison.json")
    freeze = read_json(run, "development_selection.json")
    training = read_json(run, "training_audit.json")
    numerical = read_json(run, "numerical_checks.json")
    bound = read_json(run, "bound_audit.json")
    timing = read_json(run, "timing.json")
    environment = read_json(run, "environment.json")
    data = read_json(run, "data_audit.json")
    mask = read_json(run, "mask.json")
    tokens = read_json(run, "tokenization_audit.json")

    names = ["dense", "full_width_control", "sparse75"] + [
        f"{objective}_r{rank}_s{seed}"
        for rank in config["ranks"]
        for seed in config["seeds"]
        for objective in ("hidden", "gap")
    ]
    records = {(row["method"], row["split"]): row for row in development + heldout}
    expected = {(name, split) for name in names for split in SPLITS}
    if set(records) != expected or len(records) != len(development + heldout):
        raise ValueError("The completed quality artifacts must contain every arm and split once")
    comparisons = {(r["rank"], r["seed"], r["split"]): r for r in comparison}
    expected_comparisons = {
        (rank, seed, split)
        for rank in config["ranks"]
        for seed in [*config["seeds"], "paired_seed_mean"]
        for split in SPLITS
    }
    if set(comparisons) != expected_comparisons or len(comparisons) != len(comparison):
        raise ValueError("The paired comparisons must contain every rank, seed and split once")
    if sha_file(run / "evaluation_manifest.jsonl") != freeze["manifest_sha256"]:
        raise ValueError("Manifest no longer matches the frozen selection")
    if sha_file(run / "corrections.pt") != freeze["checkpoint_sha256"]:
        raise ValueError("Correction checkpoint no longer matches the frozen selection")
    if sha_file(run / "development_selection.json") != decision["selection_sha256"]:
        raise ValueError("Selection file no longer matches the completed decision")

    selected = decision["selected_arm"]
    primary_rank = decision["objective_improvement"]["primary_rank"]
    objective_pass = decision["objective_improvement"]["established"]
    quality_pass = decision["absolute_quality_pass"]
    skipped = timing.get("status") == "skipped_by_preregistered_quality_gate"
    speed_pass = bool(timing.get("speed_claims", {}).get(selected, False))
    all_pairs_match = all(
        row["initial_state_equal"] and row["minibatch_order_equal"] for row in training.values()
    )
    primary = [comparisons[(primary_rank, "paired_seed_mean", s)] for s in SPLITS[1:]]
    if objective_pass != all(row["established"] for row in primary):
        raise ValueError("Decision disagrees with the frozen primary paired comparison")
    actual_quality = all(records[(selected, s)]["passes_quality_gate"] for s in SPLITS)
    if quality_pass != actual_quality:
        raise ValueError("Decision disagrees with the selected arm's quality gates")
    development_corrections = [
        row for row in development if row["method"].startswith(("hidden_", "gap_"))
    ]
    development_passes = sum(row["passes_quality_gate"] for row in development_corrections)

    if not objective_pass:
        opening = (
            "Output-gap training did not establish better held-out agreement than hidden-state "
            "MSE under the preregistered matched-budget comparison."
        )
    elif not quality_pass:
        opening = (
            "Output-gap training improved held-out agreement under the preregistered comparison, "
            "but the selected correction failed the absolute quality gate."
        )
    elif not speed_pass:
        opening = (
            "Output-gap training passed the held-out agreement and absolute quality gates; "
            "a deployable saving has not been established."
        )
    else:
        opening = (
            "Output-gap training passed the held-out agreement and absolute quality gates, "
            "and the frozen arm's timing audit supports cached-decode latency savings."
        )
    lines = [
        "# DepthRouter output-gap correction: controlled experiment",
        "",
        opening,
        (
            f"At the frozen primary rank {primary_rank}, the mean paired gap-minus-hidden "
            f"differences were {interval_pp(primary[0])} percentage points on matched prompts "
            f"and {interval_pp(primary[1])} on OOD prompts (95% intervals)."
        ),
        (
            f"{development_passes} of {len(development_corrections)} correction fits passed "
            "the development absolute agreement-and-KL gate."
        ),
        f"Run `{run.name}` ended with `{decision['status']}`. "
        + (
            "Timing and free-running generation were skipped by the preregistered gate; "
            "there is no measured speedup or physical-traffic result from this run."
            if skipped
            else "Timing and generation results are reported separately below."
        ),
        "",
        (
            "The notebook's synthetic examples motivated changing the training objective. "
            "They are not Qwen observations and contribute no evidence to this result."
        ),
        "",
        "## Controlled comparison",
        "",
        (
            f"The model is `{config['model']}` at revision `{config['revision']}`. "
            f"Only final MLP layer {config['target_layer']} changed. The fixed calibration-selected "
            f"mask retains {mask['retained']:,}/{mask['total']:,} neurons "
            f"({100 * mask['retained'] / mask['total']:.0f}% retained, 25% omitted), "
            "physically packed into gate/up/down weights. Every sparse arm uses those same weights. "
            "The original full model and a full-width packed control are both included."
        ),
        "",
        (
            "Both correction objectives use `Linear(896, rank, bias=False) -> tanh -> "
            "Linear(rank, 896, bias=True)` with FP32 parameters and the same final-MLP input. "
            "The correction is added to the actual sparse output in FP32 and cast back to BF16; "
            "the model's ordinary residual addition, final RMSNorm and full vocabulary head then run. "
            "The deployed correction has no teacher, candidate IDs or stored labels."
        ),
        "",
        (
            "The target is the actual dense BF16 MLP output minus the actual packed sparse BF16 "
            "MLP output, measured by separate whole-model forwards. Hidden training minimizes "
            "hidden residual MSE. Gap training minimizes the seven contrasts between the cheap "
            "top-eight token rows and the cheap argmax row, using `A = E * gamma`. Each objective "
            "is divided by its calibration target mean square. Candidate search and head-row "
            "gathering are training-only operations for this architecture."
        ),
        "",
        (
            f"Ranks: {config['ranks']}; seeds: {config['seeds']}; "
            f"{config['correction_steps']} AdamW steps per fit, minibatch "
            f"{config['correction_batch_size']}, learning rate "
            f"{config['correction_learning_rate']}, weight decay "
            f"{config['correction_weight_decay']}. Rank 8 has 15,232 parameters and rank 16 "
            "has 29,568. Each seed/rank pair has identical initialization and minibatch order: "
            f"{'verified for all pairs' if all_pairs_match else 'FAILED audit'}. "
            "Only the loss objective changes within each pair."
        ),
        "",
        "## Agreement and KL for every arm",
        "",
        (
            "Each cell shows full-vocabulary greedy agreement / mean KL(full || candidate), "
            "using actual normalized model logits. Development has 256 examples; each held-out "
            "split has 512. These are first-token comparisons after each prompt."
        ),
        "",
    ]
    quality_rows = []
    for name in names:
        label = arm_label(name) + (" (selected)" if name == selected else "")
        quality_rows.append(
            [
                label,
                *[
                    f"{percent(records[(name, split)]['top1_agreement'])} / "
                    f"{records[(name, split)]['mean_kl']:.6f}"
                    for split in SPLITS
                ],
            ]
        )
    lines.extend(table(["Arm", *SPLIT_LABELS], quality_rows))
    lines.extend(
        [
            "",
            f"Development selected `{selected}`, fixing primary rank {primary_rank} before "
            "either held-out model evaluation. Selection rule: " + config["selection_rule"] + ". "
            "The primary objective test retains all three seeds at that fixed rank; a favorable "
            "held-out seed or alternate rank cannot replace it.",
            "",
            "## Paired objective differences",
            "",
            (
                "Values are output-gap minus hidden-MSE agreement in percentage points, followed "
                f"by 95% paired bootstrap intervals ({config['bootstrap_draws']:,} draws). The mean "
                "row averages the three seed differences within each prompt before resampling prompts. "
                "These intervals are conditional on the fixed template suite; repeated numeric/content "
                "variants are not independent natural-language template samples. No generalization "
                "to a natural-language population is claimed."
            ),
            "",
        ]
    )
    difference_rows = []
    for rank in config["ranks"]:
        for seed in [*config["seeds"], "paired_seed_mean"]:
            label = "paired seed mean" if seed == "paired_seed_mean" else str(seed)
            difference_rows.append(
                [rank, label, *[interval_pp(comparisons[(rank, seed, split)]) for split in SPLITS]]
            )
    lines.extend(table(["Rank", "Seed", *SPLIT_LABELS], difference_rows))
    lines.extend(
        [
            "",
            (
                "The frozen primary test requires positive direction for every seed "
                "and a paired-mean interval strictly above zero in both held-out splits."
            ),
            "",
        ]
    )
    lines.extend(
        table(
            ["Primary split", "All seeds positive", "Mean CI excludes zero above", "Established"],
            [
                [
                    row["split"],
                    row["direction_positive_all_seeds"],
                    row["ci95"][0] > 0,
                    row["established"],
                ]
                for row in primary
            ],
        )
    )
    lines.extend(
        [
            "",
            (
                f"The absolute gate is agreement >= {percent(config['quality_gate_top1'])} and "
                f"mean KL < {config['quality_gate_kl']} on development and both held-out splits. "
                f"Selected-arm gate: {'PASS' if quality_pass else 'FAIL'}."
            ),
            "",
        ]
    )
    lines.extend(
        table(
            ["Selected-arm split", "Agreement gate", "KL gate", "Combined"],
            [
                [
                    split,
                    records[(selected, split)]["top1_agreement"] >= config["quality_gate_top1"],
                    records[(selected, split)]["mean_kl"] < config["quality_gate_kl"],
                    records[(selected, split)]["passes_quality_gate"],
                ]
                for split in SPLITS
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Failure diagnostics and numerical checks",
            "",
            (
                "The first margin group contains all examples at or below the empirical first quartile; "
                "ties can make its size differ from exactly one quarter. Excluded winners are full-model "
                "winners absent from the uncorrected sparse top-eight. Held-out diagnostics are "
                "measured after freezing; teacher winners are never added to training candidates."
            ),
            "",
        ]
    )
    diagnostic_names = ["sparse75", selected.replace("gap_", "hidden_", 1), selected]
    diagnostics = []
    for split in SPLITS:
        for name in diagnostic_names:
            row = records[(name, split)]
            low = row["margin_quartiles"][0]
            diagnostics.append(
                [
                    split,
                    arm_label(name),
                    f"{percent(low['top1_agreement'])} ({low['n']})",
                    f"{row['excluded_winner_failures']}/{row['excluded_winners']}",
                    f"{row['hidden_residual_mse']:.6g}",
                    f"{row['output_gap_mse']:.6g}",
                ]
            )
    lines.extend(
        table(
            [
                "Split",
                "Arm",
                "Low-margin agreement (n)",
                "Excluded-winner errors / cases",
                "Hidden MSE",
                "Gap MSE",
            ],
            diagnostics,
        )
    )
    lines.extend(
        [
            "",
            (
                f"All {len(records)} arm/split summaries in `development_quality.json` and "
                "`heldout_quality.json` retain hidden and gap MSE, all margin-quartile counts "
                "and agreement rates, excluded-winner failures, and top-logit tie counts. "
                "The compact diagnostic table above shows the uncorrected arm and the "
                "development-selected objective pair; the raw summaries retain every seed."
            ),
            "",
            (
                f"The full-width packed control had {numerical['full_width_control_mismatches']} "
                f"greedy mismatches and maximum absolute logit error "
                f"{numerical['full_width_control_max_logit_error']:.6g}. "
                f"Checkpoint roundtrip exact: {numerical['checkpoint_roundtrip_exact']}; "
                f"checkpoint unchanged after held-out evaluation: "
                f"{numerical['checkpoint_unchanged_after_heldout']}. "
                f"Calibration contrast-identity maximum absolute error: "
                f"{numerical['gap_identity_max_abs_error']:.6g}. "
                "Full-precision algebra is not asserted to be an exact BF16-kernel identity."
            ),
            "",
            "## Bound and cost audit",
            "",
            (
                "The weight-only bound uses `R = beta * ||x||² + ||correction(x)||` and "
                "`epsilon = max_i ||E_i * gamma|| * R`. Including the correction norm is necessary "
                "to bound the correction error. If `R >= ||u_sparse + correction||`, even an upper "
                "bound on the score spread cannot pass the uniform `gap > 2 * epsilon` test. "
                f"Here beta is {bound['beta_omitted']:.6g}."
            ),
            "",
        ]
    )
    lines.extend(
        table(
            [
                "Correction",
                "Calibration cases",
                "Certain nonacceptance",
                "Certified",
                "Minimum R / approximate-state norm",
            ],
            [
                [
                    arm_label(row["method"]),
                    row["calibration_examples"],
                    row["certain_nonacceptance_count"],
                    row["certified_examples"]
                    if row["certified_examples"] is not None
                    else "unknown",
                    f"{row['min_radius_over_approximate_state_norm']:.1f}",
                ]
                for row in bound["rows"]
            ],
        )
    )
    lines.extend(
        [
            "",
            (
                "This is a real-arithmetic bound-looseness audit evaluated on BF16 captures. "
                "It does not establish a deterministic BF16 certificate. No deployable certificate, "
                "certified fallback policy, or false-certificate rate is claimed."
            ),
            "",
        ]
    )
    if skipped:
        cost_text = (
            "Unmeasured: timing and generation skipped by the registered quality/objective gate."
        )
        lines.append(
            "The registered prerequisites failed, so correction-inclusive MLP latency, cached-decode "
            "latency, free-running task quality and physical DRAM traffic were not measured. "
            "There are no raw timing samples. The retained-neuron fraction is not a speedup "
            "or physical-byte measurement. Historical measurements are not substituted."
        )
    else:
        cost_text = (
            "Measured cached-decode savings supported."
            if speed_pass
            else "Timing ran; correction-inclusive savings were not established."
        )
        lines.extend(
            table(
                [
                    "Scope",
                    "Batch",
                    "Method",
                    "Mean wall ms",
                    "Strongest dense",
                    "Paired speed ratio [95% CI]",
                    "Savings supported",
                ],
                [
                    [
                        row["scope"],
                        row["batch_size"],
                        row["method"],
                        f"{row['mean_wall_ms']:.6f}",
                        row["strongest_dense_method"],
                        json.dumps(row["versus_strongest_dense"]),
                        row["timing_ci_supports_savings"],
                    ]
                    for row in timing["rows"]
                ],
            )
        )
        lines.extend(
            [
                "",
                (
                    "Timing compares eager and CUDA-graph execution, including the complete "
                    "correction and casts. Cached decode overwrites one fixed cache position; it is "
                    "not whole-sequence throughput. Physical DRAM traffic remains unmeasured."
                ),
                "",
                "Free-running generation summaries: `"
                + json.dumps(timing.get("free_running_generation", {}), sort_keys=True)
                + "`.",
            ]
        )
    lines.extend(
        [
            "",
            "## Provenance and reproduction",
            "",
            (
                f"Artifacts: `artifacts/output_gap_completion/{run.name}/`. No result rows or logs "
                "were discarded by this report; all fitted ranks/seeds and every saved evaluation "
                "split are shown. The notebook is used only as the mathematical rationale."
            ),
            "",
            (
                "The manifest has 256 calibration, 256 development, 512 matched and 512 OOD prompts. "
                "Template families are disjoint between splits; exact, number-normalized and lexical "
                f"Jaccard >= {data['validation']['near_duplicate_threshold']} overlaps with other "
                "splits or registered historical prompts are rejected. Numeric/content variants "
                "within a family are allowed. The matched split shares task classes, not templates; "
                "the OOD split changes task classes. This is a controlled synthetic prompt suite, "
                "not natural-language benchmark evidence."
            ),
            "",
            (
                f"Unique tokenized prompts: {tokens['unique_token_sequences']}; "
                f"truncated prompts: {tokens['truncated']}. Training and policy choice preceded "
                "held-out model access; `research_log.jsonl` records the frozen checkpoint/selection "
                "hash before each held-out opening."
            ),
            "",
        ]
    )
    lines.extend(
        table(
            ["Artifact", "SHA-256"],
            [
                ["evaluation_manifest.jsonl", freeze["manifest_sha256"]],
                ["corrections.pt", freeze["checkpoint_sha256"]],
                ["development_selection.json", decision["selection_sha256"]],
                ["mask indices", mask["sha256"]],
            ],
        )
    )
    lines.extend(["", "Historical exclusion sources:", ""])
    lines.extend(
        table(
            ["Source", "Rows", "SHA-256"],
            [[row["path"], row["rows"], row["sha256"]] for row in data["historical_sources"]],
        )
    )
    lines.extend(["", "Frozen executed source fingerprints:", ""])
    lines.extend(
        table(
            ["Source", "SHA-256"],
            [[name, digest] for name, digest in sorted(prereg["source_sha256"].items())],
        )
    )
    change_path = run / "source_change_audit.json"
    if change_path.exists():
        changes = read_json(run, "source_change_audit.json")
        lines.extend(
            [
                "",
                (
                    "Exact registered sources are preserved in `executed_sources/`; the table "
                    "above records their original hashes. The snapshots preserve exact original "
                    "bytes, including Windows CRLF, with Git text conversion disabled. "
                    "A post-run source change is recorded "
                    f"in `source_change_audit.json`: {changes['post_run_change']}. The original "
                    f"timing source is `{changes['original_timing_source_snapshot']}`. Timing "
                    f"executed in this run: {changes['timing_executed_in_canonical_run']}; "
                    f"training/evaluation affected: {changes['affects_training_or_evaluation']}. "
                    "The preregistration and results were retained unchanged."
                ),
            ]
        )
    lines.extend(
        [
            "",
            (
                f"Execution: {environment['gpu']} ({environment['gpu_memory'] / 2**30:.2f} GiB); "
                f"Python {environment['python']}; PyTorch {environment['torch']}; CUDA "
                f"{environment['cuda']}; Transformers {environment['transformers']}. Model dtype "
                f"`{environment['model_dtype']}`; correction dtype `{environment['correction_dtype']}`; "
                f"TF32 allowed: {environment['tf32_allowed']}. Platform: `{environment['platform']}`. "
                f"NVIDIA driver snapshot: `{environment['nvidia_smi']}`. "
                f"Installed model implementation SHA-256: `{environment['model_implementation_sha256']}`."
            ),
            "",
            (
                "From the repository root in Windows PowerShell, reproduce into a new run directory "
                "with this exact command (the pinned model must already exist in the configured local cache):"
            ),
            "",
            "```powershell",
            (
                "powershell -ExecutionPolicy Bypass -File .\\scripts\\run_output_gap_completion.ps1 "
                f"-RunId {run.name}-reproduction"
            ),
            "```",
            "",
            (
                "The launcher refuses to reuse an existing RunId. The raw CSVs preserve each prompt's "
                "reference/candidate token, KL, ties, margins and error metrics. Training hashes/losses, "
                "full CIs, bound audit, source fingerprints and environment are retained in the JSON files."
            ),
            "",
        ]
    )
    passed = ["paired initialization/minibatches"] if all_pairs_match else []
    if numerical["full_width_control_max_logit_error"] == 0:
        passed.append("exact full-width packed control")
    if numerical["checkpoint_roundtrip_exact"] and numerical["checkpoint_unchanged_after_heldout"]:
        passed.append("checkpoint integrity")
    if objective_pass:
        passed.append("primary objective comparison")
    if quality_pass:
        passed.append("absolute quality gate")
    failed = []
    if not objective_pass:
        failed.append("preregistered held-out objective-improvement test")
    if not quality_pass:
        failed.append("selected-arm absolute quality gate")
    if all(row["certified_examples"] == 0 for row in bound["rows"]):
        failed.append("useful acceptance from this analytic norm bound")
    lines.extend(
        table(
            ["Finding", "Outcome"],
            [
                [
                    "Tested",
                    "Changing only the correction training objective at matched rank, mask and budget.",
                ],
                ["Passed", "; ".join(passed) or "None established."],
                ["Failed", "; ".join(failed) or "No registered quality/objective failure."],
                ["Objective improvement", "Established." if objective_pass else "Not established."],
                ["Cost", cost_text + " Physical DRAM savings remain unknown."],
                [
                    "Next experiment",
                    "None prescribed by this bounded result; no follow-up experiment was run.",
                ],
            ],
        )
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    try:
        report = render_report(run)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"Cannot render completed report: {exc}\n")
    destination = run / "OUTPUT_GAP_COMPLETION_REPORT.md"
    destination.write_text(report, encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
