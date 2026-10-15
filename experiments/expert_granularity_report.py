"""Render a completed or explicitly incomplete granularity study from artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("development", "heldout_matched", "heldout_ood")
SEEDS = (17, 29, 43)


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _gate(value):
    return "pass" if value is True else "fail" if value is False else "unestablished"


def _percent(value):
    return f"{100 * value:.3f}%"


def _interval(estimate, interval, *, unit="%"):
    return f"{100 * estimate:.3f}{unit} [{100 * interval['lower']:.3f}, {100 * interval['upper']:.3f}]"


def _table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)]


def _capability_rows(metrics):
    records = []
    for split in SPLITS:
        records.append(("Original 28", "fixed parent", split, metrics["original"][split]))
        for seed in SEEDS:
            records.append(("Selected 28", seed, split, metrics["selected"][str(seed)][split]))
    return [[name, seed, split, value["n_contexts"], value["n_states"],
             _interval(value["cheap_coverage"], value["confidence_intervals"]["cheap_coverage"]),
             _interval(value["fixed_cheap_coverage"], value["confidence_intervals"]["fixed_cheap_coverage"]),
             _interval(value["portfolio_advantage"], value["confidence_intervals"]["portfolio_advantage"], unit=" pp"),
             _gate(value["engineering_gate"]["point_pass"]),
             _gate(value["engineering_gate"]["statistically_resolved_pass"])]
            for name, seed, split, value in records]


def build_report(run_dir, *, repo=ROOT):
    """Read evidence only; do not fit portfolios, load a model, or run inference."""
    run, repo = Path(run_dir).resolve(), Path(repo).resolve()
    registration = _json(run / "preregistration.json")
    config = registration["config"]
    decision = _json(run / "promotion_decision.json") if (run / "promotion_decision.json").exists() else None
    incomplete = _json(run / "incomplete.json") if (run / "incomplete.json").exists() else None
    if decision is None and incomplete is None:
        raise ValueError("report requires a terminal completed or explicitly incomplete run")
    if decision is not None and decision.get("status") != "completed":
        raise ValueError("terminal granularity decision does not attest completion")
    metrics = _json(run / "metrics.json") if decision is not None else None
    paired = _json(run / "paired.json") if decision is not None else None
    try:
        relative = run.relative_to(repo).as_posix()
    except ValueError:
        relative = run.as_posix()
    lines = ["# Frozen-parent granularity: equal 28-choice portfolio audit", "",
             f"Run: `{relative}`. Preregistered UTC: `{registration['utc']}`.", "", "## Outcome", ""]
    if decision is None:
        lines += ["**Incomplete.** No positive or negative capability conclusion is established.",
                  f"Recorded error: `{incomplete['error_type']}` — {incomplete['error']}",
                  "Partial observations are retained; they are not a smaller substitute experiment.", ""]
    elif decision["capability_pass"]:
        lines += ["**The preregistered equal-budget capability gate passed.** Every selected 28-mask portfolio",
                  "passed the absolute gates on every evaluation split, and the seed-mean paired OOD coverage",
                  "gain had a positive lower confidence bound. This is a capability result, not causal routing",
                  "or measured savings. A separate recognition study requires new authorization.", ""]
    else:
        lines += ["**The preregistered equal-budget capability gate did not pass.** Preserve the per-seed and",
                  "paired results below; do not substitute the larger 1,820-choice diagnostic oracle for the",
                  "28-choice comparison. Router recognition and inference economics were not tested.", ""]
    if decision is not None:
        original_ood = metrics["original"]["heldout_ood"]["cheap_coverage"]
        selected_ood = [metrics["selected"][str(seed)]["heldout_ood"]["cheap_coverage"] for seed in SEEDS]
        gain = paired["heldout_ood"]["aggregate"]["net_coverage"]
        lines += [(f"On the same fresh OOD states, original-28 coverage was {_percent(original_ood)}; "
                   f"selected-28 coverage ranged from {_percent(min(selected_ood))} to {_percent(max(selected_ood))}."),
                  f"Seed-mean paired gain: {_interval(gain['estimate'], gain, unit=' pp')} (95% CI).", "",
                  "The earlier failed confirmation set is not the comparative baseline. These are paired",
                  "original-versus-refined measurements on this cycle's same fresh held-out states."]
        if original_ood >= 0.8:
            lines += ["The original portfolio itself also exceeds 80% OOD coverage on this fresh set."]
        lines += ["A positive paired gain supports finer-mask headroom; it does not establish granularity as the",
                  "sole cause of the earlier confirmation failure, or show that OOD generalization is solved.", ""]
    lines += ["The earlier C-29 confirmation failure remains unchanged. All 128 previously inspected confirmation",
              "contexts were quarantined; none was reused in this cycle. No partition winner was selected",
              "from held-out outcomes, no seed was discarded, and no original model weight was fitted.", "",
              "## Controlled question and scope", "",
              "Does finer neuron grouping improve held-out acceptable-state coverage while keeping the same",
              "active width and exactly the same number of deployed choices? The comparisons are:", ""]
    lines += _table(["Arm", "Grouping", "Available choices", "Role"], [
        ["Original 28", "6 of 8 frozen C-29 parents", 28, "Equal-budget reference"],
        ["Selected 28", "12 of 16 balanced child experts", 28, "Primary calibration-fitted portfolio"],
        ["Full 1,820 oracle", "12 of 16 balanced child experts", 1820, "Diagnostic upper bound only"],
    ])
    lines += ["", f"Model: `{config['model']}` at `{config['revision']}`; {config['dtype']}, {config['attention']}.",
              "All weights and the final-layer C-29 parent partition remain frozen. Each 608-neuron parent",
              "is randomly split into two 304-neuron children using NumPy PCG64 seeds 17, 29 and 43.",
              "Every candidate executes exactly 3,648 of 4,864 neurons (75%). All 28 original masks are exact",
              "members of each 1,820-mask universe. Real packed BF16 gate/up/down projections, residual",
              "addition, final normalization and full-vocabulary head are used; no masked FP32 shortcut.", ""]
    if (run / "environment.json").exists():
        environment = _json(run / "environment.json")
        lines += [(f"Execution environment: {environment['gpu']}; PyTorch {environment['torch']}; "
                   f"CUDA {environment['cuda']}; Transformers {environment['transformers']}."), ""]
    lines += ["## Data, fitting and uncertainty", "",
              "The 1,600-context manifest is a byte-preserving derivative of previously unopened recognition",
              "partitions: 768 calibration, 128 development, 512 matched held-out, 192 OOD held-out.",
              "IDs, prompt strings, JSONL lines, original order, task assignments and 12.5% enterprise-keyword",
              "quotas are unchanged. The full prior 1,728-row and filtered 1,600-row exclusion audits were",
              "rerun against historical data and the quarantined confirmation contexts. Source responses are not used.", "",
              "Freshness means the specifically pinned prior run produced no model outputs on the retained",
              "contexts. It does not prove global source novelty or semantic independence. Dolly's held-out",
              "creative-writing/summarization categories and enterprise keyword screen are not a production",
              "enterprise benchmark; source-article IDs are unavailable and OOD enterprise rows are creative-writing-heavy.", "",
              "Calibration greedily selects exactly 28 distinct masks per seed by newly covered states, then",
              "lower mean full-vocabulary KL, then canonical mask ID. This is explicit discrete fitting, not",
              "zero fitting. Portfolios are sealed before development. Development independently freezes the",
              "best fixed member of original 28, selected 28 and diagnostic 1,820 portfolios before either",
              "held-out split. The old study's fixed mask is not this study's primary fixed comparator.", "",
              f"Dense-prefix continuation positions: `{config['continuation_positions']}`; batch {config['batch_size']};",
              f"max prompt length {config['max_prompt_tokens']} tokens; stop recording after individual EOS.",
              "All captured valid states are retained. The 95% intervals resample whole contexts 2,000 times",
              "with seed 17, keeping their token positions together. Seed-mean contrasts retain all three",
              "seeds per draw and do not quantify uncertainty over future refinement seeds. No multiplicity-adjusted",
              "population claim is made.", ""]
    captures = []
    for split in ("calibration", *SPLITS):
        path = run / f"capture-{split}.json"
        if path.exists():
            value = _json(path)
            captures.append([split, value["contexts_requested"], value["contexts_evaluated"], value["states"],
                             len(value["zero_state_context_ids"]), value["prompt_truncations"]])
    if captures:
        lines += _table(["Captured split", "Requested contexts", "Evaluated contexts", "States", "Zero-state contexts", "Truncated prompts"], captures)
        lines += [""]
    if decision is not None:
        portfolios, fixed = _json(run / "portfolio_selection.json"), _json(run / "fixed_selection.json")
        lines += ["## Primary equal-budget result", "",
                  "Local acceptability requires exact dense greedy-token agreement AND KL(dense||candidate)<0.01.",
                  "Each selected seed must reach at least 80% coverage and at least 5 percentage points of ideal",
                  "active-width advantage over its development-frozen fixed-member-plus-dense oracle on all",
                  "three evaluation splits. Point engineering gates and statistically resolved passes are separate.",
                  "A failed resolved-pass check means that a pass is not established; it does not by itself",
                  "establish population failure. An interval spanning its threshold remains unresolved.", ""]
        lines += _table(["Portfolio", "Seed", "Split", "Contexts", "States", "Coverage [95% CI]",
                         "Fixed coverage [95% CI]", "Ideal width advantage [95% CI]", "Point", "Resolved pass"],
                        _capability_rows(metrics))
        lines += ["", "Ideal active width = 1 − 0.25 × coverage; advantage = 0.25 × (coverage − fixed coverage).",
                  "These are arithmetic proxies excluding routing, dispatch, duplicate storage and every hardware overhead.", "",
                  "### Paired selected-28 minus original-28", "",
                  "Rescue means a state newly has an acceptable choice; loss means a previously covered state",
                  "loses all acceptable choices in the selected portfolio. Net gain = rescue − loss.", ""]
        paired_rows = []
        for split in SPLITS:
            for seed in (*map(str, SEEDS), "seed mean"):
                value = paired[split]["aggregate"] if seed == "seed mean" else paired[split]["per_seed"][seed]
                paired_rows.append([split, seed,
                    _interval(value["rescue"]["estimate"], value["rescue"], unit=" pp"),
                    _interval(value["loss"]["estimate"], value["loss"], unit=" pp"),
                    _interval(value["net_coverage"]["estimate"], value["net_coverage"], unit=" pp"),
                    _interval(value["ideal_width_improvement"]["estimate"], value["ideal_width_improvement"], unit=" pp")])
        lines += _table(["Split", "Seed", "Rescue [CI]", "Loss [CI]", "Net coverage gain [CI]", "Ideal width improvement [CI]"], paired_rows)
        lines += ["", "The added paired improvement rule requires the seed-mean OOD net coverage lower bound to be",
                  "strictly greater than zero. It cannot relax the per-seed absolute gates.", ""]
        lines += _table(["Gate", "Result"], [
            ["All three selected portfolios, all splits: point gates", _gate(decision["all_seed_all_split_point_pass"])],
            ["All three selected portfolios, all splits: resolved lower bounds", _gate(decision["all_seed_all_split_statistically_resolved_pass"])],
            ["Seed-mean paired OOD gain: positive lower bound", _gate(decision["aggregate_ood_resolved_positive_gain"])],
            ["Registered capability decision", _gate(decision["capability_pass"])],
        ])
        lines += ["", "### Larger-search diagnostic, not the equal-budget result", ""]
        full_rows = []
        for seed in SEEDS:
            for split in SPLITS:
                value = metrics["full_oracle"][str(seed)][split]
                full_rows.append([seed, split,
                    _interval(value["cheap_coverage"], value["confidence_intervals"]["cheap_coverage"]),
                    _percent(value["fixed_cheap_coverage"]), value["candidate_count"]])
        lines += _table(["Refinement seed", "Split", "Full-oracle coverage [CI]", "Its fixed coverage", "Choices"], full_rows)
        lines += ["", "The full 1,820-mask oracle is teacher-assisted and has about 65 times as many choices as either",
                  "primary portfolio. Its inclusion of all original masks makes nondecreasing full-oracle coverage",
                  "a numerical nesting control, not evidence of an equally cheap or causal deployed selector.", "",
                  "### Frozen selections", ""]
        selection_rows = [[seed, portfolios[str(seed)]["size"], _percent(portfolios[str(seed)]["calibration_coverage"]),
                           fixed["selected"][str(seed)]["candidate_id"],
                           fixed["full_oracle"][str(seed)]["candidate_id"]] for seed in SEEDS]
        lines += _table(["Seed", "Selected choices", "Calibration coverage", "Selected-28 fixed ID", "Full-oracle fixed ID"], selection_rows)
        lines += ["", f"Original-28 development-frozen fixed canonical ID: `{fixed['original']['candidate_id']}`.",
                  "All 28 selection steps, marginal gains, KL tie-breaks, candidate IDs and seals are retained in",
                  "`portfolio_selection.json` and `fixed_selection.json`. Per-mask arrays are preserved in lossless chunks.", ""]
    lines += ["## Numerical integrity", ""]
    verification_path = run / "postrun_integrity_verification.json"
    if verification_path.exists():
        verification = _json(verification_path)
        if (verification.get("pass") is not True
                or verification.get("scope") != "complete_granularity_integrity"):
            raise ValueError("saved post-run verification does not establish complete integrity")
        lines += [(f"Independent CPU-only post-run verification passed: "
                   f"{verification['registered_sources_checked']} registered sources, "
                   f"{verification['matrix_manifests_checked']} matrix archives and "
                   f"{verification['candidate_cells_checked']:,} candidate-state cells checked."),
                  "Saved evidence: `postrun_integrity_verification.json`. Selections, metrics, paired intervals,",
                  "gates, inherited equality, data exclusions and quarantine were independently recomputed.", ""]
    parity_path = run / "full_model_cached_validation.json"
    if parity_path.exists():
        parity = _json(parity_path)
        checks = [row for rows in parity.values() for row in rows]
        lines += [(f"Cached evaluator versus actual full-model substitution: {len(checks)} retained layout checks, "
                   f"{config['batch_size']} calibration prefixes per layout. Exact logits in every saved check: "
                   f"`{all(row['exact_logits'] for row in checks)}`."),
                  "The prospective sample includes inherited masks, first/last new masks, seeded random new",
                  "masks and both full-width controls. This is sampled full-model validation, not replay of every",
                  "new mask in the whole model.", ""]
    if decision is not None:
        inherited = _json(run / "inherited_mask_validation.json")
        checks = [value for per_seed in inherited.values() for value in per_seed.values()]
        lines += [f"Inherited original-mask raw equality was checked for {len(checks)} seed/split combinations.",
                  f"Every saved equality assertion passed: `{all(value['exact_masks_and_all_raw_measurements'] for value in checks)}`.",
                  "The offline verifier independently compares exact neuron indices, all candidate scalar/token",
                  "fields and both controls for those inherited columns, then checks full-oracle coverage nesting.", ""]
        control_rows = []
        for split in ("calibration", *SPLITS):
            descriptor = _json(run / f"matrix-{split}-original.json")
            with np.load(run / descriptor["metadata"]["file"], allow_pickle=False) as matrix:
                control_rows.append([split, _percent(float(np.mean(matrix["control_token"] == matrix["dense_token"]))),
                    f"{float(np.mean(matrix['control_kl'], dtype=np.float64)):.8g}",
                    _percent(float(np.mean(matrix["permuted_control_token"] == matrix["dense_token"]))),
                    f"{float(np.mean(matrix['permuted_control_kl'], dtype=np.float64)):.8g}"])
        lines += _table(["Split", "Original-order agreement", "Original-order KL", "Parent-permuted agreement", "Parent-permuted KL"], control_rows)
        lines += ["", "Original-order full width must be exactly dense. Parent-permuted reduction is measured",
                  "separately and is not assumed bitwise equal to the original order. Perfect greedy agreement",
                  "can coexist with nonzero KL. Original tensor hashes before and after execution are retained.", ""]
    lines += ["## Cost boundaries and next-step scope", "",
              "This cycle ends at capability. No causal router was trained and no inference timing was performed.",
              "Latency, physical DRAM traffic, throughput, free-generation/task quality and net savings are",
              "**unmeasured**, not zero. Global bandwidth-solver promotion remains false even if capability passes.",
              "At most 28 packed candidate banks are resident during offline evaluation; the larger search",
              "cost is research cost, not an inference-speed measurement.", "",
              "Removing 25% of one of 24 MLPs is only about 1.04% of aggregate MLP arithmetic before overhead.",
              "Nothing here establishes that a selector can recognize the good subsets cheaply enough to",
              "yield real savings. A capability pass would justify proposing a separately authorized",
              "recognition experiment, not automatically running one.", ""]
    elapsed = decision if decision is not None else incomplete
    lines += [(f"Recorded single-GPU wall elapsed: {elapsed['single_gpu_wall_elapsed_seconds']:.3f} seconds; "
              f"ceiling {config['gpu_wall_budget_seconds']} seconds. This includes research execution/serialization,"),
              "not a latency benchmark. The run uses one bounded budget; it is not silently replaced by a smaller audit.", "",
              "## Reproduction and evidence", "",
              "Registered source/config/data snapshots, three refined partitions, prospective parity sample IDs,",
              "state-prefix records, every measured candidate value, selections, seals and failures are retained.",
              "Candidate matrices use hashed metadata/layout archives and lossless chunks of at most 256 columns.",
              "Local activation caches may be ignored by Git; capture metadata records their hashes and reconstruction provenance.", "",
              "Repository attributes preserve the sealed historical source line endings on fresh cross-platform checkouts.", "",
              "```powershell", f"python scripts/verify_expert_granularity.py --run-dir {relative} --validate-data",
              f"python experiments/expert_granularity_report.py --run-dir {relative}",
              ".\\scripts\\run_expert_granularity.ps1  # new output directory; never overwrite the recorded run",
              "```", "",
              "The verifier uses no language model or GPU. It recomputes partition splits, all 28-mask selections,",
              "development comparators, exact inherited equality, metrics, paired intervals and gates from raw",
              "artifacts. It checks saved before/after tensor hashes and event/seal consistency, not new hardware",
              "measurements or cryptographic proof of real-world chronology.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--report", type=Path, default=ROOT / "EXPERT_GRANULARITY_REPORT.md")
    args = parser.parse_args()
    text = build_report(args.run_dir, repo=args.repo)
    args.report.write_bytes(text.encode())
    print(str(args.report.resolve()))


if __name__ == "__main__":
    main()
