"""Build the fixed-partition recognition report from completed recorded evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _percent(value):
    return "unknown" if value is None else f"{100 * value:.3f}%"


def _number(value):
    return "unknown" if value is None else f"{value:.6g}"


def _interval(estimate, interval, *, percent=True, unit=None):
    scale = 100 if percent else 1
    suffix = unit if unit is not None else "%" if percent else ""
    value = f"{scale * estimate:.3f}{suffix}"
    if interval and interval.get("lower") is not None:
        value += f" [{scale * interval['lower']:.3f}, {scale * interval['upper']:.3f}]"
    return value


def _gate(value):
    return "pass" if value is True else "fail" if value is False else "unknown"


def _table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)]


def build_report(run_dir, *, repo=ROOT):
    """Render saved results; never fit, select, execute a model, or change inputs."""
    run, repo = Path(run_dir).resolve(), Path(repo).resolve()
    registration = _json(run / "preregistration.json")
    config = registration["config"]
    terminal = run / "promotion_decision.json"
    incomplete = _json(run / "incomplete.json") if (run / "incomplete.json").exists() else None
    decision = _json(terminal) if terminal.exists() else None
    if decision is None and incomplete is None:
        raise ValueError("report requires a terminal decision or explicit incomplete record")
    try:
        relative = run.relative_to(repo).as_posix()
    except ValueError:
        relative = run.as_posix()
    lines = ["# Frozen C-29 causal expert recognition", "",
             f"Run: `{relative}`. Preregistered UTC: `{registration['utc']}`.", ""]
    if decision is None:
        lines += ["## Outcome", "", "Incomplete: no quality or economic conclusion is established.", "",
                  f"Recorded failure: `{incomplete['error_type']}` — {incomplete['error']}", "",
                  f"Elapsed single-GPU wall budget: {_number(incomplete.get('single_gpu_wall_elapsed_seconds'))} seconds.", ""]
    elif not decision["confirmation_pass"]:
        lines += ["## Outcome", "", "**Independent capability confirmation failed.** The cycle stopped before selector fitting,",
                  "development selection, fresh held-out evaluation, runtime quality or timing. No alternative",
                  "partition was substituted. Recognition itself was **not tested**; this is not a router failure",
                  "or a finding that every partition is unrecognizable.", ""]
    else:
        eligible = [name for name, value in decision["recognition"].items()
                    if value["eligible_for_runtime_quality"]]
        promoted = decision.get("promoted_policy_families", [])
        if promoted:
            outcome = "Measured economic gates passed for: " + ", ".join(promoted) + "."
        elif eligible:
            outcome = "Recognition passed for " + ", ".join(eligible) + "; economic success was not established."
        else:
            outcome = "Independent capability confirmed, but no policy family cleared the recognition continuation gate."
        lines += ["## Outcome", "", "**" + outcome + "**", "",
                  "The frozen cheap candidate portfolio and the causal recognition problem are distinct. Teacher",
                  "oracle choices are not a deployable router. All-dense policies can meet fidelity but are not",
                  "recognition success.", ""]
    lines += ["This is a new study of one discovery-selected partition. The preceding construction study's",
              "all-three-construction-seed failure and skipped later stages remain unchanged. C-29 and fixed",
              "candidate 8 (`[0,1,2,4,6,7]`) were selected using old development evidence, never these new outcomes.", "",
              "## Frozen scope and data", "",
              f"Model: `{config['model']}` at `{config['revision']}`; {config['dtype']}, {config['attention']}.",
              "Original model weights frozen; only the last MLP is sparsified. Eight 608-neuron experts,",
              "28 six-of-eight choices, 3,648/4,864 active neurons for each cheap call; no residual correction.", "",
              "Manifest source: pinned human-written Dolly. All 6,163 registered historical prompts and all 2,304",
              "prior expert-study contexts are excluded; exact source-document and instruction-family IDs",
              "and lexical similarity screening separate splits. Source responses are omitted. The 12.5%",
              "enterprise stratum is a keyword-topic screen, not a production-workload benchmark. Category",
              "holdout and lexical independence do not establish broad semantic or article independence.", ""]
    environment_path = run / "environment.json"
    if environment_path.exists():
        environment = _json(environment_path)
        lines += [(f"Measured environment: {environment['gpu']}; PyTorch {environment['torch']}; "
                   f"CUDA {environment['cuda']}; Transformers {environment['transformers']}."), ""]
    lines += _table(["Split", "Frozen contexts"], [[name, size] for name, size in config["split_sizes"].items()])
    lines += ["", "Confirmation is 64 matched + 64 OOD contexts and never trains/selects a router. The 1,728-context",
              "recipe was reduced prospectively from 2,816 because only 41 fresh OOD enterprise rows remained",
              "under the unchanged source/exclusion/quota constraints. OOD enterprise examples are creative-writing-heavy.", "",
              f"Sampled actual continuation positions: `{config['continuation_positions']}`, dense prefixes,",
              f"batch {config['batch_size']}, stop recording after EOS; max prompt length {config['max_prompt_tokens']} tokens.",
              "All intervals below are 95% whole-context bootstrap intervals (2,000 draws), not independent-token",
              "intervals.", ""]
    if (run / "recognition_metrics.json").exists():
        lines += ["Seed-mean contrasts condition on the three tested seeds; they do not estimate future-seed uncertainty.", ""]
    captures = []
    for split in config["split_sizes"]:
        path = run / f"capture-{split}.json"
        if path.exists():
            value = _json(path)
            captures.append([split, value["contexts_evaluated"], value["states"],
                             len(value["zero_state_context_ids"]), value["prompt_truncations"]])
    if captures:
        lines += _table(["Actually captured", "Contexts", "States", "Zero-state contexts", "Truncated prompts"], captures)
        lines += [""]
    path = run / "confirmation_decision.json"
    if path.exists():
        confirmation = _json(path)
        lines += ["## Capability confirmation", "", "Each stratum must achieve >=80% acceptable-candidate coverage and >=5 percentage-point ideal",
                  "active-width advantage over the frozen fixed-plus-dense oracle. Acceptable means exact",
                  "greedy agreement and KL(dense||candidate)<0.01. Point engineering gates and statistically",
                  "resolved passes are reported separately.", ""]
        records = []
        for name, value in confirmation["strata"].items():
            ci = value["confidence_intervals"]
            records.append([name, value["n_contexts"], value["n_states"],
                _interval(value["cheap_coverage"], ci["cheap_coverage"]),
                _interval(value["fixed_cheap_coverage"], ci["fixed_cheap_coverage"]),
                _interval(value["portfolio_advantage"], ci["portfolio_advantage"], unit=" pp"),
                _gate(value["engineering_gate"]["point_pass"]),
                _gate(value["engineering_gate"]["statistically_resolved_pass"])])
        lines += _table(["Stratum", "Contexts", "States", "Portfolio coverage [CI]", "Fixed coverage [CI]",
                         "Ideal width advantage [CI]", "Point", "Resolved pass"], records)
        for name, value in confirmation["strata"].items():
            interval = value["confidence_intervals"]["cheap_coverage"]
            if value["cheap_coverage"] < 0.8 <= interval["upper"]:
                lines += ["", f"The {name} coverage interval crosses the 80% threshold. The observed sample misses",
                          "the preregistered point-estimate engineering gate; this is **not** a statistically",
                          "resolved finding that population coverage is below 80%. The frozen stop rule still applies."]
        matrix_path = run / "matrix-confirmation.npz"
        if matrix_path.exists():
            state_records = [json.loads(line) for line in
                             (run / "states-confirmation.jsonl").read_text().splitlines()]
            with np.load(matrix_path, allow_pickle=False) as matrix:
                controls = [[name, _percent(float(np.mean(matrix[prefix + "_token"] == matrix["dense_token"]))),
                             _number(float(np.mean(matrix[prefix + "_kl"], dtype=np.float64)))]
                            for name, prefix in (("All-eight original order", "control"),
                                                 ("All-eight expert-permuted order", "permuted_control"))]
                candidate_measurements = int(matrix["kl"].size)
                acceptable = ((matrix["kl"] < 0.01)
                              & (matrix["candidate_token"] == matrix["dense_token"][:, None])).any(1)
                task_rows = []
                for task in ("creative_writing", "summarization"):
                    selected = np.array([row["task"] == task for row in state_records])
                    if selected.any():
                        task_rows.append([task, int(acceptable[selected].sum()), int(selected.sum()),
                                          _percent(float(acceptable[selected].mean()))])
            lines += ["", f"Retained raw cheap-candidate measurements: {candidate_measurements:,}.", ""]
            lines += _table(["Numerical control", "Greedy agreement", "Mean KL"], controls)
            lines += ["", "The expert-permuted all-eight reduction is a numerical control, not an assumed bitwise",
                      "equivalent implementation. A nonzero KL can coexist with perfect greedy agreement.", ""]
            if task_rows:
                lines += ["Post-hoc descriptive OOD task breakdown (not additional gates or a tuning rule):", ""]
                lines += _table(["Task", "States with acceptable choice", "States", "Coverage"], task_rows)
        parity_path = run / "full_model_cached_validation.json"
        if parity_path.exists():
            parity = _json(parity_path)
            lines += ["", (f"Full-model substitution versus cached evaluator: {len(parity)} layouts "
                      f"(28 cheap plus all-eight), {parity[0]['states']} prefixes per layout. "
                      f"Exact logit equality in every check: `{all(row['exact_logits'] for row in parity)}`.")]
        lines += ["", "Ideal active width = 1 − 0.25 × cheap fraction. This is an arithmetic proxy only. Sparsifying 25%",
                  "of one of 24 MLPs removes at most about 1.04% of aggregate MLP arithmetic before router",
                  "overhead; it is not a latency, whole-model FLOP or memory-traffic measurement.", ""]
    metrics_path = run / "recognition_metrics.json"
    if metrics_path.exists():
        metrics = _json(metrics_path)
        choices = _json(run / "frozen_policy_choices.json")
        paired = _json(run / "paired_portfolio_minus_binary.json")
        decisions = _json(run / "recognition_decision.json")
        lines += ["## Recognition results: every seed", "",
                  "Both families receive only final-MLP input x. Linear and 64-wide SiLU architectures each",
                  "received 200 Adam updates with batch 64/lr 0.001. Calibration-only normalization, identical",
                  "paired minibatch schedules and shared initial backbones; 28 vs 1 output heads mean unequal",
                  "parameter counts/arithmetic despite equal backbone/update budgets. Final iterate only.", "",
                  "Architecture/threshold selection used development only, then all choices/checkpoints were",
                  "sealed before fresh held-out access. Fidelity gates: >=99% greedy agreement and mean KL <0.01.", ""]
        records = []
        for family in ("portfolio", "binary"):
            for seed in config["seeds"]:
                choice = choices[family][str(seed)]
                for split in ("development", "heldout_matched", "heldout_ood"):
                    value = metrics[family][str(seed)][split]
                    ci = value["intervals"]
                    records.append([family, seed, split, f"{choice['architecture']} / {choice['threshold']}",
                        _interval(value["greedy_agreement"], ci.get("greedy_agreement")),
                        _number(value["mean_kl"]), _interval(value["cheap_fraction"], ci.get("cheap_fraction")),
                        _percent(value["false_cheap_fraction"]), _gate(value["quality_gate"]["point_pass"]),
                        _gate(value["quality_gate"]["statistically_resolved_pass"])])
        lines += _table(["Family", "Seed", "Split", "Architecture / threshold", "Agreement [CI]", "Mean KL",
                         "Cheap use [CI]", "False cheap/all", "Point", "Resolved"], records)
        lines += ["", "False-cheap means the selected cheap candidate violates either local greedy or KL <0.01 acceptance.",
                  "All raw policy summaries retain conditional false-cheap fractions, dense fractions, KL p95/p99/max,",
                  "width proxies, counts and intervals. A learned confidence score is not a deterministic certificate.", "",
                  "### Paired portfolio minus binary", ""]
        lines += _table(["Split", "Cheap-use difference [95% CI]", "Agreement difference [95% CI]", "Mean KL difference [95% CI]"],
            [[split, _interval(value["cheap_fraction"]["estimate"], value["cheap_fraction"]),
              _interval(value["greedy_agreement"]["estimate"], value["greedy_agreement"]),
              _interval(value["mean_kl"]["estimate"], value["mean_kl"], percent=False)]
             for split, value in paired.items()])
        lines += ["", "Differences are paired over contexts after averaging seeds. Portfolio promotion additionally",
                  "requires the cheap-use advantage's lower bound to exceed zero on both held-out sets.", ""]
        lines += _table(["Family", "All seeds fidelity", "All seeds nonzero cheap use", "Extra portfolio advantage", "Runtime eligible"],
            [[family, _gate(value["all_seed_quality_point_pass"]), _gate(value["all_seed_nonzero_cheap_use"]),
              "not required" if family == "binary" else _gate(value["portfolio_advantage_pass"]),
              _gate(value["eligible_for_runtime_quality"])] for family, value in decisions.items()])
        lines += ["", "Teacher-only portfolio/fixed oracles, always-fixed, original dense, random candidate selection",
                  "on learned cheap positions, and separately randomized cheap positions with matched counts",
                  "are retained in each `policies-<split>.json`/`.npz`. They are diagnostics, not tuning inputs.", ""]
    lines += ["## Runtime quality and measured economics", ""]
    runtime = decision.get("runtime") if decision else None
    policies = runtime.get("policies", {}) if isinstance(runtime, dict) else {}
    if not policies:
        reason = ("independent capability confirmation failed" if decision and not decision["confirmation_pass"]
                  else "the recognition continuation gate failed" if decision else "the run is incomplete")
        lines += [f"Not run: {reason}.",
                  "Inference latency, correction/routing savings and physical DRAM traffic are **unknown**, not zero.",
                  "No Nsight counter measurement was attempted without the required quality gate.", ""]
    else:
        for family, result in policies.items():
            lines += [f"### {family}", "", "Prospective runtime primary: seed 17. Genuine batch 1 dense-prefix capture, actual routed packed",
                      "execution and full-model/cache parity precede timing. No threshold refit follows runtime checks.", ""]
            lines += _table(["Split", "Agreement", "Mean KL", "Cheap use", "Quality point"],
                [[split, _percent(value["greedy_agreement"]), _number(value["mean_kl"]),
                  _percent(value["cheap_fraction"]), _gate(value["quality_gate"]["point_pass"])]
                 for split, value in result["quality"].items()])
            lines += ["", f"Timing status: `{result.get('timing_status', 'unknown')}`.", ""]
            for scope in ("isolated", "cached_decode"):
                measured = result.get(scope)
                if not measured:
                    continue
                records = []
                for row in measured.get("rows", []):
                    contrasts = []
                    for baseline in ("dense:eager", "dense:graph"):
                        contrast = row["versus_dense"].get(baseline)
                        contrasts.append("unknown" if not contrast else
                            f"{contrast['ratio_of_means']:.3f}x "
                            f"[{contrast['paired_block_bootstrap_ci95'][0]:.3f}, "
                            f"{contrast['paired_block_bootstrap_ci95'][1]:.3f}]")
                    records.append([row["method"], f"{row['mean_wall_ms']:.6f}", *contrasts,
                                    _gate(row["resolved_savings"])])
                lines += [f"{scope}: {measured['scope']}", ""]
                if records:
                    lines += _table(["Method", "Mean wall ms/state", "vs eager dense [CI]", "vs graph dense [CI]", "Savings resolved"], records)
                else:
                    lines += ["No valid complete timing comparison; " + json.dumps(measured.get("failure"))]
                memory = measured.get("memory", {})
                lines += ["", (f"Runtime dense frequency: {_percent(measured.get('dense_frequency'))}; "
                          f"extra resident bytes before graphs: {memory.get('extra_resident_bytes_before_graphs', 'unknown')}; "
                          f"candidate banks: {memory.get('candidate_banks', 'unknown')}."), ""]
            counters = result.get("physical_counters")
            if counters:
                lines += [f"Physical counter status: `{counters.get('status', 'unknown')}`. "
                          + str(counters.get("exact_failure") or counters.get("reason") or
                                "Raw counter output requires region validation; physical bytes remain unknown."), ""]
        lines += ["Timing includes normalization, selector, threshold, host synchronization, dispatch, launches",
                  "and fallback. Packed candidate banks duplicate overlapping weights. Resident bytes are not",
                  "measured DRAM traffic. Cached decode is fixed-prefix one-token replay, not free-generation",
                  "throughput; prefill/cache allocation/graph capture are excluded. Economic success requires",
                  "lower speedup bounds >1 against both eager and graph dense at isolated and full-model scopes.", ""]
    lines += ["Physical DRAM traffic remains unknown unless an explicit valid counter artifact establishes it.",
              "No photonic/simulated bandwidth claim and no free-generation/task-quality claim is made.", "",
              "## Integrity and reproduction", ""]
    if decision:
        lines += [f"Original weights unchanged: `{decision['original_weights_unchanged']}` (execution-record assertion).",
                  (f"Single-GPU wall budget used: {decision['single_gpu_wall_elapsed_seconds']:.3f} seconds of "
                  f"{decision['gpu_budget_ceiling_seconds']} seconds; this is not inference latency."), ""]
    lines += ["Registered input snapshots and hashes: `preregistration.json` / `registered_sources/`."]
    if metrics_path.exists():
        lines += ["Raw matrices, state-prefix metadata, all fitted checkpoints/loss logs, choices/actions/scores,",
                  "controls and any conditional runtime evidence remain under the run directory."]
    else:
        lines += ["This run retains the confirmation matrix, state-prefix metadata, controls and stop decision.",
                  "No fitted selector checkpoint, training log, development choice or runtime measurement exists:",
                  "those stages were not executed. Their implementation is present but untested on this study's model outputs."]
    lines += ["Activation caches",
              "may be local ignored files; their hashes and reconstruction provenance are recorded.", "",
              "```powershell", f"python scripts/verify_expert_recognition.py --run-dir {relative} --validate-data",
              f"python experiments/expert_recognition_report.py --run-dir {relative}",
              ".\\scripts\\run_expert_recognition.ps1  # new output directory; do not overwrite this run",
              "```", "", "The verifier recomputes recorded metrics, gates and selection from raw artifacts without a",
              "language model or GPU; it does not independently remeasure the GPU or prove chronology from hashes alone.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--report", type=Path, default=ROOT / "EXPERT_RECOGNITION_REPORT.md")
    args = parser.parse_args()
    report = build_report(args.run_dir, repo=args.repo)
    args.report.write_bytes(report.encode("utf-8"))
    print(str(args.report.resolve()))


if __name__ == "__main__":
    main()
