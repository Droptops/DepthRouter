"""Render a bounded expert audit from frozen JSON/NPZ evidence, without a model.

The report does not tune, rank held-out partitions, or change promotion gates.
It verifies complete-run raw matrices against summaries and faithfully renders
partial runs when an incomplete marker is present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from depth_router.expert_audit_metrics import (
    candidate_combinations,
    candidate_state_metrics,
    numerical_control_summary,
    paired_context_difference,
    summarize_candidates,
)

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("calibration", "development", "heldout_matched", "heldout_ood")
ARM_NAMES = {
    "A": "balanced random",
    "B": "balanced co-activation",
    "C": "hidden-output reconstruction",
    "D": "final-distribution KL",
}


def _read(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _pct(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "unknown"
    return f"{100 * value:+.2f}" if signed else f"{100 * value:.2f}"


def _ci(interval: dict[str, Any], *, percent: bool = True, signed: bool = False) -> str:
    factor = 100 if percent else 1
    fmt = ("+" if signed else "") + (".3f" if percent else ".6f")
    return (f"{format(factor * interval['estimate'], fmt)} "
            f"[{format(factor * interval['lower'], fmt)}, "
            f"{format(factor * interval['upper'], fmt)}]")


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines += ["| " + " | ".join(str(cell).replace("|", "\\|") for cell in row) + " |"
              for row in rows]
    return "\n".join(lines)


def _failure_label(summary: dict[str, Any]) -> str:
    if summary["split"] == "calibration":
        return "fit-data diagnostic"
    gate = summary["engineering_gate"]
    failures = []
    if not gate["numerically_interpretable"]:
        failures.append("numerical control")
    if not gate["coverage_point_pass"]:
        failures.append("coverage")
    if not gate["portfolio_advantage_point_pass"]:
        failures.append("portfolio advantage")
    if failures:
        return "FAIL: " + ", ".join(failures)
    return "PASS; resolved" if gate["statistically_resolved_pass"] else "PASS; unresolved CI"


def _validate_matrix(matrix: dict[str, np.ndarray], records: list[dict[str, Any]]) -> None:
    states = len(matrix["dense_token"])
    if matrix["kl"].shape != (states, 28):
        raise ValueError("raw matrix does not contain all 28 candidates for every state")
    if not np.array_equal(matrix["selected_experts"], np.asarray(candidate_combinations())):
        raise ValueError("raw matrix candidate ordering differs from the canonical enumeration")
    if len(records) != states:
        raise ValueError("state metadata count differs from the raw matrix")
    if not np.array_equal(matrix["context_ids"], [row["id"] for row in records]):
        raise ValueError("raw matrix contexts do not align with state metadata")
    if not np.array_equal(matrix["positions"], [row["position"] for row in records]):
        raise ValueError("raw matrix continuation positions do not align with state metadata")


def _verify_summary(matrix: dict[str, np.ndarray], summary: dict[str, Any]) -> None:
    vectors = candidate_state_metrics(
        matrix["kl"], matrix["candidate_token"] == matrix["dense_token"][:, None],
        fixed_selection=summary["fixed_selection"], candidate_experts=matrix["selected_experts"],
    )
    for metric in ("cheap_coverage", "fixed_cheap_coverage", "portfolio_advantage"):
        if not np.isclose(vectors[metric].mean(), summary[metric], rtol=0, atol=1e-12):
            raise ValueError(f"stored summary disagrees with raw matrix: {metric}")
    if summary["n_states"] != len(matrix["dense_token"]):
        raise ValueError("stored summary state count disagrees with raw matrix")
    if summary["n_contexts"] != len(np.unique(matrix["context_ids"])):
        raise ValueError("stored summary context count disagrees with raw matrix")


def load_evidence(run_dir: Path) -> dict[str, Any]:
    """Load and cross-check artifacts; never load weights or execute a model."""

    prereg = _read(run_dir / "preregistration.json")
    if prereg is None:
        raise ValueError("run directory must contain preregistration.json")
    config = prereg["config"]
    promotion = _read(run_dir / "promotion_decision.json", {})
    incomplete = _read(run_dir / "incomplete.json")
    complete = promotion.get("phase1_complete", False)
    summaries = _read(run_dir / "all_summaries.json", {})
    if not summaries:
        for path in sorted(run_dir.glob("summary-*.json")):
            summary = _read(path)
            key = path.stem.removeprefix("summary-" + summary["split"] + "-")
            summaries.setdefault(key, {})[summary["split"]] = summary
    fixed = _read(run_dir / "fixed_selection.json", {})
    captures = {split: _read(run_dir / f"capture-{split}.json") for split in SPLITS}
    records = {split: _jsonl(run_dir / f"states-{split}.jsonl") for split in SPLITS}
    keys = [f"{arm}-{seed}" for arm in ARM_NAMES for seed in config["seeds"]]
    matrices, controls, task_rows, margin_rows, artifacts = {}, [], [], [], []
    for split in SPLITS:
        for key in keys:
            path = run_dir / f"matrix-{split}-{key}.npz"
            if not path.exists():
                if complete:
                    raise ValueError(f"complete run is missing raw candidate evidence: {path.name}")
                continue
            with np.load(path, allow_pickle=False) as source:
                matrix = {name: source[name] for name in source.files}
            _validate_matrix(matrix, records[split])
            matrices[(split, key)] = matrix
            artifacts.append({"file": path.name, "states": len(matrix["dense_token"]),
                              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            summary = summaries.get(key, {}).get(split)
            if summary is None and key in fixed:
                if complete and split != "calibration":
                    raise ValueError(f"complete run is missing summary: {split}/{key}")
                summary = summarize_candidates(
                    matrix["kl"], matrix["candidate_token"] == matrix["dense_token"][:, None],
                    matrix["hidden_mse"], matrix["context_ids"], matrix["dense_margin"],
                    fixed_selection=fixed[key], split=split,
                    candidate_experts=matrix["selected_experts"],
                    control_kl=matrix["control_kl"],
                    control_agreement=matrix["control_token"] == matrix["dense_token"],
                    bootstrap_repetitions=config["bootstrap_repetitions"],
                    bootstrap_seed=config["bootstrap_seed"],
                )
                summaries.setdefault(key, {})[split] = summary
            if summary is not None:
                _verify_summary(matrix, summary)
            for name in ("control", "permuted_control"):
                control = numerical_control_summary(
                    matrix[name + "_kl"], matrix[name + "_token"] == matrix["dense_token"],
                    matrix["context_ids"], bootstrap_repetitions=config["bootstrap_repetitions"],
                    bootstrap_seed=config["bootstrap_seed"],
                )
                controls.append({"split": split, "partition": key, "layout": name, **control})
            acceptable = ((matrix["candidate_token"] == matrix["dense_token"][:, None])
                          & (matrix["kl"] < 0.01)).any(axis=1)
            tasks = np.array([row["task"] for row in records[split]])
            for task in sorted(set(tasks)):
                selected = tasks == task
                interval = paired_context_difference(
                    acceptable[selected].astype(float), np.zeros(selected.sum()),
                    matrix["context_ids"][selected],
                    bootstrap_repetitions=config["bootstrap_repetitions"],
                    bootstrap_seed=config["bootstrap_seed"],
                )
                task_rows.append([key, split, task, int(selected.sum()), _ci(interval)])
            if summary:
                for row in summary["by_dense_margin"]:
                    upper = row["margin_max_exclusive"]
                    label = f"[{row['margin_min_inclusive']}, {upper if upper is not None else 'inf'})"
                    margin_rows.append([key, split, label, row["n_states"],
                                        _pct(row["cheap_coverage"]),
                                        _pct(row["fixed_cheap_coverage"])])
    if complete and (incomplete is not None or not summaries):
        raise ValueError("complete promotion decision conflicts with incomplete or missing summaries")
    registered = run_dir / "registered_sources"
    return {
        "run_dir": run_dir, "prereg": prereg, "config": config, "promotion": promotion,
        "incomplete": incomplete, "complete": complete, "summaries": summaries,
        "paired": _read(run_dir / "paired_D_minus_C.json", {}),
        "environment": _read(run_dir / "environment.json", {}),
        "validation": _read(run_dir / "full_model_cached_validation.json", {}),
        "captures": captures, "records": records, "matrices": matrices, "controls": controls,
        "task_rows": task_rows, "margin_rows": margin_rows, "artifacts": artifacts,
        "data_audit": _read(registered / "data/expert_audit/data_audit.json", {}),
        "provenance": _read(registered / "data/expert_audit/source_provenance.json", {}),
        "construction": {seed: _read(run_dir / f"construction-{seed}.json")
                         for seed in config["seeds"]},
    }


def _outcome(evidence: dict[str, Any]) -> str:
    promotion = evidence["promotion"]
    if not evidence["complete"]:
        error = evidence["incomplete"]
        detail = f" {error['error_type']}: {error['error']}." if error else ""
        return ("The bounded capability audit is incomplete; no capability or deployment "
                f"success is established.{detail}")
    family = promotion["development_selected_family"]
    gates = promotion["phase1_gates"]
    passing = [key for key, gate in sorted(gates.items()) if gate["point_pass"]]
    resolved = [key for key in passing if gates[key]["statistically_resolved_pass"]]
    individual = (
        f"{len(passing)} of {len(gates)} individual partitions passed the user-specified "
        "coverage and portfolio-advantage gates on development and both held-out splits"
        + (f": {', '.join(passing)}. " if passing else ". ")
        + f"{len(resolved)} also passed the corresponding lower-confidence-bound checks. "
    )
    if promotion["selected_family_eligible"]:
        return (individual + f"Development selected family {family} ({ARM_NAMES[family]}). Its "
                "capability point gates passed across all seeds and required splits. This is "
                "an offline existence result; learned selection and measured savings require "
                "their separately recorded prerequisite stages.")
    reasons = {reason.split(":", 1)[-1]
               for key, gate in gates.items() if key.startswith(family + "-")
               for reason in gate["failure_reasons"]}
    labels = []
    if "insufficient_cheap_candidate_coverage" in reasons:
        labels.append("insufficient acceptable cheap candidates")
    if "insufficient_advantage_over_fixed_subset" in reasons:
        labels.append("insufficient portfolio advantage over the fixed subset")
    if "numerical_control_failed_or_missing" in reasons:
        labels.append("failed or missing numerical controls")
    explanation = "; ".join(labels) or "a required capability gate failed"
    return (individual + f"Development selected family {family} ({ARM_NAMES[family]}), which "
            "failed the predeclared additional family-level continuation criterion: every "
            f"one of the {len(evidence['config']['seeds'])} seeds must pass. Selected-family "
            f"failure: {explanation}. The result applies to these partitions, candidate sets, "
            "data, and fitting budget; it does not rule out useful cheap computation generally.")


def _compact_capability_rows(evidence: dict[str, Any]) -> list[list[Any]]:
    rows = []
    for key in sorted(evidence["summaries"], key=lambda key: (key[0], int(key.split("-")[1]))):
        values = []
        for split in SPLITS[1:]:
            summary = evidence["summaries"][key].get(split)
            values.append(
                f"{_pct(summary['cheap_coverage'])} / {_pct(summary['portfolio_advantage'])}"
                if summary else "not run"
            )
        gate = evidence["promotion"].get("phase1_gates", {}).get(key)
        status = ("PASS; resolved" if gate["statistically_resolved_pass"] else
                  "PASS; unresolved CI" if gate["point_pass"] else "FAIL") if gate else "incomplete"
        rows.append([key, *values, status])
    return rows


def _selected_failures(evidence: dict[str, Any]) -> str:
    promotion = evidence["promotion"]
    family = promotion.get("development_selected_family")
    rows = []
    for key, splits in sorted(evidence["summaries"].items()):
        if key[0] != family:
            continue
        for split in SPLITS[1:]:
            summary = splits.get(split)
            if summary is None:
                continue
            gate, ci = summary["engineering_gate"], summary["confidence_intervals"]
            if not gate["coverage_point_pass"]:
                interval = ci["cheap_coverage"]
                resolution = ("the interval includes the 80% threshold"
                              if interval["upper"] >= 0.8 else "the interval is below 80%")
                rows.append(f"{key} {split} coverage was {_ci(interval)}%; {resolution}")
            if not gate["portfolio_advantage_point_pass"]:
                rows.append(f"{key} {split} portfolio advantage was "
                            f"{_ci(ci['portfolio_advantage'])} percentage points")
    return ". ".join(rows) + "." if rows else "No selected-family point failure is recorded."


def _control_extrema_rows(evidence: dict[str, Any]) -> list[list[Any]]:
    rows = []
    for layout in ("control", "permuted_control"):
        values = [row for row in evidence["controls"] if row["layout"] == layout]
        if not values:
            continue
        rows.append([layout, len(values), f"{100 * min(row['greedy_agreement'] for row in values):.6f}",
                     f"{max(row['mean_kl'] for row in values):.9g}",
                     f"{sum(row['point_pass'] for row in values)}/{len(values)}"])
    return rows


def _capability_rows(evidence: dict[str, Any], selected_only: bool = False) -> list[list[Any]]:
    rows = []
    family = evidence["promotion"].get("development_selected_family")
    for key in sorted(evidence["summaries"], key=lambda key: (key[0], int(key.split("-")[1]))):
        if selected_only and key[0] != family:
            continue
        for split in SPLITS:
            if selected_only and split == "calibration":
                continue
            summary = evidence["summaries"][key].get(split)
            if summary is None:
                continue
            ci = summary["confidence_intervals"]
            rows.append([key, split, f"{summary['n_contexts']}/{summary['n_states']}",
                         _ci(ci["cheap_coverage"]), _pct(summary["fixed_cheap_coverage"]),
                         _pct(summary["oracle_with_dense"]["ideal_active_width"]),
                         _ci(ci["portfolio_advantage"]), _failure_label(summary)])
    return rows


CAPABILITY_HEADERS = ["Arm-seed", "Split", "Contexts/states", "Coverage % [95% CI]",
                      "Fixed coverage %", "Ideal active width %", "Advantage pp [95% CI]",
                      "Engineering gate"]


def render_reports(run_dir: Path, *, root: Path = ROOT) -> tuple[str, str]:
    """Return the full run report and concise root report as Markdown."""

    evidence = load_evidence(run_dir)
    config, promotion, env = evidence["config"], evidence["promotion"], evidence["environment"]
    try:
        relative = run_dir.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        relative = run_dir.resolve().as_posix()
    full = ["# DepthRouter final-MLP expert construction audit", _outcome(evidence)]
    if evidence["complete"] and not promotion["selected_family_eligible"]:
        full.append(_selected_failures(evidence))
    matrix_count = len(evidence["matrices"])
    candidate_evaluations = sum(len(matrix["dense_token"]) * 28
                                for matrix in evidence["matrices"].values())
    full += [
        (f"Run `{run_dir.name}` preserved {matrix_count} complete candidate matrices and "
        f"{candidate_evaluations:,} state/candidate measurements. Model weights unchanged: "
        f"`{promotion.get('original_weights_unchanged', 'unverified')}`. Promotion: "
        f"`{promotion.get('promote', False)}`."),
        "## Scope and frozen construction recipe",
        (f"Model `{config['model']}` at `{config['revision']}`; research base commit "
        f"`{config['base_commit']}`. Only the last of 24 MLPs is partitioned: 4,864 neurons "
        "in eight exactly balanced experts of 608 neurons. Every cheap candidate retains "
        "six experts (3,648 neurons, 75% retained), and all 28 subsets are measured. "
        "The original gate/up rows and matching down columns remain frozen. There is no "
        "residual correction network."),
        _table(["Arm", "Construction"], [[key, value] for key, value in ARM_NAMES.items()]),
        (f"Seeds: `{config['seeds']}`. Assignment recipe: "
        f"`{json.dumps(config['assignment'], sort_keys=True)}`. Co-activation recipe: "
        f"`{json.dumps(config['coactivation'], sort_keys=True)}`. C and D share initialization, "
        "minibatches, sampled candidate order, exact capacity constraints, optimization "
        "budget, and final-iterate selection. Per-token fitting uses the minimum loss among "
        "the shared sampled candidates. Hard forwards execute packed BF16 projections; "
        "a dense-activation straight-through gradient is an optimization surrogate. "
        "C minimizes normalized hidden reconstruction MSE; D minimizes KL(full || candidate) "
        "between actual normalized full-vocabulary distributions after residual addition, final "
        "RMSNorm and vocabulary head. No shortlist replaces the full distribution."),
    ]
    construction_rows = []
    for seed, audit in evidence["construction"].items():
        if audit is None:
            construction_rows.append([seed, "missing/incomplete", "unknown", "unknown", "unknown"])
            continue
        objectives = audit["objectives"]
        construction_rows.append([
            seed, "/".join(str(audit[key]) for key in (
                "initialization_equal", "minibatch_order_equal", "candidate_order_equal")),
            len(objectives["hidden"]["loss_by_step"]),
            len(objectives["output"]["loss_by_step"]),
            (f"{objectives['hidden']['neurons_changed_from_initial']}/"
            f"{objectives['output']['neurons_changed_from_initial']}"),
        ])
    full.append(_table(["Seed", "Init/batches/candidates matched", "C steps", "D steps",
                        "Neurons changed C/D"], construction_rows))
    full += [
        "## Data, evaluated positions, and exclusions",
        (f"Source: `{evidence['provenance'].get('dataset', 'unavailable')}` at "
        f"`{evidence['provenance'].get('revision', 'unavailable')}`, license "
        f"`{evidence['provenance'].get('license', 'unavailable')}`. Human-written prompts "
        "combine source instructions and available context; source answers are discarded. "
        "Matched tasks exclude creative writing and summarization; those categories form "
        "the OOD split. Each split has a 12.5% instruction-keyword enterprise stratum, "
        "which is a topic screen rather than a production enterprise benchmark. Source "
        "documents and exact instruction families are disjoint. Lexical duplicate screening "
        "does not establish independence of all semantic template families."),
        (f"Historical prompt exclusions: {evidence['data_audit'].get('historical_prompt_count', 'unknown')} "
        "prompt entries from the registered historical manifests. Historical outputs are "
        "not training or evaluation evidence in this cycle. Dataset manifest file SHA-256: "
        f"`{evidence['data_audit'].get('manifest_file_sha256', 'unavailable')}`."),
        (f"Requested continuation positions: `{config['continuation_positions']}`; these are "
        "cached decoding states after that many dense-generated continuation tokens. "
        "Every candidate uses the same dense prefix at each state. Contexts stop at EOS; "
        "no post-EOS states are treated as observed examples. Prompt token limit: "
        f"{config['max_prompt_tokens']} with left truncation. Missing positions are reported "
        "below and are excluded from state-level denominators; results are conditional "
        "on the observed dense continuations."),
    ]
    capture_rows = []
    for split, capture in evidence["captures"].items():
        if capture is None:
            capture_rows.append([split, "not run", "unknown", "unknown", "unknown", "unknown"])
            continue
        expected = capture["contexts_requested"] * len(config["continuation_positions"])
        capture_rows.append([split, capture["contexts_requested"], capture["contexts_evaluated"],
                             capture["states"], expected - capture["states"],
                             capture["prompt_truncations"]])
    full.append(_table(["Split", "Contexts requested", "Contexts with states", "States",
                        "Missing planned states (EOS)", "Truncated prompts"], capture_rows))
    position_rows = []
    for split, records in evidence["records"].items():
        counts = Counter(row["position"] for row in records)
        for position in config["continuation_positions"]:
            position_rows.append([split, position, counts[position]])
    full.append(_table(["Split", "Continuation position", "Observed states"], position_rows))
    full += [
        "## Capability for every arm and seed",
        ("A locally acceptable candidate matches the original dense greedy token and has "
        "KL(full || candidate) strictly below 0.01. Fixed subsets are selected on development "
        "by highest acceptable coverage, then lowest mean KL, then canonical candidate index. "
        "The same frozen subset is used for every subsequent split. The cheap-only oracle "
        "prefers greedy agreement, then minimum KL, then candidate index. The portfolio "
        "oracle takes an acceptable cheap candidate when available and dense otherwise. "
        "The fixed oracle applies the identical acceptance test to its single subset."),
        ("Coverage is the fraction of evaluated token states with an acceptable cheap "
        "candidate. Whole contexts, including all their continuation positions, are "
        f"resampled jointly for 95% percentile intervals ({config['bootstrap_repetitions']} "
        "draws). Token-state weighting is retained when EOS creates unequal context lengths. "
        "Paired seed comparisons average the specified seeds within each state before the "
        "same context draws; intervals are conditional on this seed set. Marginal intervals "
        "are not multiplicity-adjusted or deterministic certificates."),
        ("Engineering gates require coverage >=80% and portfolio ideal-active-width advantage "
        ">=5 percentage points on development and both held-out splits. An ideal oracle "
        "with cheap fraction a has average active width 1 - 0.25a; portfolio advantage is "
        "0.25 times (portfolio coverage - fixed coverage). This arithmetic proxy excludes "
        "routing, synchronization, movement, kernel launches, and head cost. Point passes "
        "are distinguished from findings whose corresponding lower confidence bounds pass."),
        ("A separate, predeclared family-level continuation rule requires all three seeds "
         "of the development-selected family to pass those gates. Thus individual partition "
         "successes remain valid capability findings even when that additional continuation "
         "rule stops the cycle. Family ranking uses development mean portfolio advantage, "
         "then coverage, then arm name; held-out results cannot replace the selected family "
         "or discard a failing seed."),
        _table(CAPABILITY_HEADERS, _capability_rows(evidence)),
        ("Calibration rows use the development-frozen fixed subset retrospectively for "
        "training-data diagnostics and do not contribute to promotion gates."),
        "## Raw fidelity and oracle dense routing",
        ("Original dense self-reference is 100% greedy agreement and zero KL by definition. "
        "The following table retains raw fixed and cheap-only oracle errors separately "
        "from the dense-option oracles. Both dense-option oracles have 100% greedy agreement "
        "by construction and use teacher information unavailable at deployment."),
    ]
    fidelity_rows = []
    for key in sorted(evidence["summaries"]):
        for split in SPLITS:
            summary = evidence["summaries"][key].get(split)
            if summary is None:
                continue
            fixed, oracle = summary["best_fixed"], summary["oracle_cheap_only"]
            portfolio, fixed_dense = summary["oracle_with_dense"], summary["fixed_oracle_with_dense"]
            fidelity_rows.append([
                key, split, ",".join(map(str, summary["fixed_selection"]["experts"])),
                f"{_pct(fixed['greedy_agreement'])}/{fixed['mean_kl']:.6g}",
                f"{_pct(oracle['greedy_agreement'])}/{oracle['mean_kl']:.6g}",
                f"{portfolio['mean_kl']:.6g}/{_pct(portfolio['dense_fraction'])}",
                f"{fixed_dense['mean_kl']:.6g}/{_pct(fixed_dense['dense_fraction'])}",
            ])
    full.append(_table(["Arm-seed", "Split", "Frozen fixed experts", "Fixed agree % / KL",
                        "Cheap oracle agree % / KL", "Portfolio+dense KL / dense %",
                        "Fixed+dense KL / dense %"], fidelity_rows))
    paired_rows = []
    for split in SPLITS:
        for metric, interval in evidence["paired"].get(split, {}).items():
            paired_rows.append([split, metric, _ci(interval, percent=metric != "oracle_kl", signed=True),
                                "raw KL" if metric == "oracle_kl" else "percentage points"])
    full += ["## Paired D minus C construction-objective comparison",
             _table(["Split", "Metric", "Difference [95% CI]", "Units"], paired_rows),
             ("Positive coverage, agreement, or portfolio-advantage differences favor D; "
             "negative KL differences favor D. These paired estimates do not imply that "
             "either arm passes the absolute engineering gates.")]
    full += [
        "## Numerical controls and actual model execution",
        ("The gating all-eight control uses original neuron index order, matching the "
        "globally index-sorted candidate layout. A separate expert-contiguous permutation "
        "control measures reduction-order sensitivity. Its results are shown explicitly; "
        "it is not silently substituted for the validated sorted layout. BF16 exactness "
        "is assessed from observations, not inferred from real-arithmetic permutation identities."),
    ]
    controls_rows = [[row["partition"], row["split"], row["layout"],
                      _ci(row["agreement_ci"]), f"{row['mean_kl']:.8g}",
                      "PASS" if row["point_pass"] else "FAIL"]
                     for row in evidence["controls"]]
    full.append(_table(["Arm-seed", "Split", "All-eight layout", "Agreement % [95% CI]",
                        "Mean KL", ">=99% and KL<0.01"], controls_rows))
    validation_rows = []
    for key, checks in evidence["validation"].items():
        validation_rows.append([key, len(checks), sum(check["states"] for check in checks),
                                all(check["exact_logits"] for check in checks),
                                max(check["max_abs_logit_error"] for check in checks)])
    full.append(_table(["Arm-seed", "Full-model substitution checks", "Checked state/candidates",
                        "Bitwise exact cached logits", "Maximum absolute logit error"], validation_rows))
    full.append("Cached dense residual/norm/head reconstruction is checked on each captured "
                "batch. Full-model substitution checks cover all 28 candidates and the sorted "
                "all-eight control on the recorded calibration replay batch; they do not "
                "establish correctness for untested batch shapes or deployment kernels.")
    full += ["## Gate decision, selection, and execution"]
    gate_rows = [[key, gate["point_pass"], gate["statistically_resolved_pass"],
                  "; ".join(gate["failure_reasons"]) or "none"]
                 for key, gate in sorted(promotion.get("phase1_gates", {}).items())]
    full.append(_table(["Arm-seed", "All-split point pass", "Resolved pass", "Failure reasons"], gate_rows))
    phase2, phase3 = promotion.get("phase2", "not run; audit incomplete"), promotion.get(
        "phase3", "not run; prerequisites incomplete")
    full.append(f"Phase 2: `{phase2}`. Phase 3: `{phase3}`. "
                "If the capability gate fails, learned routing, free-generation task "
                "evaluation and timing are skipped under the stopping rule. An oracle "
                "does not establish recognition from pre-execution inputs, and dense "
                "fallback fidelity does not establish useful savings.")
    full.append(f"Measured latency: `{promotion.get('measured_latency')}`. Measured physical "
                f"DRAM traffic: `{promotion.get('measured_dram_bytes')}`. Counter status: "
                f"`{promotion.get('physical_counter_attempt', 'not attempted; prerequisites incomplete')}`. "
                "Null means unknown, not zero. Without a passing selector there is no "
                "selector-plus-MLP timing, end-to-end cached-decoding benchmark, physical "
                "traffic measurement, or free-generation task-success result. Offline "
                "candidate-search costs are not deployment latency measurements.")
    elapsed = promotion.get("single_gpu_wall_elapsed_seconds",
                            (evidence["incomplete"] or {}).get("single_gpu_wall_elapsed_seconds"))
    full.append(f"Bounded audit elapsed wall time: `{elapsed}` seconds; single-GPU ceiling: "
                f"{config['gpu_wall_budget_seconds']} seconds. This includes loading, capture, "
                "construction, exhaustive offline measurement, and CPU waits; it is not GPU "
                "kernel time. Removing 25% of one of 24 equal-width MLPs saves at most "
                "approximately 1.04% of total MLP arithmetic before overhead. No model-wide "
                "speedup follows from that proxy.")
    full += ["## Execution environment",
             _table(["Field", "Recorded value"], [[key, env.get(key, "unavailable")]
                    for key in ("gpu", "torch", "transformers", "cuda", "python", "platform",
                                "model_class", "model_source_sha256", "dtype", "attention",
                                "allow_tf32", "cuda_graphs", "stage")]),
             (f"Capture batch size: {config['batch_size']}. Full environment and NVIDIA driver "
             "output are preserved in `environment.json`.")]
    full += [
        "## Relation to earlier evidence and primary sources",
        ("This is a separate diagnostic. Earlier individual-neuron prediction and naive "
        "contiguous pages failed their quality gates. The fused sparse study measured an "
        "isolated kernel benefit but dynamic prediction/top-k/gather overhead overwhelmed "
        "it; the graph-captured BF16 dense operator remained the strongest exact baseline. "
        "Those historical timings are not new measurements in this audit. PR 16 held the "
        "mask fixed while changing correction objectives; none of its 12 correction fits "
        "passed development. This audit changes expert construction and adds no correction. "
        "Earlier contribution-ranking oracles were heuristics; this audit exhaustively "
        "searches only its 28 frozen six-of-eight candidates, not all neuron subsets or "
        "possible partitions. Synthetic notebook examples are mathematical checks, not "
        "Qwen or hardware evidence."),
        ("[DOT-MoE](https://proceedings.mlr.press/v306/bamba26a.html) motivates balanced "
        "transport-based neuron assignments and straight-through optimization. "
        "[DynaMoE](https://arxiv.org/abs/2502.12325) motivates predicting sufficient "
        "computation per token. [Goal-oriented model reduction](https://arxiv.org/abs/1603.00336) "
        "motivates evaluating a quantity of interest rather than only intermediate-state "
        "reconstruction. These are conceptual foundations; this bounded final-MLP audit "
        "claims neither novelty nor faithful reproduction of those methods."),
        "## Reproduction and retained artifacts",
        "Run from the repository root with a fresh run ID:",
        ("```powershell\npowershell -ExecutionPolicy Bypass -File scripts/run_expert_construction.ps1 "
        "-RunId fresh-expert-audit\n```"),
        "Regenerate this report without model execution:",
        f"```text\npython experiments/expert_construction_report.py --run-dir {relative}\n```",
        ("`preregistration.json` and `registered_sources/` preserve source/config hashes before "
        "model access. `heldout_seal.json` freezes partitions, development-selected subsets, "
        "family selection and preregistration before held-out captures. `construction-*.json` "
        "preserves paired schedules/hashes and losses; `partition-*.json` contains every "
        "neuron assignment. `states-*.jsonl` preserves context IDs, actual prefix token IDs, "
        "positions, dense tokens and margins. All scalar candidate matrices, including "
        "failed arms, remain in `matrix-*.npz`. Large activation caches are local reproducible "
        "intermediates; capture files record their hashes. `promotion_decision.json` is the "
        "machine-readable gate record; `paired_D_minus_C.json` holds paired intervals. "
        "`incomplete.json`, when present, records interrupted work without claiming completion."),
        _table(["Raw matrix", "States", "SHA-256"],
               [[row["file"], row["states"], row["sha256"]] for row in evidence["artifacts"]]),
        "## Task-stratified capability diagnostics",
        ("These strata are descriptive, use the same fixed acceptance rule, and do not select "
        "partitions or change promotion thresholds."),
        _table(["Arm-seed", "Split", "Task", "States", "Coverage % [95% CI]"], evidence["task_rows"]),
        "## Dense-margin diagnostics",
        _table(["Arm-seed", "Split", "Dense top-two margin", "States", "Coverage %",
                "Fixed coverage %"], evidence["margin_rows"]),
    ]
    concise = [
        "# DepthRouter expert construction: bounded capability audit", _outcome(evidence),
        _selected_failures(evidence) if evidence["complete"] else "Incomplete stages remain unclaimed.",
        "## Experiment and bounded conclusion",
        (f"Evaluated {matrix_count} raw candidate matrices / {candidate_evaluations:,} "
         f"state-candidate pairs. Model `{config['model']}` at `{config['revision']}`; "
         f"base `{config['base_commit']}`; seeds `{config['seeds']}`. Only final-layer MLP "
         "computation changes: eight balanced 608-neuron experts, every six-of-eight subset, "
         "75% width retained. Original weights are frozen; no residual correction is added."),
        ("A = balanced random; B = balanced co-activation; C = hidden reconstruction; "
         "D = final full-vocabulary KL. C/D use paired initialization, data order, candidate "
         f"sampling, constraints and {config['assignment'].get('steps', 'unknown')} fitting "
         "updates per objective and seed. The exhaustive 28-candidate oracle answers "
         "existence only and is never a deployable selector."),
        ("Individual passes establish useful candidate portfolios within this audit. "
         "Family continuation additionally requires consistency across every preregistered "
         "seed. That extra rule does not erase successful individual partitions or establish "
         "a general impossibility result. No held-out seed, family, threshold, or extra "
         "training budget is selected after seeing these outcomes."),
        "## Every partition: coverage and advantage",
        ("Each split cell is acceptable-state coverage (%) / portfolio advantage over the "
         "development-frozen subset (percentage points). Coverage >=80% and advantage >=5 "
         "points are required on development and both held-out splits. The final column "
         "aggregates those user-specified engineering gates for each individual partition."),
        _table(["Arm-seed", "Development coverage / advantage", "Matched coverage / advantage",
                "OOD coverage / advantage", "Individual all-split gate"],
               _compact_capability_rows(evidence)),
        "## Development-selected family, with uncertainty",
        _table(CAPABILITY_HEADERS, _capability_rows(evidence, selected_only=True)),
        ("Intervals resample whole contexts and retain token-state weighting; they are "
         "conditional on the evaluated seed set and are not multiplicity-adjusted. "
         "A point failure whose interval includes the threshold is not a statistically "
         "resolved failure. Ideal active width is 1 - 0.25 × cheap coverage; it excludes "
         "routing, movement, synchronization and all hardware overhead."),
        "## Paired D minus C",
    ]
    paired_heldout = [row for row in paired_rows if row[0].startswith("heldout")]
    concise += [
        _table(["Split", "Metric", "Difference [95% CI]", "Units"], paired_heldout),
        ("Acceptable coverage requires both the same greedy token and KL<0.01. Greedy "
         "agreement is reported separately: a small agreement increase does not imply "
         "better local acceptability. Positive coverage/agreement/advantage changes favor "
         "D, whereas negative KL changes favor D. These conclusions concern this bounded "
         "construction recipe, not every possible output-distribution objective."),
        "## Numerical controls and data completeness",
        _table(["All-eight layout", "Arm/split controls", "Minimum agreement %", "Maximum mean KL",
                "Point passes"], _control_extrema_rows(evidence)),
        ("The original-index all-eight layout is the gating control and matches the "
         "globally sorted sparse layout. The expert-contiguous permutation is an additional "
         "reduction-order diagnostic. Full-model substitution checks and every raw control "
         "are retained in the detailed report; real-arithmetic identities are not used as "
         "evidence of BF16 exactness."),
        _table(["Split", "Contexts requested", "Contexts with states", "States",
                "Missing planned states (EOS)", "Truncated prompts"], capture_rows),
        (f"Dense-prefix continuation positions: `{config['continuation_positions']}`. "
         "No post-EOS states enter denominators. Fresh human-written Dolly source contexts "
         "have disjoint documents and exact lexical instruction families; 12.5% are "
         "enterprise-keyword topics. Creative writing and summarization are reserved OOD. "
         "Lexical exclusions do not guarantee semantic-family independence; source "
         "answers are unused."),
        "## Selection and execution",
        (f"Phase 2: `{phase2}`. Phase 3: `{phase3}`. Learned selector quality, free-generation "
         "task success, selector-plus-MLP latency, end-to-end decoding cost and physical DRAM "
         "traffic remain unmeasured when prerequisites fail. Unknown costs are retained as "
         "null; arithmetic proxies are not measured savings."),
        (f"Hardware: `{env.get('gpu', 'unavailable')}`; dtype `{env.get('dtype', 'unavailable')}`; "
         f"PyTorch `{env.get('torch', 'unavailable')}`, CUDA `{env.get('cuda', 'unavailable')}`, "
         f"Transformers `{env.get('transformers', 'unavailable')}`, attention "
         f"`{env.get('attention', 'unavailable')}`, batch {config['batch_size']}. "
         f"Audit wall time: `{elapsed}` seconds, bounded by {config['gpu_wall_budget_seconds']} "
         "seconds; this is not inference latency. Even ideal 25% removal from one of 24 "
         "MLPs is only about 1.04% of total MLP arithmetic before overhead."),
        "## Reproduction and retained evidence",
        ("```powershell\npowershell -ExecutionPolicy Bypass -File scripts/run_expert_construction.ps1 "
         "-RunId fresh-expert-audit\n```"),
        f"```text\npython experiments/expert_construction_report.py --run-dir {relative}\n```",
        (f"The [complete report]({relative}/REPORT.md) contains every arm/seed/split, raw "
         "fidelity, full controls, data provenance, task/margin strata, source hashes, "
         "and all raw matrix hashes. "
         f"The [machine-readable decision]({relative}/promotion_decision.json) records the "
         "individual gates and additional family-level continuation decision."),
        ("Earlier negative results remain intact. Historical contribution-ranking oracles "
         "and synthetic notebook checks are distinct from this exhaustive candidate audit. "
         "DOT-MoE, DynaMoE, and goal-oriented model reduction are conceptual sources, not "
         "claimed reproductions or novelty evidence; source links are in the detailed report."),
    ]
    return "\n\n".join(full) + "\n", "\n\n".join(concise) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--root-report", type=Path, default=ROOT / "EXPERT_CONSTRUCTION_REPORT.md")
    args = parser.parse_args()
    full, concise = render_reports(args.run_dir.resolve())
    (args.run_dir / "REPORT.md").write_text(full, encoding="utf-8")
    args.root_report.write_text(concise, encoding="utf-8")
    print(f"Wrote {args.run_dir / 'REPORT.md'} and {args.root_report}")


if __name__ == "__main__":
    main()
