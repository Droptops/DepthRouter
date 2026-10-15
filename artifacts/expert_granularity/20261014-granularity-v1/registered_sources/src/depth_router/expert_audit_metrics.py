"""Offline capability statistics for the preregistered eight-expert audit.

All oracle decisions use teacher observations and are diagnostic only. Nothing
in this module is a deployment selector. Bootstrap draws resample complete
contexts, preserving their continuation positions and all paired seed results.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

LOCAL_KL_LIMIT = 0.01
COVERAGE_GATE = 0.8
PORTFOLIO_ADVANTAGE_GATE = 0.05
NUMERICAL_AGREEMENT_GATE = 0.99
_KL_NEGATIVE_TOLERANCE = 1e-5


def candidate_combinations() -> tuple[tuple[int, ...], ...]:
    """Enumerate every six-of-eight subset once, in canonical index order."""

    return tuple(itertools.combinations(range(8), 6))


def _candidate_schema(candidate_experts: Sequence[Sequence[int]] | None) -> str:
    expected = candidate_combinations()
    if candidate_experts is not None:
        supplied = np.asarray(candidate_experts)
        if supplied.shape != (28, 6) or not np.array_equal(supplied, np.asarray(expected)):
            raise ValueError("candidate_experts must contain all 28 canonical six-of-eight subsets")
    return hashlib.sha256(json.dumps(expected, separators=(",", ":")).encode()).hexdigest()


def _boolean_array(values: Any, name: str) -> np.ndarray:
    value = np.asarray(values)
    if not np.isin(value, [False, True]).all():
        raise ValueError(f"{name} must contain only boolean agreement values")
    return value.astype(bool, copy=False)


def _candidate_arrays(kl: Any, agreement: Any) -> tuple[np.ndarray, np.ndarray]:
    divergence = np.asarray(kl, dtype=np.float64)
    agree = _boolean_array(agreement, "agreement")
    if divergence.ndim != 2 or divergence.shape[1] != 28 or divergence.shape[0] == 0:
        raise ValueError("KL must be a nonempty [states, 28] complete candidate matrix")
    if agree.shape != divergence.shape:
        raise ValueError("agreement must have the same [states, 28] shape as KL")
    if not np.isfinite(divergence).all() or np.any(divergence < -_KL_NEGATIVE_TOLERANCE):
        raise ValueError("KL must be finite and nonnegative within 1e-5 roundoff tolerance")
    return divergence, agree


def _context_index(context_ids: Any, n_states: int) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(context_ids)
    if ids.ndim != 1 or len(ids) != n_states:
        raise ValueError("context_ids must have one entry per token state")
    if ids.dtype.kind in "fc" and not np.isfinite(ids).all():
        raise ValueError("context_ids must be finite")
    unique, inverse = np.unique(ids, return_inverse=True)
    return unique, inverse


def _bootstrap_columns(
    values: np.ndarray,
    context_ids: Any,
    repetitions: int,
    seed: int,
) -> list[dict[str, Any]]:
    if repetitions < 1:
        raise ValueError("bootstrap_repetitions must be positive")
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("bootstrap values must be finite nonempty [states, metrics]")
    contexts, inverse = _context_index(context_ids, len(values))
    counts = np.bincount(inverse)
    sums = np.zeros((len(contexts), values.shape[1]), dtype=np.float64)
    np.add.at(sums, inverse, values)
    rng = np.random.default_rng(seed)
    draws = np.empty((repetitions, values.shape[1]), dtype=np.float64)
    # Chunk draws to bound memory even for the calibration split. Each draw
    # selects contexts once for every column, so paired metrics remain paired.
    for start in range(0, repetitions, 128):
        stop = min(start + 128, repetitions)
        indices = rng.integers(0, len(contexts), size=(stop - start, len(contexts)))
        draws[start:stop] = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)[:, None]
    intervals = np.quantile(draws, [0.025, 0.975], axis=0)
    return [
        {
            "estimate": float(values[:, column].mean()),
            "lower": float(intervals[0, column]),
            "upper": float(intervals[1, column]),
            "confidence": 0.95,
            "method": "percentile_context_bootstrap",
            "n_contexts": len(contexts),
            "n_states": len(values),
            "resamples": repetitions,
            "bootstrap_seed": seed,
            "weighting": "token_states_with_whole_context_resampling",
        }
        for column in range(values.shape[1])
    ]


def paired_context_difference(
    left: Any,
    right: Any,
    context_ids: Any,
    *,
    bootstrap_repetitions: int = 2000,
    bootstrap_seed: int = 17,
) -> dict[str, Any]:
    """Estimate left minus right, paired by state and optionally by seed.

    Inputs are [states] or [seeds, states] in identical state/seed order.
    Seeds are averaged before resampling contexts; this conditions inference
    on the specified seed set and does not estimate variation over new seeds.
    For token-varying context lengths, each sampled context brings every state
    with it, and the estimator retains its token-state weighting.
    """

    lhs = np.asarray(left, dtype=np.float64)
    rhs = np.asarray(right, dtype=np.float64)
    if lhs.shape != rhs.shape or lhs.ndim not in (1, 2) or lhs.size == 0:
        raise ValueError("paired arrays must share nonempty [states] or [seeds, states] shape")
    if not np.isfinite(lhs).all() or not np.isfinite(rhs).all():
        raise ValueError("paired arrays must be finite")
    difference = lhs - rhs
    seed_count = 1 if lhs.ndim == 1 else lhs.shape[0]
    if difference.ndim == 2:
        difference = difference.mean(axis=0)
    result = _bootstrap_columns(
        difference[:, None], context_ids, bootstrap_repetitions, bootstrap_seed
    )[0]
    result.update({"direction": "left_minus_right", "n_paired_seeds": seed_count})
    result["seed_uncertainty"] = "conditional_on_the_evaluated_seed_set"
    return result


def select_fixed_candidate(
    kl: Any,
    agreement: Any,
    *,
    split: str = "development",
    candidate_experts: Sequence[Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Freeze the fixed subset on development, never on held-out outputs.

    Maximize local acceptable-state coverage, then minimize mean KL over all
    development states, then prefer the smallest canonical candidate index.
    """

    if split != "development":
        raise ValueError("fixed-candidate selection is permitted on development only")
    schema = _candidate_schema(candidate_experts)
    divergence, agree = _candidate_arrays(kl, agreement)
    coverage = (agree & (divergence < LOCAL_KL_LIMIT)).mean(axis=0)
    mean_kl = divergence.mean(axis=0)
    index = min(range(28), key=lambda item: (-coverage[item], mean_kl[item], item))
    return {
        "index": index,
        "experts": list(candidate_combinations()[index]),
        "selected_on": "development",
        "selection_rule": "maximum_acceptable_coverage_then_minimum_mean_kl_then_index",
        "candidate_schema_sha256": schema,
        "development_coverage": float(coverage[index]),
        "development_mean_kl": float(mean_kl[index]),
        "development_states": len(divergence),
    }


def _fixed_index(fixed_selection: Mapping[str, Any], schema: str) -> int:
    index = fixed_selection.get("index")
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < 28:
        raise ValueError("fixed_selection must contain a canonical candidate index")
    if fixed_selection.get("selected_on") != "development":
        raise ValueError("fixed_selection must have development-only provenance")
    if fixed_selection.get("candidate_schema_sha256") != schema:
        raise ValueError("fixed_selection candidate schema does not match this matrix")
    if fixed_selection.get("experts") != list(candidate_combinations()[index]):
        raise ValueError("fixed_selection experts do not match its frozen index")
    return index


def candidate_state_metrics(
    kl: Any,
    agreement: Any,
    *,
    fixed_selection: Mapping[str, Any],
    candidate_experts: Sequence[Sequence[int]] | None = None,
) -> dict[str, np.ndarray]:
    """Return paired state statistics for offline aggregation across arms/seeds."""

    schema = _candidate_schema(candidate_experts)
    divergence, agree = _candidate_arrays(kl, agreement)
    fixed = _fixed_index(fixed_selection, schema)
    acceptable = agree & (divergence < LOCAL_KL_LIMIT)
    coverage = acceptable.any(axis=1)
    fixed_coverage = acceptable[:, fixed]
    # Lexicographic oracle: greedy agreement first, then KL, then the canonical
    # index via argmin's first-index tie behavior. If none agrees, minimize KL.
    same_token = np.where(agree, divergence, np.inf).argmin(axis=1)
    oracle_index = np.where(agree.any(axis=1), same_token, divergence.argmin(axis=1))
    row = np.arange(len(divergence))
    oracle_agree = agree[row, oracle_index]
    oracle_kl = divergence[row, oracle_index]
    return {
        "cheap_coverage": coverage.astype(float),
        "fixed_cheap_coverage": fixed_coverage.astype(float),
        "ideal_active_width": 1.0 - 0.25 * coverage,
        "fixed_ideal_active_width": 1.0 - 0.25 * fixed_coverage,
        "portfolio_advantage": 0.25 * (coverage.astype(float) - fixed_coverage),
        "fixed_agreement": agree[:, fixed].astype(float),
        "fixed_kl": divergence[:, fixed],
        "oracle_agreement": oracle_agree.astype(float),
        "oracle_kl": oracle_kl,
        "oracle_with_dense_kl": np.where(coverage, oracle_kl, 0.0),
        "fixed_with_dense_kl": np.where(fixed_coverage, divergence[:, fixed], 0.0),
        "oracle_candidate_index": oracle_index,
    }


def _fidelity(kl: np.ndarray, agreement: np.ndarray, hidden_mse: np.ndarray) -> dict[str, float]:
    return {
        "greedy_agreement": float(agreement.mean()),
        "mean_kl": float(kl.mean()),
        "p95_kl": float(np.quantile(kl, 0.95)),
        "mean_hidden_mse": float(hidden_mse.mean()),
    }


def numerical_control_summary(
    kl: Any,
    agreement: Any,
    context_ids: Any,
    *,
    bootstrap_repetitions: int = 2000,
    bootstrap_seed: int = 17,
) -> dict[str, Any]:
    """Check actual all-eight execution against the original dense model."""

    divergence = np.asarray(kl, dtype=np.float64)
    agree = _boolean_array(agreement, "control_agreement")
    if divergence.ndim != 1 or agree.shape != divergence.shape or len(divergence) == 0:
        raise ValueError("numerical control arrays must have matching nonempty [states] shapes")
    if not np.isfinite(divergence).all() or np.any(divergence < -_KL_NEGATIVE_TOLERANCE):
        raise ValueError("control KL must be finite and nonnegative within roundoff tolerance")
    agreement_ci, kl_ci = _bootstrap_columns(
        np.stack([agree, divergence], axis=1),
        context_ids,
        bootstrap_repetitions,
        bootstrap_seed,
    )
    mean_kl = math.fsum(divergence.tolist()) / len(divergence)
    point_pass = bool(agree.mean() >= NUMERICAL_AGREEMENT_GATE and mean_kl < LOCAL_KL_LIMIT)
    return {
        "greedy_agreement": float(agree.mean()),
        "mean_kl": mean_kl,
        "agreement_ci": agreement_ci,
        "mean_kl_ci": kl_ci,
        "point_pass": point_pass,
        "statistically_resolved_pass": bool(
            point_pass and agreement_ci["lower"] >= NUMERICAL_AGREEMENT_GATE
            and kl_ci["upper"] < LOCAL_KL_LIMIT
        ),
        "reference": "original_dense_model",
    }


def summarize_candidates(
    kl: Any,
    agreement: Any,
    hidden_mse: Any,
    context_ids: Any,
    dense_margin: Any,
    *,
    fixed_selection: Mapping[str, Any],
    split: str,
    candidate_experts: Sequence[Sequence[int]] | None = None,
    control_kl: Any | None = None,
    control_agreement: Any | None = None,
    bootstrap_repetitions: int = 2000,
    bootstrap_seed: int = 17,
) -> dict[str, Any]:
    """Summarize a complete frozen candidate matrix, never tuning its fixed subset."""

    schema = _candidate_schema(candidate_experts)
    divergence, agree = _candidate_arrays(kl, agreement)
    fixed = _fixed_index(fixed_selection, schema)
    error = np.asarray(hidden_mse, dtype=np.float64)
    margin = np.asarray(dense_margin, dtype=np.float64)
    if error.shape != divergence.shape or not np.isfinite(error).all() or np.any(error < 0):
        raise ValueError("hidden_mse must be finite nonnegative [states, 28]")
    if margin.shape != (len(divergence),) or not np.isfinite(margin).all() or np.any(margin < 0):
        raise ValueError("dense_margin must be finite nonnegative [states]")
    contexts, _ = _context_index(context_ids, len(divergence))
    states = candidate_state_metrics(
        divergence, agree, fixed_selection=fixed_selection, candidate_experts=candidate_experts
    )
    metric_names = [name for name in states if name != "oracle_candidate_index"]
    intervals = _bootstrap_columns(
        np.stack([states[name] for name in metric_names], axis=1),
        context_ids,
        bootstrap_repetitions,
        bootstrap_seed,
    )
    ci = dict(zip(metric_names, intervals))
    oracle = states["oracle_candidate_index"]
    row = np.arange(len(divergence))
    coverage = states["cheap_coverage"].astype(bool)
    fixed_coverage = states["fixed_cheap_coverage"].astype(bool)
    if (control_kl is None) != (control_agreement is None):
        raise ValueError("both control_kl and control_agreement must be supplied together")
    control = None
    if control_kl is not None:
        if np.asarray(control_kl).shape != (len(divergence),):
            raise ValueError("numerical control must cover the same evaluated token states")
        control = numerical_control_summary(
            control_kl, control_agreement, context_ids,
            bootstrap_repetitions=bootstrap_repetitions, bootstrap_seed=bootstrap_seed,
        )
    numerical_pass = control is not None and control["point_pass"]
    coverage_pass = ci["cheap_coverage"]["estimate"] >= COVERAGE_GATE
    advantage_pass = ci["portfolio_advantage"]["estimate"] >= PORTFOLIO_ADVANTAGE_GATE
    summary: dict[str, Any] = {
        "split": split,
        "n_states": len(divergence),
        "n_contexts": len(contexts),
        "candidate_count": 28,
        "candidate_schema_sha256": schema,
        "fixed_selection": dict(fixed_selection),
        "local_acceptance": {"same_greedy_token": True, "kl_strictly_less_than": LOCAL_KL_LIMIT},
        "confidence_intervals": ci,
        "cheap_coverage": ci["cheap_coverage"]["estimate"],
        "fixed_cheap_coverage": ci["fixed_cheap_coverage"]["estimate"],
        "portfolio_advantage": ci["portfolio_advantage"]["estimate"],
        "best_fixed": _fidelity(divergence[:, fixed], agree[:, fixed], error[:, fixed]),
        "oracle_cheap_only": {
            **_fidelity(divergence[row, oracle], agree[row, oracle], error[row, oracle]),
            "selection_rule": "greedy_agreement_then_minimum_kl_then_index",
            "candidate_counts": np.bincount(oracle, minlength=28).tolist(),
            "deployable": False,
        },
        "oracle_with_dense": {
            **_fidelity(states["oracle_with_dense_kl"], np.ones_like(coverage),
                        np.where(coverage, error[row, oracle], 0.0)),
            "cheap_fraction": float(coverage.mean()),
            "dense_fraction": float((~coverage).mean()),
            "ideal_active_width": ci["ideal_active_width"]["estimate"],
            "deployable": False,
        },
        "fixed_oracle_with_dense": {
            **_fidelity(states["fixed_with_dense_kl"], np.ones_like(coverage),
                        np.where(fixed_coverage, error[:, fixed], 0.0)),
            "cheap_fraction": float(fixed_coverage.mean()),
            "dense_fraction": float((~fixed_coverage).mean()),
            "ideal_active_width": ci["fixed_ideal_active_width"]["estimate"],
            "deployable": False,
        },
        "numerical_control": control,
        "engineering_gate": {
            "coverage_threshold": COVERAGE_GATE,
            "advantage_threshold": PORTFOLIO_ADVANTAGE_GATE,
            "numerically_interpretable": numerical_pass,
            "coverage_point_pass": coverage_pass,
            "portfolio_advantage_point_pass": advantage_pass,
            "point_pass": numerical_pass and coverage_pass and advantage_pass,
            "coverage_statistically_resolved": ci["cheap_coverage"]["lower"] >= COVERAGE_GATE,
            "advantage_statistically_resolved": ci["portfolio_advantage"]["lower"]
            >= PORTFOLIO_ADVANTAGE_GATE,
            "statistically_resolved_pass": bool(
                numerical_pass and ci["cheap_coverage"]["lower"] >= COVERAGE_GATE
                and ci["portfolio_advantage"]["lower"] >= PORTFOLIO_ADVANTAGE_GATE
            ),
        },
        "cost_interpretation": "ideal_arithmetic_proxy_excludes_routing_and_hardware_overhead",
        "measured_latency": None,
        "measured_dram_bytes": None,
        "oracle_is_deployable": False,
        "per_candidate": [
            {
                "index": index,
                "experts": list(candidate_combinations()[index]),
                "cheap_coverage": float((agree[:, index]
                                         & (divergence[:, index] < LOCAL_KL_LIMIT)).mean()),
                **_fidelity(divergence[:, index], agree[:, index], error[:, index]),
            }
            for index in range(28)
        ],
    }
    # Fixed, preregisterable margin boundaries; these are reporting strata only.
    bins = (0.0, 0.1, 0.5, 1.0, 2.0, float("inf"))
    strata = []
    for lower, upper in itertools.pairwise(bins):
        selected = (margin >= lower) & (margin < upper)
        strata.append({
            "margin_min_inclusive": lower,
            "margin_max_exclusive": upper if np.isfinite(upper) else None,
            "n_states": int(selected.sum()),
            "cheap_coverage": float(coverage[selected].mean()) if selected.any() else None,
            "fixed_cheap_coverage": float(fixed_coverage[selected].mean())
            if selected.any() else None,
        })
    summary["by_dense_margin"] = strata
    return summary


def portfolio_gate(split_summaries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Require the same development-frozen subset and gates on all three splits."""

    required = ("development", "heldout", "ood")
    aliases = {
        "development": (),
        "heldout": ("matched_heldout", "heldout_matched"),
        "ood": ("ood_heldout", "heldout_ood"),
    }
    summaries = {
        split: next(
            (split_summaries[key] for key in (split, *aliases[split]) if key in split_summaries),
            None,
        )
        for split in required
    }
    failures: list[str] = []
    indices: set[tuple[int, str]] = set()
    for split, summary in summaries.items():
        if summary is None:
            failures.append(f"{split}:missing_split")
            continue
        if summary.get("split") not in (split, *aliases[split]):
            raise ValueError("split summary is stored under a different split label")
        selection = summary["fixed_selection"]
        index = _fixed_index(selection, summary["candidate_schema_sha256"])
        indices.add((index, selection["candidate_schema_sha256"]))
        gate = summary["engineering_gate"]
        if not gate["numerically_interpretable"]:
            failures.append(f"{split}:numerical_control_failed_or_missing")
        if not gate["coverage_point_pass"]:
            failures.append(f"{split}:insufficient_cheap_candidate_coverage")
        if not gate["portfolio_advantage_point_pass"]:
            failures.append(f"{split}:insufficient_advantage_over_fixed_subset")
    if len(indices) > 1:
        raise ValueError("all splits must reuse the same development-frozen fixed candidate")
    point_pass = not failures
    resolved = point_pass and all(
        summary["engineering_gate"]["statistically_resolved_pass"]
        for summary in summaries.values() if summary is not None
    )
    return {
        "phase": "capability_audit",
        "point_pass": point_pass,
        "statistically_resolved_pass": resolved,
        "router_training_allowed": point_pass,
        "stop_before_router_training": not point_pass,
        "failure_reasons": failures,
        "oracle_is_deployable": False,
        "economic_savings_established": False,
        "scope": "tested_partitions_candidate_set_data_and_fitting_budget_only",
    }
