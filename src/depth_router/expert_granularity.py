"""Pure CPU partition refinement and teacher-only granularity audit statistics.

All selection in this module is offline: calibration chooses a fixed portfolio,
development chooses its fixed comparator, and held-out matrices only evaluate
those frozen choices. An acceptable-candidate oracle is not a runtime router.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

import numpy as np

from depth_router.expert_audit_metrics import (
    numerical_control_summary,
    paired_context_difference,
)

WIDTH = 4864
PARENTS = 8
CHILDREN = 16
ACTIVE_CHILDREN = 12
PORTFOLIO_SIZE = 28
LOCAL_KL = 0.01


def array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(array.shape).encode())
    digest.update(json.dumps(array.tolist(), sort_keys=True, allow_nan=False).encode()
                  if array.dtype.hasobject else array.tobytes())
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def _integer_vector(value: Any, *, name: str, low: int, high: int) -> np.ndarray:
    array = np.asarray(value)
    if (array.ndim != 1 or not len(array) or array.dtype.kind not in "iu"
            or np.any((array < low) | (array >= high))):
        raise ValueError(f"{name} must be a nonempty integer vector in [{low}, {high})")
    return array.astype(np.int64, copy=False)


def _labels(value: Any, experts: int) -> np.ndarray:
    labels = _integer_vector(value, name="partition", low=0, high=experts)
    if len(labels) != WIDTH or not np.array_equal(
        np.bincount(labels, minlength=experts), np.full(experts, WIDTH // experts)
    ):
        raise ValueError(f"partition must have {WIDTH} neurons and {experts} equal experts")
    return labels


def split_parent_experts(parent_labels: Any, *, seed: int) -> np.ndarray:
    """Randomly split each frozen 608-neuron parent into two exact 304 halves.

    The only randomness is NumPy PCG64 initialized once with ``seed``; parents
    are visited in order 0..7 and each parent's original neuron IDs are sorted
    before one permutation. Child labels are exactly ``2*parent + child``.
    """
    parent = _labels(parent_labels, PARENTS)
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    generator = np.random.default_rng(seed)
    children = np.empty(WIDTH, dtype=np.int64)
    for index in range(PARENTS):
        neuron_ids = generator.permutation(np.flatnonzero(parent == index))
        children[neuron_ids[:304]] = 2 * index
        children[neuron_ids[304:]] = 2 * index + 1
    validate_refinement(parent, children)
    return children


def validate_refinement(parent_labels: Any, child_labels: Any) -> dict[str, Any]:
    parent, children = _labels(parent_labels, PARENTS), _labels(child_labels, CHILDREN)
    if not np.array_equal(children // 2, parent):
        raise ValueError("child experts must refine their frozen parent: child // 2 == parent")
    return {"neurons": WIDTH, "parents": PARENTS, "children": CHILDREN,
            "parent_size": 608, "child_size": 304, "active_neurons": 3648,
            "parent_partition_sha256": array_sha256(parent),
            "partition_sha256": array_sha256(children), "exact_parent_refinement": True}


@lru_cache(maxsize=1)
def candidate_combinations() -> tuple[tuple[int, ...], ...]:
    return tuple(itertools.combinations(range(CHILDREN), ACTIVE_CHILDREN))


@lru_cache(maxsize=1)
def candidate_schema_sha256() -> str:
    return _json_hash(candidate_combinations())


def original_candidate_indices() -> np.ndarray:
    """Map old lexicographic six-of-eight masks into the 1,820-mask universe."""
    positions = {experts: index for index, experts in enumerate(candidate_combinations())}
    return np.array([positions[tuple(child for parent in parents
                                    for child in (2 * parent, 2 * parent + 1))]
                     for parents in itertools.combinations(range(PARENTS), 6)], dtype=np.int64)


def partition_indices(child_labels: Any, selected_experts: Any) -> np.ndarray:
    labels = _labels(child_labels, CHILDREN)
    experts = _integer_vector(selected_experts, name="selected_experts", low=0, high=CHILDREN)
    if len(np.unique(experts)) != len(experts):
        raise ValueError("selected_experts must be unique")
    # Global original-neuron order is shared by gate/up rows and down columns.
    return np.flatnonzero(np.isin(labels, experts)).astype(np.int64, copy=False)


def _candidate_ids(value: Any) -> np.ndarray:
    ids = _integer_vector(value, name="candidate_ids", low=0, high=len(candidate_combinations()))
    if len(np.unique(ids)) != len(ids):
        raise ValueError("candidate_ids must identify unique canonical masks")
    return ids


def _partition_hash(matrix: Mapping[str, Any]) -> str:
    value = str(np.asarray(matrix["partition_sha256"]).item())
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("matrix partition_sha256 must be a lowercase SHA256 hex string")
    return value


def matrix_arrays(matrix: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate complete observations for the declared canonical candidate IDs."""
    kl = np.asarray(matrix["kl"])
    ids = _candidate_ids(matrix["candidate_ids"])
    if (kl.ndim != 2 or not len(kl) or kl.shape[1] != len(ids)
            or kl.dtype.kind != "f" or not np.isfinite(kl).all() or np.any(kl < -1e-5)):
        raise ValueError("KL must be finite [states, candidates], nonnegative within roundoff")
    if "complete" in matrix and not bool(np.asarray(matrix["complete"]).item()):
        raise ValueError("incomplete candidate archives cannot enter selection or completed metrics")
    for field, shape in (("measured", kl.shape), ("control_measured", (len(kl),))):
        if field in matrix:
            observed = np.asarray(matrix[field])
            if observed.shape != shape or observed.dtype.kind != "b" or not observed.all():
                raise ValueError("candidate archives must have every declared measurement completed")
    if "agreement" in matrix:
        agreement = np.asarray(matrix["agreement"])
    else:
        token = np.asarray(matrix["candidate_token"])
        dense = np.asarray(matrix["dense_token"])
        if token.dtype.kind not in "iu" or dense.dtype.kind not in "iu" or dense.shape != (len(kl),):
            raise ValueError("candidate and dense tokens must be integer state-aligned arrays")
        agreement = token == dense[:, None]
    if agreement.shape != kl.shape or not np.isin(agreement, [False, True]).all():
        raise ValueError("agreement must have the same boolean [states, candidates] shape")
    if "selected_experts" in matrix and not np.array_equal(
        matrix["selected_experts"], np.asarray(candidate_combinations())[ids]
    ):
        raise ValueError("selected_experts disagrees with canonical candidate IDs")
    if "partition" in matrix and array_sha256(_labels(matrix["partition"], CHILDREN)) != _partition_hash(matrix):
        raise ValueError("matrix partition hash does not match its neuron labels")
    _partition_hash(matrix)
    return kl, agreement.astype(bool, copy=False), ids


def _contexts(matrix: Mapping[str, Any], context_ids: Any | None, count: int) -> np.ndarray:
    value = matrix.get("context_ids") if context_ids is None else context_ids
    ids = np.asarray(value)
    if ids.shape != (count,) or (ids.dtype.kind in "fc" and not np.isfinite(ids).all()):
        raise ValueError("context_ids must contain one valid context identifier per state")
    if "context_ids" in matrix and not np.array_equal(ids, matrix["context_ids"]):
        raise ValueError("supplied context IDs do not match matrix state order")
    return ids


def _columns(matrix_ids: np.ndarray, portfolio_ids: Any) -> tuple[np.ndarray, np.ndarray]:
    chosen = np.sort(_candidate_ids(portfolio_ids))
    lookup = {int(candidate): column for column, candidate in enumerate(matrix_ids)}
    if any(int(candidate) not in lookup for candidate in chosen):
        raise ValueError("matrix lacks an observation for a requested portfolio candidate")
    return np.array([lookup[int(candidate)] for candidate in chosen]), chosen


def _stage(matrix: Mapping[str, Any], supplied: str, expected: str) -> None:
    if supplied != expected:
        raise ValueError(f"this selection is permitted on {expected} only")
    if "split" in matrix and str(np.asarray(matrix["split"]).item()) != expected:
        raise ValueError("matrix split provenance disagrees with the selection stage")


def _matrix_provenance(matrix: Mapping[str, Any]) -> dict[str, str]:
    kl, agreement, ids = matrix_arrays(matrix)
    contexts = _contexts(matrix, None, len(kl))
    return {"partition_sha256": _partition_hash(matrix),
            "candidate_schema_sha256": candidate_schema_sha256(),
            "matrix_candidate_ids_sha256": array_sha256(ids),
            "matrix_kl_sha256": array_sha256(kl),
            "matrix_agreement_sha256": array_sha256(agreement),
            "matrix_context_ids_sha256": array_sha256(contexts)}


def _seal(selection: dict[str, Any]) -> dict[str, Any]:
    return {**selection, "selection_sha256": _json_hash(selection)}


def verify_frozen_selection(selection: Mapping[str, Any], *, selected_on: str) -> None:
    fields = {key: value for key, value in selection.items() if key != "selection_sha256"}
    if (selection.get("selected_on") != selected_on
            or selection.get("selection_sha256") != _json_hash(fields)):
        raise ValueError("frozen selection was modified or has incorrect stage provenance")
    if selection.get("candidate_schema_sha256") != candidate_schema_sha256():
        raise ValueError("frozen selection uses a different canonical candidate schema")


def select_calibration_portfolio(
    matrix: Mapping[str, Any], *, split: str = "calibration", size: int = PORTFOLIO_SIZE
) -> dict[str, Any]:
    """Greedy marginal acceptable-state coverage; exactly 28 distinct masks.

    Ties use minimum mean KL over all calibration states, then global canonical
    mask index. Saturation does not shorten the budget or change the tie rule.
    """
    _stage(matrix, split, "calibration")
    kl, agreement, ids = matrix_arrays(matrix)
    if size != PORTFOLIO_SIZE or not np.array_equal(np.sort(ids), np.arange(1820)):
        raise ValueError("calibration selection requires all 1820 masks and exactly 28 choices")
    acceptable = agreement & (kl < LOCAL_KL)
    mean_kl = kl.mean(axis=0, dtype=np.float64)
    covered = np.zeros(len(kl), dtype=bool)
    used = np.zeros(len(ids), dtype=bool)
    steps, order = [], []
    for step in range(PORTFOLIO_SIZE):
        marginal = np.count_nonzero(acceptable & ~covered[:, None], axis=0)
        available = np.flatnonzero(~used)
        ranking = np.lexsort((ids[available], mean_kl[available], -marginal[available]))
        column = int(available[ranking[0]])
        used[column] = True
        covered |= acceptable[:, column]
        candidate = int(ids[column])
        order.append(candidate)
        steps.append({"step": step + 1, "candidate_id": candidate,
                      "newly_covered_states": int(marginal[column]),
                      "calibration_mean_kl": float(mean_kl[column]),
                      "cumulative_covered_states": int(covered.sum()),
                      "cumulative_coverage": float(covered.mean())})
    return _seal({
        **_matrix_provenance(matrix), "selected_on": "calibration",
        "candidate_ids": sorted(order), "selection_order": order, "size": PORTFOLIO_SIZE,
        "selection_rule": "greedy_marginal_acceptance_then_mean_kl_then_global_lex_index",
        "calibration_states": len(kl), "calibration_coverage": float(covered.mean()),
        "local_acceptance": {"same_greedy_token": True, "kl_strictly_less_than": LOCAL_KL},
        "steps": steps,
    })


def select_development_fixed(
    matrix: Mapping[str, Any], portfolio_ids: Any, *, split: str = "development"
) -> dict[str, Any]:
    _stage(matrix, split, "development")
    kl, agreement, ids = matrix_arrays(matrix)
    columns, chosen = _columns(ids, portfolio_ids)
    coverage = (agreement[:, columns] & (kl[:, columns] < LOCAL_KL)).mean(axis=0)
    mean_kl = kl[:, columns].mean(axis=0, dtype=np.float64)
    winner = int(np.lexsort((chosen, mean_kl, -coverage))[0])
    return _seal({
        **_matrix_provenance(matrix), "selected_on": "development",
        "candidate_id": int(chosen[winner]), "portfolio_candidate_ids": chosen.tolist(),
        "portfolio_ids_sha256": array_sha256(chosen),
        "experts": list(candidate_combinations()[chosen[winner]]),
        "selection_rule": "maximum_acceptance_then_mean_kl_then_global_lex_index",
        "development_states": len(kl), "development_coverage": float(coverage[winner]),
        "development_mean_kl": float(mean_kl[winner]),
    })


def _verify_fixed(
    matrix: Mapping[str, Any], portfolio_ids: np.ndarray, fixed: Mapping[str, Any]
) -> int:
    verify_frozen_selection(fixed, selected_on="development")
    if (fixed["partition_sha256"] != _partition_hash(matrix)
            or fixed["portfolio_ids_sha256"] != array_sha256(portfolio_ids)
            or fixed["portfolio_candidate_ids"] != portfolio_ids.tolist()):
        raise ValueError("fixed comparator belongs to a different partition or frozen portfolio")
    candidate = fixed["candidate_id"]
    if candidate not in portfolio_ids or fixed["experts"] != list(candidate_combinations()[candidate]):
        raise ValueError("fixed comparator is not the stated canonical portfolio member")
    return int(candidate)


def portfolio_coverage(matrix: Mapping[str, Any], portfolio_ids: Any) -> np.ndarray:
    kl, agreement, ids = matrix_arrays(matrix)
    columns, _ = _columns(ids, portfolio_ids)
    return (agreement[:, columns] & (kl[:, columns] < LOCAL_KL)).any(axis=1)


def portfolio_state_metrics(
    matrix: Mapping[str, Any], portfolio_ids: Any, fixed_selection: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    kl, agreement, ids = matrix_arrays(matrix)
    columns, chosen = _columns(ids, portfolio_ids)
    fixed = _verify_fixed(matrix, chosen, fixed_selection)
    fixed_column = int(np.flatnonzero(ids == fixed)[0])
    sub_kl, sub_agreement = kl[:, columns], agreement[:, columns]
    covered = (sub_agreement & (sub_kl < LOCAL_KL)).any(axis=1)
    fixed_covered = agreement[:, fixed_column] & (kl[:, fixed_column] < LOCAL_KL)
    # Sorted global IDs make argmin's first-index tie canonical, regardless of
    # the order in which the caller stored columns in the raw matrix.
    same = np.where(sub_agreement, sub_kl, np.inf).argmin(axis=1)
    oracle = np.where(sub_agreement.any(axis=1), same, sub_kl.argmin(axis=1))
    row = np.arange(len(kl))
    oracle_kl = sub_kl[row, oracle]
    return {
        "cheap_coverage": covered.astype(float),
        "fixed_cheap_coverage": fixed_covered.astype(float),
        "portfolio_advantage": 0.25 * (covered.astype(float) - fixed_covered),
        "ideal_active_width": 1.0 - 0.25 * covered,
        "fixed_ideal_active_width": 1.0 - 0.25 * fixed_covered,
        "fixed_agreement": agreement[:, fixed_column].astype(float),
        "fixed_kl": kl[:, fixed_column],
        "oracle_agreement": sub_agreement[row, oracle].astype(float),
        "oracle_kl": oracle_kl,
        "oracle_with_dense_kl": np.where(covered, oracle_kl, 0.0),
        "fixed_with_dense_kl": np.where(fixed_covered, kl[:, fixed_column], 0.0),
        "oracle_candidate_id": chosen[oracle],
    }


def summarize_portfolio(
    matrix: Mapping[str, Any], portfolio_ids: Any, fixed_selection: Mapping[str, Any],
    context_ids: Any | None = None, *, split: str,
    bootstrap_repetitions: int = 2000, bootstrap_seed: int = 17,
) -> dict[str, Any]:
    if "split" in matrix and str(np.asarray(matrix["split"]).item()) != split:
        raise ValueError("matrix split provenance disagrees with the evaluation split")
    state = portfolio_state_metrics(matrix, portfolio_ids, fixed_selection)
    n = len(state["cheap_coverage"])
    contexts = _contexts(matrix, context_ids, n)
    chosen = np.sort(_candidate_ids(portfolio_ids))
    metrics = {key: value for key, value in state.items() if key != "oracle_candidate_id"}
    intervals = {key: paired_context_difference(
        values, np.zeros(n), contexts, bootstrap_repetitions=bootstrap_repetitions,
        bootstrap_seed=bootstrap_seed) for key, values in metrics.items()}
    means = {key: float(values.mean(dtype=np.float64)) for key, values in metrics.items()}
    if "control_kl" in matrix and ("control_agreement" in matrix or "control_token" in matrix):
        agreement = (matrix["control_agreement"] if "control_agreement" in matrix else
                     np.asarray(matrix["control_token"]) == np.asarray(matrix["dense_token"]))
        numerical = numerical_control_summary(
            matrix["control_kl"], agreement, contexts,
            bootstrap_repetitions=bootstrap_repetitions, bootstrap_seed=bootstrap_seed)
    else:
        numerical = {"point_pass": False, "statistically_resolved_pass": False,
                     "reason": "missing_actual_layout_all_expert_numerical_control"}
    coverage_pass = means["cheap_coverage"] >= 0.8
    advantage_pass = means["portfolio_advantage"] >= 0.05
    resolved_coverage = intervals["cheap_coverage"]["lower"] >= 0.8
    resolved_advantage = intervals["portfolio_advantage"]["lower"] >= 0.05
    return {
        **means, "split": split, "n_states": n, "n_contexts": len(np.unique(contexts)),
        "candidate_ids": chosen.tolist(), "candidate_count": len(chosen),
        "partition_sha256": _partition_hash(matrix),
        "candidate_schema_sha256": candidate_schema_sha256(),
        "fixed_selection": dict(fixed_selection), "confidence_intervals": intervals,
        "numerical_control": numerical,
        "engineering_gate": {
            "coverage_threshold": 0.8, "advantage_threshold": 0.05,
            "coverage_point_pass": coverage_pass, "portfolio_advantage_point_pass": advantage_pass,
            "coverage_statistically_resolved": resolved_coverage,
            "advantage_statistically_resolved": resolved_advantage,
            "numerically_interpretable": numerical["point_pass"],
            "point_pass": coverage_pass and advantage_pass and numerical["point_pass"],
            "statistically_resolved_pass": resolved_coverage and resolved_advantage
                and numerical["statistically_resolved_pass"],
        },
        "oracle_cheap_only": {
            "greedy_agreement": means["oracle_agreement"], "mean_kl": means["oracle_kl"],
            "p95_kl": float(np.quantile(state["oracle_kl"], 0.95)), "deployable": False},
        "oracle_with_dense": {
            "greedy_agreement": 1.0, "mean_kl": means["oracle_with_dense_kl"],
            "cheap_fraction": means["cheap_coverage"],
            "ideal_active_width": means["ideal_active_width"], "deployable": False},
        "fixed_oracle_with_dense": {
            "greedy_agreement": 1.0, "mean_kl": means["fixed_with_dense_kl"],
            "cheap_fraction": means["fixed_cheap_coverage"],
            "ideal_active_width": means["fixed_ideal_active_width"], "deployable": False},
        "local_acceptance": {"same_greedy_token": True, "kl_strictly_less_than": LOCAL_KL},
        "oracle_is_deployable": False,
        "cost_interpretation": "ideal_width_proxy_excludes_all_routing_and_hardware_overhead",
        "measured_latency": None, "measured_dram_bytes": None,
    }


def paired_coverage_difference(
    selected_coverage: Any, original_coverage: Any, context_ids: Any, *,
    bootstrap_repetitions: int = 2000, bootstrap_seed: int = 17,
) -> dict[str, Any]:
    """Rescue, loss and net gain, optionally seed-mean before context bootstrap."""
    selected, original = np.asarray(selected_coverage), np.asarray(original_coverage)
    if (selected.shape != original.shape or selected.ndim not in (1, 2) or not selected.size
            or not np.isin(selected, [False, True]).all()
            or not np.isin(original, [False, True]).all()):
        raise ValueError("paired coverage must share boolean [states] or [seeds, states] shape")
    selected, original = selected.astype(bool), original.astype(bool)
    rescue = selected & ~original
    loss = ~selected & original
    net = selected.astype(float) - original
    values = {"rescue": rescue.astype(float), "loss": loss.astype(float),
              "net_coverage": net, "ideal_width_improvement": 0.25 * net}
    results = {key: paired_context_difference(
        value, np.zeros_like(value), context_ids,
        bootstrap_repetitions=bootstrap_repetitions, bootstrap_seed=bootstrap_seed)
        for key, value in values.items()}
    results["state_counts"] = {
        "rescue": np.count_nonzero(rescue, axis=-1).tolist(),
        "loss": np.count_nonzero(loss, axis=-1).tolist(),
        "net": net.sum(axis=-1).astype(int).tolist(),
        "states_per_seed": selected.shape[-1],
    }
    results["direction"] = "selected_portfolio_minus_original_portfolio"
    results["oracle_not_runtime"] = True
    return results


def paired_portfolio_difference(
    left_matrix: Mapping[str, Any], left_ids: Any,
    right_matrix: Mapping[str, Any], right_ids: Any, context_ids: Any | None = None,
    *, bootstrap_repetitions: int = 2000, bootstrap_seed: int = 17,
) -> dict[str, Any]:
    left, right = portfolio_coverage(left_matrix, left_ids), portfolio_coverage(right_matrix, right_ids)
    contexts = _contexts(left_matrix, context_ids, len(left))
    _contexts(right_matrix, contexts, len(right))
    for field in ("positions", "dense_token"):
        if field in left_matrix and field in right_matrix and not np.array_equal(
            left_matrix[field], right_matrix[field]
        ):
            raise ValueError(f"paired matrices disagree on reference state {field}")
    return paired_coverage_difference(left, right, contexts,
        bootstrap_repetitions=bootstrap_repetitions, bootstrap_seed=bootstrap_seed)


def cycle_gate(
    selected_summaries: Mapping[int | str, Mapping[str, Mapping[str, Any]]],
    aggregate_ood_difference: Mapping[str, Any], *,
    required_seeds: Sequence[int] = (17, 29, 43),
    required_splits: Sequence[str] = ("development", "heldout_matched", "heldout_ood"),
) -> dict[str, Any]:
    normalized = {str(key): value for key, value in selected_summaries.items()}
    if len(normalized) != len(selected_summaries) or set(normalized) != set(map(str, required_seeds)):
        raise ValueError("cycle gate requires exactly all three preregistered refinement seeds")
    individual = {}
    for seed in required_seeds:
        summaries = normalized[str(seed)]
        if set(summaries) != set(required_splits):
            raise ValueError("cycle gate requires exactly development and both held-out summaries")
        if any(summaries[split]["candidate_count"] != PORTFOLIO_SIZE for split in required_splits):
            raise ValueError("cycle promotion is for the selected 28-mask portfolios only")
        individual[str(seed)] = {
            "point_pass": all(summaries[split]["engineering_gate"]["point_pass"]
                              for split in required_splits),
            "statistically_resolved_pass": all(
                summaries[split]["engineering_gate"]["statistically_resolved_pass"]
                for split in required_splits),
        }
    interval = aggregate_ood_difference["net_coverage"]
    if (interval.get("n_paired_seeds") != len(required_seeds)
            or not np.isfinite([interval["estimate"], interval["lower"], interval["upper"]]).all()):
        raise ValueError("OOD gain must be the finite paired interval over all preregistered seeds")
    all_point = all(value["point_pass"] for value in individual.values())
    gain_pass = interval["lower"] > 0
    return {
        "individual_selected_portfolios": individual,
        "all_seed_all_split_point_pass": all_point,
        "all_seed_all_split_statistically_resolved_pass": all(
            value["statistically_resolved_pass"] for value in individual.values()),
        "aggregate_ood_net_coverage_ci": dict(interval),
        "aggregate_ood_resolved_positive_gain": gain_pass,
        "point_pass": all_point and gain_pass,
        "promotion_scope": "frozen_28_mask_capability_not_router_or_measured_economics",
        "measured_latency": None, "measured_dram_bytes": None,
    }
