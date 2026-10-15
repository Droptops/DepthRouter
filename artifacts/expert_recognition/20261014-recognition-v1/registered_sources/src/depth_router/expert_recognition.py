"""Small causal selectors and offline diagnostics for fixed expert recognition.

Only ``Selector`` and ``actions_from_scores`` are deployment components. The
matrix-based metrics, labels and oracles require teacher observations and must
never be called by a deployed selector. Dense action 28 is the original MLP.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from depth_router.expert_audit_metrics import paired_context_difference

DENSE_ACTION = 28
ARCHITECTURES = ("linear", "mlp_64")
DEFAULT_THRESHOLDS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 1.0)


def array_sha256(array: Any) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(value.shape).encode())
    # Object-array bytes contain memory addresses, not reproducible values.
    digest.update(json.dumps(value.tolist(), sort_keys=True, allow_nan=False).encode()
                  if value.dtype.hasobject else value.tobytes())
    return digest.hexdigest()


def state_sha256(model: nn.Module) -> str:
    """Hash names, dtypes, shapes and exact parameter/buffer contents."""
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        digest.update(key.encode())
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _features(features: Any) -> torch.Tensor:
    x = torch.as_tensor(features).detach().to(device="cpu", dtype=torch.float32)
    if x.ndim != 2 or min(x.shape) == 0 or not torch.isfinite(x).all():
        raise ValueError("features must be finite nonempty [states, hidden_width]")
    return x


class Selector(nn.Module):
    """FP32 x-only model; normalization is fixed from calibration, not test data."""

    def __init__(
        self, mean: torch.Tensor, scale: torch.Tensor, architecture: str, outputs: int
    ) -> None:
        super().__init__()
        if architecture not in ARCHITECTURES or outputs not in (1, 28):
            raise ValueError("architecture must be linear/mlp_64 and outputs must be 1/28")
        mean = torch.as_tensor(mean).detach().to(dtype=torch.float32, device="cpu")
        scale = torch.as_tensor(scale).detach().to(dtype=torch.float32, device="cpu")
        if (mean.ndim != 1 or scale.shape != mean.shape or not len(mean)
                or not torch.isfinite(mean).all() or not torch.isfinite(scale).all()
                or torch.any(scale <= 0)):
            raise ValueError("normalization must have finite equal-sized vectors and positive scale")
        self.architecture = architecture
        self.outputs = outputs
        self.register_buffer("mean", mean.clone())
        self.register_buffer("scale", scale.clone())
        self.backbone = (nn.Identity() if architecture == "linear"
                         else nn.Sequential(nn.Linear(len(mean), 64), nn.SiLU()))
        self.head = nn.Linear(len(mean) if architecture == "linear" else 64, outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # This float cast and normalization are part of the runtime path/cost.
        return self.head(self.backbone((x.float() - self.mean) / self.scale))

    def scores(self, x: torch.Tensor) -> torch.Tensor:
        return self(x).sigmoid()

    def actions(
        self, x: torch.Tensor, threshold: float, *, fixed_index: int | None = None
    ) -> torch.Tensor:
        _validate_threshold(threshold)
        _validate_fixed_index(fixed_index, self.outputs)
        if threshold == 1.0:
            return torch.full(x.shape[:-1], DENSE_ACTION, dtype=torch.long, device=x.device)
        scores = self.scores(x)
        confidence, candidate = scores.max(dim=-1)
        if fixed_index is not None:
            candidate = torch.full_like(candidate, fixed_index)
        return torch.where(confidence >= threshold, candidate, DENSE_ACTION)


def _validate_threshold(threshold: float) -> None:
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and in [0, 1]")


def _validate_fixed_index(index: int | None, outputs: int) -> None:
    if outputs == 1:
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < 28:
            raise ValueError("binary scores require fixed_index in [0, 27]")
    elif index is not None:
        raise ValueError("portfolio scores must not be supplied a fixed_index")


def matrix_arrays(matrix: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    kl = np.asarray(matrix["kl"], dtype=np.float64)
    agreement = (np.asarray(matrix["agreement"]) if "agreement" in matrix else
                 np.asarray(matrix["candidate_token"]) == np.asarray(matrix["dense_token"])[:, None])
    if kl.ndim != 2 or kl.shape[1] != 28 or not len(kl) or agreement.shape != kl.shape:
        raise ValueError("matrix must contain nonempty complete [states, 28] candidate results")
    if not np.isfinite(kl).all() or np.any(kl < -1e-5):
        raise ValueError("KL must be finite and nonnegative within 1e-5 roundoff tolerance")
    if not np.isin(agreement, [False, True]).all():
        raise ValueError("agreement must be boolean")
    return kl, agreement.astype(bool)


def acceptable_labels(matrix: Mapping[str, Any]) -> np.ndarray:
    kl, agreement = matrix_arrays(matrix)
    return agreement & (kl < 0.01)


def train_selector_pair(
    calibration_features: Any,
    calibration_labels: Any,
    fixed_index: int,
    *,
    architecture: str,
    seed: int,
    steps: int = 200,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    device: str | torch.device = "cpu",
    step_callback: Callable[[int], None] | None = None,
) -> tuple[Selector, Selector, dict[str, Any]]:
    """Fit portfolio/binary with identical calibration normalization and schedule.

    Shared hidden weights are initially identical. The binary head starts from
    the portfolio's fixed-candidate head row. Parameter counts differ by their
    unavoidable output heads, recorded explicitly. No development data is used.
    """
    x = _features(calibration_features)
    y = torch.as_tensor(calibration_labels).detach().to(device="cpu")
    if y.shape != (len(x), 28) or not ((y == 0) | (y == 1)).all():
        raise ValueError("calibration_labels must be boolean [states, 28]")
    _validate_fixed_index(fixed_index, 1)
    if steps < 1 or batch_size < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("steps, batch_size and learning_rate must be positive")
    mean = x.mean(0)
    scale = x.std(0, unbiased=False).clamp_min(1e-5)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        portfolio = Selector(mean, scale, architecture, 28)
        binary = Selector(mean, scale, architecture, 1)
        binary.backbone.load_state_dict(portfolio.backbone.state_dict())
        with torch.no_grad():
            binary.head.weight.copy_(portfolio.head.weight[fixed_index:fixed_index + 1])
            binary.head.bias.copy_(portfolio.head.bias[fixed_index:fixed_index + 1])
    initial = {"portfolio": state_sha256(portfolio), "binary": state_sha256(binary)}
    shared_backbone_hash = state_sha256(portfolio.backbone)
    assert shared_backbone_hash == state_sha256(binary.backbone)
    schedule = np.random.default_rng(seed).integers(0, len(x), (steps, batch_size), dtype=np.int64)
    x = x.to(device)
    y = y.to(device=device, dtype=torch.float32)
    portfolio = portfolio.to(device)
    binary = binary.to(device)
    models = {"portfolio": portfolio, "binary": binary}
    optimizers = {name: torch.optim.Adam(model.parameters(), lr=learning_rate)
                  for name, model in models.items()}
    losses: dict[str, list[float]] = {name: [] for name in models}
    for step, row_ids in enumerate(schedule):
        if step_callback is not None:
            step_callback(step)
        ids = torch.as_tensor(row_ids, device=device)
        for name, model in models.items():
            model.train()
            optimizer = optimizers[name]
            optimizer.zero_grad(set_to_none=True)
            target = y[ids] if name == "portfolio" else y[ids, fixed_index:fixed_index + 1]
            loss = F.binary_cross_entropy_with_logits(model(x[ids]), target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite selector loss: {name} step {step}")
            loss.backward()
            optimizer.step()
            losses[name].append(float(loss.detach().cpu()))
    for model in models.values():
        model.eval()
    metadata = {
        "architecture": architecture, "seed": seed, "steps": steps,
        "batch_size": batch_size, "learning_rate": learning_rate, "optimizer": "Adam",
        "loss": "mean_multilabel_BCE_portfolio_and_binary_BCE_fixed",
        "selected_checkpoint": "final_iterate_no_early_stopping",
        "fixed_index": fixed_index, "calibration_states": len(x),
        "calibration_features_sha256": array_sha256(x.cpu().numpy()),
        "calibration_labels_sha256": array_sha256(y.cpu().numpy()),
        "shared_schedule_sha256": array_sha256(schedule),
        "initial_state_sha256": initial,
        "shared_initial_backbone_sha256": shared_backbone_hash,
        "normalization_sha256": array_sha256(torch.stack((mean, scale)).numpy()),
        "normalization_source": "calibration_only_population_std_floor_1e-5",
        "runtime_dtype": "float32_including_normalization",
        "parameter_counts": {name: sum(p.numel() for p in model.parameters())
                             for name, model in models.items()},
        "capacity_matching": "identical_backbone_width; output_heads_28_vs_1_not_exact_capacity",
        "final_state_sha256": {name: state_sha256(model) for name, model in models.items()},
        "losses": losses,
    }
    return portfolio, binary, metadata


def checkpoint_payload(model: Selector, metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "format_version": 1, "architecture": model.architecture, "outputs": model.outputs,
        "state_dict": {key: value.detach().cpu().clone()
                       for key, value in model.state_dict().items()},
        "state_sha256": state_sha256(model), "metadata": dict(metadata),
    }


def save_checkpoint(path: str | Path, model: Selector, metadata: Mapping[str, Any]) -> str:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite frozen checkpoint: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(model, metadata), destination)
    return hashlib.sha256(destination.read_bytes()).hexdigest()


def load_checkpoint(path: str | Path, *, device: str | torch.device = "cpu") -> Selector:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload["state_dict"]
    model = Selector(state["mean"], state["scale"], payload["architecture"], payload["outputs"])
    model.load_state_dict(state)
    if state_sha256(model) != payload["state_sha256"]:
        raise ValueError("checkpoint state hash does not match frozen checkpoint")
    return model.to(device).eval()


def predict_scores(model: Selector, features: Any, *, batch_size: int = 256) -> np.ndarray:
    x = _features(features)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    device = model.mean.device
    model.eval()
    with torch.no_grad():
        return torch.cat([model.scores(x[start:start + batch_size].to(device)).cpu()
                          for start in range(0, len(x), batch_size)]).numpy()


def actions_from_scores(
    scores: Any, threshold: float, *, fixed_index: int | None = None
) -> np.ndarray:
    probabilities = np.asarray(scores, dtype=np.float64)
    if (probabilities.ndim != 2 or not len(probabilities) or probabilities.shape[1] not in (1, 28)
            or not np.isfinite(probabilities).all()
            or np.any((probabilities < 0) | (probabilities > 1))):
        raise ValueError("scores must be finite probabilities [states, 1 or 28]")
    _validate_threshold(threshold)
    _validate_fixed_index(fixed_index, probabilities.shape[1])
    if threshold == 1.0:
        return np.full(len(probabilities), DENSE_ACTION, dtype=np.int64)
    candidate = probabilities.argmax(axis=1)
    if fixed_index is not None:
        candidate.fill(fixed_index)
    return np.where(probabilities.max(axis=1) >= threshold, candidate, DENSE_ACTION)


def policy_state_metrics(actions: Any, matrix: Mapping[str, Any]) -> dict[str, np.ndarray]:
    kl, agreement = matrix_arrays(matrix)
    chosen = np.asarray(actions)
    if (chosen.shape != (len(kl),) or chosen.dtype.kind not in "iu"
            or np.any((chosen < 0) | (chosen > DENSE_ACTION))):
        raise ValueError("actions must be integer [states] values 0..28 (28 is dense)")
    cheap = chosen != DENSE_ACTION
    selected = np.minimum(chosen, 27)
    rows = np.arange(len(kl))
    actual_kl = np.where(cheap, kl[rows, selected], 0.0)
    actual_agreement = np.where(cheap, agreement[rows, selected], True)
    false_cheap = cheap & (~actual_agreement | (actual_kl >= 0.01))
    return {
        "greedy_agreement": actual_agreement.astype(float), "mean_kl": actual_kl,
        "cheap_fraction": cheap.astype(float), "dense_fraction": (~cheap).astype(float),
        "false_cheap_fraction": false_cheap.astype(float),
        "ideal_active_width": 1.0 - 0.25 * cheap,
    }


def _conditional_false_cheap_interval(
    false_cheap: np.ndarray, cheap: np.ndarray, context_ids: np.ndarray,
    repetitions: int, seed: int,
) -> dict[str, Any]:
    _, inverse = np.unique(context_ids, return_inverse=True)
    numerator = np.bincount(inverse, weights=false_cheap)
    denominator = np.bincount(inverse, weights=cheap)
    rng = np.random.default_rng(seed)
    ratios = []
    undefined = 0
    for start in range(0, repetitions, 128):
        count = min(128, repetitions - start)
        indices = rng.integers(0, len(numerator), size=(count, len(numerator)))
        num = numerator[indices].sum(1)
        den = denominator[indices].sum(1)
        undefined += int(np.count_nonzero(den == 0))
        ratios.extend((num[den > 0] / den[den > 0]).tolist())
    return {
        "lower": float(np.quantile(ratios, 0.025)) if ratios else None,
        "upper": float(np.quantile(ratios, 0.975)) if ratios else None,
        "undefined_zero_cheap_resamples": undefined, "resamples": repetitions,
        "method": "percentile_context_bootstrap_ratio_excluding_zero_cheap_resamples",
    }


def policy_metrics(
    actions: Any,
    matrix: Mapping[str, Any],
    context_ids: Any,
    *,
    bootstrap_repetitions: int = 2000,
    bootstrap_seed: int = 17,
) -> dict[str, Any]:
    state = policy_state_metrics(actions, matrix)
    ids = np.asarray(context_ids)
    n = len(state["mean_kl"])
    if ids.shape != (n,) or bootstrap_repetitions < 0:
        raise ValueError("context_ids must match states and bootstrap_repetitions must be >=0")
    means = {key: float(value.mean()) for key, value in state.items()}
    cheap_count = int(state["cheap_fraction"].sum())
    false_count = int(state["false_cheap_fraction"].sum())
    intervals = {}
    if bootstrap_repetitions:
        intervals = {
            key: paired_context_difference(
                value, np.zeros(n), ids, bootstrap_repetitions=bootstrap_repetitions,
                bootstrap_seed=bootstrap_seed)
            for key, value in state.items()
        }
        intervals["false_cheap_among_cheap"] = _conditional_false_cheap_interval(
            state["false_cheap_fraction"], state["cheap_fraction"], ids,
            bootstrap_repetitions, bootstrap_seed)
    point_pass = means["greedy_agreement"] >= 0.99 and means["mean_kl"] < 0.01
    resolved = (intervals["greedy_agreement"]["lower"] >= 0.99
                and intervals["mean_kl"]["upper"] < 0.01) if intervals else None
    return {
        **means, "n_states": n, "n_contexts": len(np.unique(ids)),
        "cheap_count": cheap_count, "false_cheap_count": false_count,
        "false_cheap_among_cheap": false_count / cheap_count if cheap_count else None,
        "kl_p95": float(np.quantile(state["mean_kl"], 0.95)),
        "kl_p99": float(np.quantile(state["mean_kl"], 0.99)),
        "kl_max": float(state["mean_kl"].max()), "intervals": intervals,
        "quality_gate": {"point_pass": point_pass, "statistically_resolved_pass": resolved,
                         "agreement_at_least": 0.99, "mean_kl_strictly_below": 0.01},
        "has_nonzero_cheap_use": cheap_count > 0,
        "confidence_is_certificate": False,
        "ideal_width_is_measured_savings": False,
        "measured_latency": None, "measured_dram_bytes": None,
    }


def select_policy(
    scores_by_architecture: Mapping[str, Any],
    development_matrix: Mapping[str, Any],
    context_ids: Any,
    *,
    fixed_index: int | None = None,
    split: str = "development",
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    """Choose only on fresh development; retain every architecture/threshold row."""
    if split != "development":
        raise ValueError("policy selection is permitted on development only")
    if set(scores_by_architecture) != set(ARCHITECTURES):
        raise ValueError("selection requires the preregistered linear and mlp_64 architectures")
    if tuple(thresholds) != DEFAULT_THRESHOLDS:
        raise ValueError("selection requires the complete preregistered threshold grid")
    kl, agreement = matrix_arrays(development_matrix)
    grid = []
    for architecture in ARCHITECTURES:
        for threshold in thresholds:
            actions = actions_from_scores(
                scores_by_architecture[architecture], threshold, fixed_index=fixed_index)
            metrics = policy_metrics(actions, development_matrix, context_ids,
                                     bootstrap_repetitions=0)
            grid.append({"architecture": architecture, "threshold": float(threshold),
                         "metrics": metrics, "actions_sha256": array_sha256(actions)})
    passing = [row for row in grid if row["metrics"]["quality_gate"]["point_pass"]]
    selected = min(passing, key=lambda row: (
        -row["metrics"]["cheap_fraction"], ARCHITECTURES.index(row["architecture"]),
        -row["threshold"]))
    choice = {
        "selected_on": "development", "architecture": selected["architecture"],
        "threshold": selected["threshold"], "fixed_index": fixed_index,
        "selection_rule": "max_cheap_given_quality_then_smaller_architecture_then_higher_threshold",
        "development_kl_sha256": array_sha256(kl),
        "development_agreement_sha256": array_sha256(agreement),
        "development_contexts_sha256": array_sha256(np.asarray(context_ids)),
        "development_scores_sha256": {
            name: array_sha256(scores) for name, scores in scores_by_architecture.items()},
        "development_actions_sha256": selected["actions_sha256"],
        "development_metrics": selected["metrics"],
    }
    choice["choice_sha256"] = hashlib.sha256(
        json.dumps(choice, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return {**choice, "grid": grid}


def verify_frozen_choice(choice: Mapping[str, Any]) -> None:
    fields = {key: value for key, value in choice.items() if key not in ("choice_sha256", "grid")}
    actual = hashlib.sha256(
        json.dumps(fields, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    if choice.get("selected_on") != "development" or actual != choice.get("choice_sha256"):
        raise ValueError("frozen development choice was modified or has invalid provenance")


def paired_policy_difference(
    left_actions: Any, right_actions: Any, matrix: Mapping[str, Any], context_ids: Any,
    *, bootstrap_repetitions: int = 2000, bootstrap_seed: int = 17,
) -> dict[str, Any]:
    """Paired policy contrasts, optionally averaging preregistered seeds first.

    Actions may be [states] or [seeds, states]. The latter retains all seeds in
    each whole-context bootstrap draw; it does not claim new-seed uncertainty.
    """
    lhs, rhs = np.asarray(left_actions), np.asarray(right_actions)
    if lhs.shape != rhs.shape or lhs.ndim not in (1, 2) or not lhs.size:
        raise ValueError("paired actions must share nonempty [states] or [seeds, states] shape")
    if lhs.ndim == 1:
        left = policy_state_metrics(lhs, matrix)
        right = policy_state_metrics(rhs, matrix)
    else:
        left_rows = [policy_state_metrics(row, matrix) for row in lhs]
        right_rows = [policy_state_metrics(row, matrix) for row in rhs]
        left = {key: np.stack([row[key] for row in left_rows]) for key in left_rows[0]}
        right = {key: np.stack([row[key] for row in right_rows]) for key in right_rows[0]}
    return {key: paired_context_difference(
        left[key], right[key], context_ids, bootstrap_repetitions=bootstrap_repetitions,
        bootstrap_seed=bootstrap_seed) for key in left}


def random_actions(
    reference_actions: Any, *, seed: int, match_positions: bool = True
) -> np.ndarray:
    """Teacher-free random subset diagnostic with exactly matched cheap count.

    By default retain the learned cheap/dense positions, isolating candidate
    recognition. ``match_positions=False`` also randomizes the cheap positions.
    Neither variant is used to tune or choose the learned policy.
    """
    reference = np.asarray(reference_actions)
    if (reference.ndim != 1 or not len(reference) or reference.dtype.kind not in "iu"
            or np.any((reference < 0) | (reference > DENSE_ACTION))):
        raise ValueError("reference_actions must be integer actions 0..28")
    rng = np.random.default_rng(seed)
    cheap = reference != DENSE_ACTION
    indices = np.flatnonzero(cheap) if match_positions else rng.choice(
        len(reference), size=int(cheap.sum()), replace=False)
    actions = np.full(len(reference), DENSE_ACTION, dtype=np.int64)
    actions[indices] = rng.integers(0, 28, size=len(indices))
    return actions


def oracle_actions(matrix: Mapping[str, Any], *, fixed_index: int | None = None) -> np.ndarray:
    """Nondeployable teacher oracle; choose acceptable minimum-KL subset or dense."""
    kl, _ = matrix_arrays(matrix)
    labels = acceptable_labels(matrix)
    if fixed_index is not None:
        _validate_fixed_index(fixed_index, 1)
        return np.where(labels[:, fixed_index], fixed_index, DENSE_ACTION)
    choice = np.where(labels, kl, np.inf).argmin(axis=1)
    return np.where(labels.any(axis=1), choice, DENSE_ACTION)


def repeatability_gate(
    seed_split_metrics: Mapping[int, Mapping[str, Mapping[str, Any]]],
    *, required_seeds: Sequence[int] = (17, 29, 43),
    required_splits: Sequence[str] = ("development", "heldout_matched", "heldout_ood"),
) -> dict[str, Any]:
    if set(seed_split_metrics) != set(required_seeds):
        raise ValueError("repeatability requires every preregistered seed")
    individual = {}
    for seed in required_seeds:
        splits = seed_split_metrics[seed]
        if not set(required_splits).issubset(splits):
            raise ValueError("repeatability requires every preregistered evaluation split")
        individual[str(seed)] = {
            "quality_point_pass": all(splits[name]["quality_gate"]["point_pass"]
                                      for name in required_splits),
            "quality_statistically_resolved_pass": all(
                splits[name]["quality_gate"]["statistically_resolved_pass"] is True
                for name in required_splits),
            "nonzero_cheap_use_all_splits": all(splits[name]["has_nonzero_cheap_use"]
                                                for name in required_splits),
        }
    return {
        "individual_seeds": individual,
        "all_seed_quality_point_pass": all(v["quality_point_pass"] for v in individual.values()),
        "all_seed_quality_statistically_resolved_pass": all(
            v["quality_statistically_resolved_pass"] for v in individual.values()),
        "all_seed_nonzero_cheap_use": all(v["nonzero_cheap_use_all_splits"]
                                         for v in individual.values()),
        "all_seed_quality_and_nonzero_point_pass": all(
            v["quality_point_pass"] and v["nonzero_cheap_use_all_splits"]
            for v in individual.values()),
        "all_dense_is_not_recognition_success": True,
    }
