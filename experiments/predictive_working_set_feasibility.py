"""Preregistered causal working-set feasibility test for Qwen2.5-0.5B.

The held-out split is not captured until predictor rank and feature form have
been selected on development-set oracle recovery.  Dense CUDA execution is the
reference.  Oracle selection is reported only as a non-deployable upper bound.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import platform
import random
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from depth_router.photonic_parallel import contribution_scores, swiglu_subset
from depth_router.predictive_working_set import (
    LowRankWorkingSetPredictor,
    aggregate_page_scores,
    contiguous_operator_pages,
    expand_page_selections,
    make_evaluation_manifest,
    predictor_footprint,
    row_set_metrics,
    topk_rows,
    validate_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/predictive_working_set_full.json")
    parser.add_argument("--output-dir", default="artifacts/predictive_working_set/manual")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class RunLog:
    def __init__(self, path: Path) -> None:
        self.path = path

    def write(self, event: str, **fields: Any) -> None:
        row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "event": event, **fields}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def command_output(command: list[str]) -> str:
    try:
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unavailable: {exc}"


def environment_snapshot() -> dict[str, Any]:
    import transformers

    properties = torch.cuda.get_device_properties(0)
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": properties.name,
        "gpu_total_memory_bytes": properties.total_memory,
        "gpu_compute_capability": [properties.major, properties.minor],
        "nvidia_smi": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,driver_version,temperature.gpu",
                "--format=csv,noheader,nounits",
            ]
        ),
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_status": command_output(["git", "status", "--short"]),
    }


def resolve_layers(model: nn.Module) -> list[nn.Module]:
    candidates = [
        getattr(getattr(model, "model", None), "layers", None),
        getattr(model, "layers", None),
        getattr(getattr(model, "transformer", None), "h", None),
    ]
    for candidate in candidates:
        if candidate is not None and len(candidate):
            return list(candidate)
    raise ValueError("could not resolve decoder layers")


def resolve_swiglu(layer: nn.Module) -> tuple[nn.Module, nn.Linear, nn.Linear, nn.Linear]:
    mlp = getattr(layer, "mlp", None)
    gate = getattr(mlp, "gate_proj", None)
    up = getattr(mlp, "up_proj", None)
    down = getattr(mlp, "down_proj", None)
    if mlp is None or not all(isinstance(item, nn.Linear) for item in (gate, up, down)):
        raise TypeError("expected a gate/up/down SwiGLU MLP")
    return mlp, gate, up, down


def prompt_text(tokenizer: Any, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def encode_prompts(
    tokenizer: Any,
    prompts: Sequence[str],
    device: torch.device,
    max_length: int,
) -> dict[str, Tensor]:
    encoded = tokenizer(
        [prompt_text(tokenizer, prompt) for prompt in prompts],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    return {key: value.to(device) for key, value in encoded.items() if isinstance(value, Tensor)}


@dataclass
class Capture:
    ids: list[str]
    tasks: list[str]
    state: Tensor
    transition: Tensor
    scores: Tensor
    logits: Tensor


@torch.inference_mode()
def capture_split(
    model: nn.Module,
    tokenizer: Any,
    examples: Sequence[dict[str, str]],
    layer: nn.Module,
    *,
    device: torch.device,
    batch_size: int,
    max_length: int,
    log: RunLog,
) -> Capture:
    mlp, gate, up, down = resolve_swiglu(layer)
    states: list[Tensor] = []
    transitions: list[Tensor] = []
    scores: list[Tensor] = []
    logits: list[Tensor] = []
    captured: dict[str, Tensor] = {}

    def prehook(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
        x = args[0]
        state = x[:, -1, :]
        previous = x[:, -2, :]
        activation = F.silu(gate(state)) * up(state)
        captured["state"] = state.detach()
        captured["transition"] = (state - previous).detach()
        captured["scores"] = contribution_scores(activation, down.weight)

    handle = mlp.register_forward_pre_hook(prehook)
    try:
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            encoded = encode_prompts(
                tokenizer,
                [row["prompt"] for row in batch],
                device,
                max_length,
            )
            output = model(**encoded, use_cache=False)
            states.append(captured["state"].float().cpu())
            transitions.append(captured["transition"].float().cpu())
            scores.append(captured["scores"].float().cpu())
            logits.append(output.logits[:, -1, :].detach().to(torch.float16).cpu())
            if start == 0 or (start + batch_size) % (batch_size * 8) == 0:
                log.write("capture_progress", split=batch[0]["split"], completed=len(logits) * batch_size)
    finally:
        handle.remove()
    return Capture(
        ids=[row["id"] for row in examples],
        tasks=[row["task"] for row in examples],
        state=torch.cat(states),
        transition=torch.cat(transitions),
        scores=torch.cat(scores),
        logits=torch.cat(logits),
    )


def feature_tensor(capture: Capture, variant: str) -> Tensor:
    if variant == "state":
        return capture.state
    if variant == "state_transition":
        return torch.cat((capture.state, capture.transition), dim=1)
    raise ValueError(f"unknown feature variant: {variant}")


def normalized_targets(scores: Tensor) -> Tensor:
    transformed = torch.log1p(scores.clamp_min(0))
    return (transformed - transformed.mean(dim=1, keepdim=True)) / transformed.std(
        dim=1, keepdim=True
    ).clamp_min(1e-4)


def selection_metrics(predicted_scores: Tensor, true_scores: Tensor, fraction: float) -> dict[str, Any]:
    predicted = topk_rows(predicted_scores, fraction)
    oracle = topk_rows(true_scores, fraction)
    set_metrics = row_set_metrics(predicted, oracle)
    predicted_mass = true_scores.gather(1, predicted).sum(dim=1)
    oracle_mass = true_scores.gather(1, oracle).sum(dim=1).clamp_min(1e-12)
    return {
        "mean_oracle_set_recall": float(set_metrics["recall"].mean()),
        "mean_oracle_set_jaccard": float(set_metrics["jaccard"].mean()),
        "mean_oracle_utility_mass_ratio": float((predicted_mass / oracle_mass).mean()),
        "selections": predicted,
        "oracle": oracle,
    }


def train_predictor(
    calibration_features: Tensor,
    calibration_scores: Tensor,
    development_features: Tensor,
    development_scores: Tensor,
    *,
    rank: int,
    config: dict[str, Any],
    device: torch.device,
    seed: int,
) -> tuple[LowRankWorkingSetPredictor, dict[str, Any]]:
    seed_everything(seed)
    predictor = LowRankWorkingSetPredictor(
        calibration_features.shape[1], rank, calibration_scores.shape[1]
    ).to(device)
    calibration_features = calibration_features.to(device)
    development_features = development_features.to(device)
    calibration_scores = calibration_scores.to(device)
    development_scores = development_scores.to(device)
    predictor.set_normalization(calibration_features)
    targets = normalized_targets(calibration_scores)
    optimizer = torch.optim.AdamW(
        predictor.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best_recall = -1.0
    best_mass = -1.0
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    stale_epochs = 0
    started = time.perf_counter()
    for epoch in range(int(config["training_epochs"])):
        predictor.train()
        permutation = torch.randperm(calibration_features.shape[0], generator=generator)
        batch_size = int(config["training_batch_size"])
        for start in range(0, permutation.numel(), batch_size):
            index = permutation[start : start + batch_size].to(device)
            prediction = predictor(calibration_features.index_select(0, index))
            loss = F.mse_loss(prediction, targets.index_select(0, index))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        predictor.eval()
        with torch.inference_mode():
            development_prediction = predictor(development_features)
            metrics = selection_metrics(
                development_prediction,
                development_scores,
                float(config["retained_fraction"]),
            )
        recall = float(metrics["mean_oracle_set_recall"])
        mass = float(metrics["mean_oracle_utility_mass_ratio"])
        if (recall, mass) > (best_recall, best_mass):
            best_recall, best_mass = recall, mass
            best_epoch = epoch + 1
            best_state = copy.deepcopy(predictor.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= int(config["early_stop_patience"]):
            break
    if best_state is None:
        raise RuntimeError("predictor training did not produce a checkpoint")
    predictor.load_state_dict(best_state)
    predictor.eval()
    elapsed = time.perf_counter() - started
    return predictor.cpu(), {
        "rank": rank,
        "best_epoch": best_epoch,
        "epochs_executed": epoch + 1,
        "training_seconds": elapsed,
        "development_oracle_set_recall": best_recall,
        "development_oracle_utility_mass_ratio": best_mass,
    }


def train_candidate_grid(
    calibration: Capture,
    development: Capture,
    calibration_scores: Tensor,
    development_scores: Tensor,
    *,
    config: dict[str, Any],
    device: torch.device,
    log: RunLog,
    label: str,
) -> tuple[LowRankWorkingSetPredictor, str, list[dict[str, Any]]]:
    trained: list[tuple[LowRankWorkingSetPredictor, str, dict[str, Any]]] = []
    for variant_index, variant in enumerate(config["feature_variants"]):
        for rank in config["predictor_ranks"]:
            log.write("training_candidate", family=label, feature_variant=variant, rank=rank)
            predictor, metrics = train_predictor(
                feature_tensor(calibration, variant),
                calibration_scores,
                feature_tensor(development, variant),
                development_scores,
                rank=int(rank),
                config=config,
                device=device,
                seed=int(config["seed"]) + variant_index * 1000 + int(rank),
            )
            footprint = predictor_footprint(
                predictor,
                hidden_width=calibration.state.shape[1],
                intermediate_width=calibration.scores.shape[1],
            )
            result = {
                "family": label,
                "feature_variant": variant,
                **metrics,
                "footprint": asdict(footprint),
            }
            trained.append((predictor, variant, result))
            log.write("candidate_complete", **result)
    trained.sort(
        key=lambda item: (
            item[2]["development_oracle_set_recall"],
            item[2]["development_oracle_utility_mass_ratio"],
            -item[2]["footprint"]["parameters"],
        ),
        reverse=True,
    )
    selected_predictor, selected_variant, _ = trained[0]
    return selected_predictor, selected_variant, [item[2] for item in trained]


@torch.inference_mode()
def predict_scores(
    predictor: LowRankWorkingSetPredictor,
    features: Tensor,
    device: torch.device,
    batch_size: int = 64,
) -> Tensor:
    predictor = predictor.to(device)
    rows = []
    for start in range(0, features.shape[0], batch_size):
        rows.append(predictor(features[start : start + batch_size].to(device)).cpu())
    predictor.cpu()
    return torch.cat(rows)


def bootstrap_mean_ci(values: Sequence[float], seed: int, draws: int) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, array.size, size=(draws, array.size))
    means = array[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


@torch.inference_mode()
def evaluate_selections(
    name: str,
    model: nn.Module,
    tokenizer: Any,
    examples: Sequence[dict[str, str]],
    reference_logits: Tensor,
    selections: Tensor,
    layer: nn.Module,
    *,
    config: dict[str, Any],
    device: torch.device,
    recovery: dict[str, Tensor] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    mlp, gate, up, down = resolve_swiglu(layer)
    active_selection: Tensor | None = None

    def hook(_module: nn.Module, args: tuple[Tensor, ...], output: Tensor) -> Tensor:
        if active_selection is None:
            raise RuntimeError("selection was not set")
        x = args[0]
        # Quality is evaluated as a counterfactual mask in the original dense
        # arithmetic shape.  This removes kernel-shape roundoff from the
        # selection question; actual packed execution is reserved for the
        # conditional fused-performance gate.
        activation = F.silu(gate(x)) * up(x)
        mask = torch.zeros_like(activation[:, -1, :])
        mask.scatter_(1, active_selection, 1)
        activation[:, -1, :] = activation[:, -1, :] * mask
        replacement = down(activation)[:, -1:, :]
        result = output.clone()
        result[:, -1:, :] = replacement
        return result

    handle = mlp.register_forward_hook(hook)
    rows: list[dict[str, Any]] = []
    batch_size = int(config["evaluation_batch_size"])
    try:
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            active_selection = selections[start : start + len(batch)].to(device)
            encoded = encode_prompts(
                tokenizer,
                [row["prompt"] for row in batch],
                device,
                int(config["max_prompt_tokens"]),
            )
            candidate = model(**encoded, use_cache=False).logits[:, -1, :].float()
            reference = reference_logits[start : start + len(batch)].to(device).float()
            reference_log_probability = F.log_softmax(reference, dim=-1)
            candidate_log_probability = F.log_softmax(candidate, dim=-1)
            divergence = (
                reference_log_probability.exp()
                * (reference_log_probability - candidate_log_probability)
            ).sum(dim=-1)
            agreement = reference.argmax(dim=-1) == candidate.argmax(dim=-1)
            for offset, example in enumerate(batch):
                row: dict[str, Any] = {
                    "method": name,
                    "id": example["id"],
                    "task": example["task"],
                    "kl": float(divergence[offset]),
                    "top1_agreement": float(agreement[offset]),
                }
                if recovery is not None:
                    row["oracle_set_recall"] = float(recovery["recall"][start + offset])
                    row["oracle_set_jaccard"] = float(recovery["jaccard"][start + offset])
                rows.append(row)
    finally:
        handle.remove()
    kl_values = [float(row["kl"]) for row in rows]
    agreement_values = [float(row["top1_agreement"]) for row in rows]
    kl_ci = bootstrap_mean_ci(
        kl_values, int(config["seed"]) + 1, int(config["bootstrap_draws"])
    )
    agreement_ci = bootstrap_mean_ci(
        agreement_values, int(config["seed"]) + 2, int(config["bootstrap_draws"])
    )
    mean_kl = float(np.mean(kl_values))
    top1 = float(np.mean(agreement_values))
    summary = {
        "method": name,
        "examples": len(rows),
        "mean_kl": mean_kl,
        "mean_kl_ci95": list(kl_ci),
        "top1_agreement": top1,
        "top1_agreement_ci95": list(agreement_ci),
        "passes_quality_gate": mean_kl < float(config["quality_gate_kl"])
        and top1 >= float(config["quality_gate_top1"]),
        "by_task": {},
    }
    for task in sorted({row["task"] for row in rows}):
        task_rows = [row for row in rows if row["task"] == task]
        summary["by_task"][task] = {
            "examples": len(task_rows),
            "mean_kl": float(np.mean([row["kl"] for row in task_rows])),
            "top1_agreement": float(np.mean([row["top1_agreement"] for row in task_rows])),
        }
    if recovery is not None:
        summary["mean_oracle_set_recall"] = float(recovery["recall"].mean())
        summary["mean_oracle_set_jaccard"] = float(recovery["jaccard"].mean())
    return summary, rows


def benchmark_call(
    function: Callable[[], Tensor],
    *,
    warmup: int,
    repetitions: int,
    device: torch.device,
) -> dict[str, Any]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize(device)
    timings = []
    for _ in range(repetitions):
        started = torch.cuda.Event(enable_timing=True)
        finished = torch.cuda.Event(enable_timing=True)
        started.record()
        function()
        finished.record()
        finished.synchronize()
        timings.append(float(started.elapsed_time(finished)))
    array = np.asarray(timings)
    return {
        "repetitions": repetitions,
        "median_ms": float(np.median(array)),
        "mean_ms": float(array.mean()),
        "p95_ms": float(np.quantile(array, 0.95)),
        "samples_ms": timings,
    }


class CompiledPredictiveSwiGLU(nn.Module):
    """Overhead-inclusive predictor, top-k router, packed SwiGLU candidate."""

    def __init__(
        self,
        predictor: LowRankWorkingSetPredictor,
        variant: str,
        gate: nn.Linear,
        up: nn.Linear,
        down: nn.Linear,
        fraction: float,
    ) -> None:
        super().__init__()
        self.predictor = predictor
        self.variant = variant
        self.gate = gate
        self.up = up
        self.down = down
        self.retained = round(gate.out_features * fraction)

    def forward(self, state: Tensor, previous: Tensor) -> Tensor:
        features = state if self.variant == "state" else torch.cat((state, state - previous), dim=-1)
        selection = self.predictor(features).topk(self.retained, dim=-1, sorted=False).indices[0]
        return swiglu_subset(state, self.gate, self.up, self.down, selection)


def fused_performance_gate(
    predictor: LowRankWorkingSetPredictor,
    variant: str,
    capture: Capture,
    layer: nn.Module,
    *,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    _mlp, gate, up, down = resolve_swiglu(layer)
    predictor = predictor.to(device=device, dtype=gate.weight.dtype)
    candidate = CompiledPredictiveSwiGLU(
        predictor,
        variant,
        gate,
        up,
        down,
        float(config["retained_fraction"]),
    ).to(device)
    state = capture.state[:1].to(device=device, dtype=gate.weight.dtype)
    previous = (capture.state[:1] - capture.transition[:1]).to(
        device=device, dtype=gate.weight.dtype
    )
    def dense() -> Tensor:
        return down(F.silu(gate(state)) * up(state))
    result: dict[str, Any] = {
        "scope": "single-token layer-23 MLP",
        "includes": ["causal predictor", "top-k routing", "weight gather", "SwiGLU", "aggregation"],
        "claim_boundary": "Measured on CUDA; this is an operator gate, not end-to-end model latency.",
    }
    result["dense"] = benchmark_call(
        dense,
        warmup=int(config["fused_warmup"]),
        repetitions=int(config["fused_repetitions"]),
        device=device,
    )
    try:
        compiled = torch.compile(candidate, fullgraph=True, mode="max-autotune")
        result["compiled_candidate"] = benchmark_call(
            lambda: compiled(state, previous),
            warmup=int(config["fused_warmup"]),
            repetitions=int(config["fused_repetitions"]),
            device=device,
        )
        result["speedup"] = (
            result["dense"]["median_ms"] / result["compiled_candidate"]["median_ms"]
        )
        result["net_performance_improvement"] = result["speedup"] > 1.0
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - compiler dependent
        result["compile_error"] = f"{type(exc).__name__}: {exc}"
        result["net_performance_improvement"] = False
    return result


def report_markdown(summary: dict[str, Any]) -> str:
    primary = summary["individual_neuron"]["heldout"]["predictor"]
    oracle = summary["individual_neuron"]["heldout"]["oracle"]
    control = summary["individual_neuron"]["heldout"]["full_width_control"]
    lines = [
        "# Predictive working-set feasibility",
        "",
        f"**Decision: {summary['decision']}**",
        "",
        (
            "The predictor was selected using calibration/development data before the 512-prompt "
            "held-out split was captured. The quality gate required mean KL < 0.01 and dense "
            "top-1 agreement >= 99%."
        ),
        "",
        "## Individual-neuron result",
        "",
        "| Method | Held-out prompts | Mean KL | Top-1 agreement | Oracle-set recall | Pass |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| Full-width reconstruction control | {control['examples']} | "
            f"{control['mean_kl']:.6f} | {control['top1_agreement']:.3%} | n/a | "
            f"{control['passes_quality_gate']} |"
        ),
        (
            f"| Non-deployable oracle 75% | {oracle['examples']} | {oracle['mean_kl']:.6f} | "
            f"{oracle['top1_agreement']:.3%} | 100.000% | {oracle['passes_quality_gate']} |"
        ),
        (
            f"| Causal predictor 75% | {primary['examples']} | {primary['mean_kl']:.6f} | "
            f"{primary['top1_agreement']:.3%} | {primary['mean_oracle_set_recall']:.3%} | "
            f"{primary['passes_quality_gate']} |"
        ),
        "",
    ]
    if summary.get("structured_operator_pivot"):
        pivot = summary["structured_operator_pivot"]["heldout"]
        lines.extend(
            [
                "## Structured operator-page pivot",
                "",
                (
                    "The failed neuron gate triggered a 16-page contiguous layout; exactly 12 "
                    "pages (75% of 4,864 neurons) execute per token."
                ),
                "",
                "| Method | Mean KL | Top-1 agreement | Oracle-page recall | Pass |",
                "|---|---:|---:|---:|---:|",
                (
                    f"| Page oracle | {pivot['oracle']['mean_kl']:.6f} | "
                    f"{pivot['oracle']['top1_agreement']:.3%} | 100.000% | "
                    f"{pivot['oracle']['passes_quality_gate']} |"
                ),
                (
                    f"| Causal page predictor | {pivot['predictor']['mean_kl']:.6f} | "
                    f"{pivot['predictor']['top1_agreement']:.3%} | "
                    f"{pivot['predictor']['mean_oracle_set_recall']:.3%} | "
                    f"{pivot['predictor']['passes_quality_gate']} |"
                ),
                "",
            ]
        )
    performance = summary["fused_performance"]
    lines.extend(["## Fused performance gate", ""])
    if performance.get("status") == "skipped":
        lines.append(f"Skipped: {performance['reason']}")
    else:
        lines.append(
            f"The overhead-inclusive compiled operator speedup was {performance.get('speedup', 0):.3f}x; "
            f"net improvement: {performance['net_performance_improvement']}."
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            (
                "Oracle masks inspect the current dense intermediate activation and are not "
                "deployable. The learned predictor sees only the current causal MLP input (and, "
                "for the transition variant, its previous-token difference). No optical timing "
            "is inferred in this test. Quality uses dense-shape counterfactual masking to "
            "isolate selection error from packed-kernel roundoff."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log = RunLog(output_dir / "research_log.jsonl")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this experiment")
    seed_everything(int(config["seed"]))
    device = torch.device("cuda")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[config["dtype"]]

    manifest = make_evaluation_manifest()
    validate_manifest(manifest)
    (output_dir / "evaluation_manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest), encoding="utf-8"
    )
    write_json(output_dir / "resolved_config.json", config)
    write_json(output_dir / "environment.json", environment_snapshot())
    log.write("run_started", output_dir=str(output_dir), examples=len(manifest))

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["revision"])
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config["model"],
        revision=config["revision"],
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    layer = resolve_layers(model)[int(config["target_layer"])]

    by_split = {
        split: [row for row in manifest if row["split"] == split]
        for split in ("calibration", "development", "heldout")
    }
    calibration = capture_split(
        model,
        tokenizer,
        by_split["calibration"],
        layer,
        device=device,
        batch_size=int(config["capture_batch_size"]),
        max_length=int(config["max_prompt_tokens"]),
        log=log,
    )
    development = capture_split(
        model,
        tokenizer,
        by_split["development"],
        layer,
        device=device,
        batch_size=int(config["capture_batch_size"]),
        max_length=int(config["max_prompt_tokens"]),
        log=log,
    )

    neuron_predictor, neuron_variant, neuron_candidates = train_candidate_grid(
        calibration,
        development,
        calibration.scores,
        development.scores,
        config=config,
        device=device,
        log=log,
        label="individual_neuron",
    )
    torch.save(
        {
            "feature_variant": neuron_variant,
            "state_dict": neuron_predictor.state_dict(),
        },
        output_dir / "selected_neuron_predictor.pt",
    )
    write_json(output_dir / "neuron_candidate_selection.json", neuron_candidates)
    log.write("heldout_opened", reason="neuron predictor selected on development data")
    heldout = capture_split(
        model,
        tokenizer,
        by_split["heldout"],
        layer,
        device=device,
        batch_size=int(config["capture_batch_size"]),
        max_length=int(config["max_prompt_tokens"]),
        log=log,
    )

    fraction = float(config["retained_fraction"])
    oracle_neurons = topk_rows(heldout.scores, fraction)
    neuron_prediction = predict_scores(
        neuron_predictor, feature_tensor(heldout, neuron_variant), device
    )
    predicted_neurons = topk_rows(neuron_prediction, fraction)
    neuron_recovery = row_set_metrics(predicted_neurons, oracle_neurons)
    static_neurons = topk_rows(calibration.scores.mean(dim=0, keepdim=True), fraction).repeat(
        heldout.scores.shape[0], 1
    )

    individual_summaries: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    full_width = torch.arange(heldout.scores.shape[1]).unsqueeze(0).repeat(
        heldout.scores.shape[0], 1
    )
    for name, selections, recovery in (
        ("full_width_control", full_width, None),
        ("oracle", oracle_neurons, row_set_metrics(oracle_neurons, oracle_neurons)),
        ("predictor", predicted_neurons, neuron_recovery),
        ("calibration_static", static_neurons, row_set_metrics(static_neurons, oracle_neurons)),
    ):
        log.write("heldout_quality_started", method=name)
        method_summary, method_rows = evaluate_selections(
            name,
            model,
            tokenizer,
            by_split["heldout"],
            heldout.logits,
            selections,
            layer,
            config=config,
            device=device,
            recovery=recovery,
        )
        individual_summaries[name] = method_summary
        all_rows.extend(method_rows)
        log.write("heldout_quality_complete", **method_summary)

    predictor_passed = bool(individual_summaries["predictor"]["passes_quality_gate"])
    structured_pivot: dict[str, Any] | None = None
    if not predictor_passed:
        log.write("pivot_started", reason="individual-neuron predictor failed held-out quality gate")
        pages = contiguous_operator_pages(calibration.scores.shape[1], int(config["operator_pages"]))
        calibration_page_scores = aggregate_page_scores(calibration.scores, pages)
        development_page_scores = aggregate_page_scores(development.scores, pages)
        page_predictor, page_variant, page_candidates = train_candidate_grid(
            calibration,
            development,
            calibration_page_scores,
            development_page_scores,
            config=config,
            device=device,
            log=log,
            label="operator_page",
        )
        heldout_page_scores = aggregate_page_scores(heldout.scores, pages)
        page_fraction = round(len(pages) * fraction) / len(pages)
        oracle_page_ids = topk_rows(heldout_page_scores, page_fraction)
        predicted_page_scores = predict_scores(
            page_predictor, feature_tensor(heldout, page_variant), device
        )
        predicted_page_ids = topk_rows(predicted_page_scores, page_fraction)
        oracle_page_neurons = expand_page_selections(oracle_page_ids, pages)
        predicted_page_neurons = expand_page_selections(predicted_page_ids, pages)
        page_recovery = row_set_metrics(predicted_page_ids, oracle_page_ids)
        page_heldout: dict[str, Any] = {}
        for name, selections, recovery in (
            (
                "oracle",
                oracle_page_neurons,
                row_set_metrics(oracle_page_ids, oracle_page_ids),
            ),
            ("predictor", predicted_page_neurons, page_recovery),
        ):
            method_summary, method_rows = evaluate_selections(
                f"operator_page_{name}",
                model,
                tokenizer,
                by_split["heldout"],
                heldout.logits,
                selections,
                layer,
                config=config,
                device=device,
                recovery=recovery,
            )
            page_heldout[name] = method_summary
            all_rows.extend(method_rows)
            log.write("pivot_quality_complete", **method_summary)
        structured_pivot = {
            "layout": {
                "page_count": len(pages),
                "page_widths": [page.numel() for page in pages],
                "selected_pages": oracle_page_ids.shape[1],
                "retained_neurons": oracle_page_neurons.shape[1],
                "retained_fraction": oracle_page_neurons.shape[1] / heldout.scores.shape[1],
            },
            "selected_feature_variant": page_variant,
            "candidate_selection": page_candidates,
            "heldout": page_heldout,
        }
        torch.save(
            {"feature_variant": page_variant, "state_dict": page_predictor.state_dict()},
            output_dir / "selected_operator_page_predictor.pt",
        )
        write_json(output_dir / "structured_operator_pivot.json", structured_pivot)

    if predictor_passed:
        log.write("fused_performance_started")
        fused_performance = fused_performance_gate(
            neuron_predictor,
            neuron_variant,
            heldout,
            layer,
            config=config,
            device=device,
        )
        fused_performance["status"] = "executed"
        log.write("fused_performance_complete", **fused_performance)
    else:
        fused_performance = {
            "status": "skipped",
            "reason": "preregistered individual-neuron quality gate failed",
        }

    if predictor_passed and fused_performance.get("net_performance_improvement"):
        decision = "INDIVIDUAL_NEURON_FEASIBLE_AND_FASTER"
    elif predictor_passed:
        decision = "QUALITY_PASSED_BUT_NO_NET_FUSED_SPEEDUP"
    else:
        page_pass = bool(
            structured_pivot
            and structured_pivot["heldout"]["predictor"]["passes_quality_gate"]
        )
        decision = (
            "INDIVIDUAL_NEURON_FAILED_PIVOT_TO_OPERATOR_PAGES"
            if page_pass
            else "INDIVIDUAL_NEURON_FAILED_OPERATOR_PAGE_PIVOT_ALSO_FAILED"
        )
    summary = {
        "decision": decision,
        "preregistered_gate": {
            "mean_kl_strictly_below": config["quality_gate_kl"],
            "top1_agreement_at_least": config["quality_gate_top1"],
            "retained_fraction": fraction,
        },
        "dataset": {
            "total_prompts": len(manifest),
            "split_counts": {
                split: len(examples) for split, examples in by_split.items()
            },
            "heldout_opened_after_model_selection": True,
        },
        "individual_neuron": {
            "selected_feature_variant": neuron_variant,
            "selected_candidate": neuron_candidates[0],
            "candidate_selection": neuron_candidates,
            "heldout": individual_summaries,
        },
        "structured_operator_pivot": structured_pivot,
        "fused_performance": fused_performance,
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "heldout_quality.json", individual_summaries)
    write_csv(output_dir / "heldout_quality.csv", all_rows)
    worst = sorted(all_rows, key=lambda row: float(row["kl"]), reverse=True)[:40]
    write_json(output_dir / "worst_cases.json", worst)
    report = report_markdown(summary)
    (output_dir / "PREDICTIVE_WORKING_SET_REPORT.md").write_text(report, encoding="utf-8")
    log.write("run_complete", decision=decision)


if __name__ == "__main__":
    main()
