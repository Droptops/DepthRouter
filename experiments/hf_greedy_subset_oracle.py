# ruff: noqa: I001
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


DEFAULT_TEXTS = [
    "Explain why a cache miss can dominate inference latency.",
    "Write a short proof that a function-preserving neuron permutation keeps a dense MLP unchanged.",
    "Compare bandwidth-bound and compute-bound inference.",
    "A user asks the same question in several different ways. What latent structure could be shared?",
    "Summarize the difference between posterior uncertainty and model confidence.",
    "Describe how a query planner coalesces repeated I/O.",
    "What information should remain resident when future computation is uncertain?",
    "Explain why parameter count is not the same thing as bytes moved.",
    "Give an example of a compiler optimization that changes layout but not semantics.",
    "What is the difference between semantic similarity and computational co-demand?",
    "Explain a rate-distortion tradeoff in simple terms.",
    "Why can adaptive routing fail if routing metadata is too expensive?",
    "Describe a memory hierarchy from registers through device memory.",
    "Why should calibration target behavior rather than an arbitrary oracle label?",
    "What does it mean for two histories to induce the same predictive future?",
    "Describe how batch scheduling can turn many logical misses into one physical transfer.",
]


def load_texts(path: str | None) -> list[str]:
    if path is None:
        return list(DEFAULT_TEXTS)
    rows: list[str] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("{"):
            rows.append(str(json.loads(line)["text"]))
        else:
            rows.append(line)
    if not rows:
        raise ValueError("text corpus is empty")
    return rows


def resolve_layers(model: nn.Module) -> list[nn.Module]:
    candidates: list[Any] = [
        getattr(getattr(model, "model", None), "layers", None),
        getattr(model, "layers", None),
        getattr(getattr(model, "transformer", None), "h", None),
    ]
    for candidate in candidates:
        if candidate is not None and len(candidate) > 0:
            return list(candidate)
    raise ValueError("could not locate decoder layers")


def resolve_mlp(layer: nn.Module) -> nn.Module:
    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        raise TypeError("layer has no MLP")
    for name in ("gate_proj", "up_proj", "down_proj"):
        if not isinstance(getattr(mlp, name, None), nn.Linear):
            raise TypeError("expected a gate/up/down SwiGLU MLP")
    return mlp


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


@torch.inference_mode()
def capture_last_intermediate(
    model: nn.Module,
    mlp: nn.Module,
    encoded: dict[str, Tensor],
) -> Tensor:
    rows: list[Tensor] = []

    def prehook(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
        rows.append(args[0][:, -1, :].detach().float().cpu())

    handle = mlp.down_proj.register_forward_pre_hook(prehook)
    try:
        model(**encoded)
    finally:
        handle.remove()

    if len(rows) != 1:
        raise RuntimeError("expected exactly one down projection invocation")
    return rows[0]


def greedy_subset_counts(
    activation: Tensor,
    down_weight: Tensor,
    *,
    thresholds: list[float],
    block_size: int,
    max_fraction: float,
) -> dict[str, Any]:
    """Greedy fixed-coefficient subset sum over exact neuron contributions.

    Coefficients are the exact SwiGLU activations; only inclusion/exclusion is
    allowed. Each iteration scores the reduction in residual squared error from
    every remaining neuron, then admits a block of best candidates.
    """

    if activation.ndim != 2:
        raise ValueError("activation must have shape [examples, neurons]")
    if down_weight.ndim != 2 or down_weight.shape[1] != activation.shape[1]:
        raise ValueError("down weight shape does not match activation")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    if not 0 < max_fraction <= 1:
        raise ValueError("max_fraction must be in (0, 1]")
    if any(not 0 < threshold < 1 for threshold in thresholds):
        raise ValueError("thresholds must be in (0, 1)")

    u = activation.float()
    weight = down_weight.float()
    target = u @ weight.T
    residual = target.clone()
    denominator = target.norm(dim=-1).clamp_min(1e-12)
    column_norm_sq = weight.pow(2).sum(dim=0)

    batch, width = u.shape
    max_selected = min(width, max(1, round(width * max_fraction)))
    selected = torch.zeros(batch, width, dtype=torch.bool)
    counts = torch.zeros(batch, dtype=torch.long)

    reached = {
        threshold: torch.full((batch,), -1, dtype=torch.long)
        for threshold in thresholds
    }

    while int(counts.max().item()) < max_selected:
        relative = residual.norm(dim=-1) / denominator
        for threshold in thresholds:
            newly = (relative <= threshold) & (reached[threshold] < 0)
            reached[threshold][newly] = counts[newly]

        if all(bool((values >= 0).all()) for values in reached.values()):
            break

        dot = residual @ weight
        gain = 2.0 * u * dot - u.pow(2) * column_norm_sq[None, :]
        gain = gain.masked_fill(selected, float("-inf"))

        remaining = max_selected - int(counts.max().item())
        take = min(block_size, remaining)
        indices = torch.topk(gain, k=take, dim=-1, sorted=False).indices
        selected.scatter_(1, indices, True)

        coeff = torch.gather(u, 1, indices)
        neuron_vectors = weight.T[indices]
        delta = (coeff[..., None] * neuron_vectors).sum(dim=1)
        residual = residual - delta
        counts += take

    relative = residual.norm(dim=-1) / denominator
    for threshold in thresholds:
        newly = (relative <= threshold) & (reached[threshold] < 0)
        reached[threshold][newly] = counts[newly]

    output: dict[str, Any] = {
        "max_fraction": max_fraction,
        "block_size": block_size,
        "final_mean_relative_l2": float(relative.mean().item()),
        "final_p95_relative_l2": float(torch.quantile(relative, 0.95).item()),
        "thresholds": {},
    }
    for threshold in thresholds:
        values = reached[threshold]
        success = values >= 0
        fractions = values[success].float() / width
        output["thresholds"][str(threshold)] = {
            "success_fraction": float(success.float().mean().item()),
            "mean_neuron_fraction_when_reached": (
                float(fractions.mean().item()) if bool(success.any()) else None
            ),
            "p95_neuron_fraction_when_reached": (
                float(torch.quantile(fractions, 0.95).item())
                if bool(success.any())
                else None
            ),
        }

    return output


def naive_prefix_counts(
    activation: Tensor,
    down_weight: Tensor,
    *,
    thresholds: list[float],
    block_size: int,
    max_fraction: float,
) -> dict[str, Any]:
    u = activation.float()
    weight = down_weight.float()
    target = u @ weight.T
    denominator = target.norm(dim=-1).clamp_min(1e-12)
    score = u.abs() * weight.norm(dim=0)[None, :]
    order = torch.argsort(score, dim=-1, descending=True)

    batch, width = u.shape
    max_selected = min(width, max(1, round(width * max_fraction)))
    reached = {
        threshold: torch.full((batch,), -1, dtype=torch.long)
        for threshold in thresholds
    }

    selected = torch.zeros_like(u)
    for count in range(block_size, max_selected + 1, block_size):
        indices = order[:, count - block_size : count]
        selected.scatter_(1, indices, torch.gather(u, 1, indices))
        approx = selected @ weight.T
        relative = (target - approx).norm(dim=-1) / denominator
        for threshold in thresholds:
            newly = (relative <= threshold) & (reached[threshold] < 0)
            reached[threshold][newly] = count

    output: dict[str, Any] = {"thresholds": {}}
    for threshold in thresholds:
        values = reached[threshold]
        success = values >= 0
        fractions = values[success].float() / width
        output["thresholds"][str(threshold)] = {
            "success_fraction": float(success.float().mean().item()),
            "mean_neuron_fraction_when_reached": (
                float(fractions.mean().item()) if bool(success.any()) else None
            ),
            "p95_neuron_fraction_when_reached": (
                float(torch.quantile(fractions, 0.95).item())
                if bool(success.any())
                else None
            ),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--text-file")
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument("--layers", nargs="+", type=int, default=[0, 12, 23])
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.20, 0.10, 0.05, 0.01],
    )
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--max-fraction", type=float, default=0.50)
    parser.add_argument("--json-out")
    args = parser.parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit('Install with: pip install -e ".[hf]"') from exc

    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype)
    if device.type == "cpu" and dtype == torch.float16:
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(
        load_texts(args.text_file),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_length,
    )
    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
        if isinstance(value, Tensor)
    }

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    layers = resolve_layers(model)
    if any(idx < 0 or idx >= len(layers) for idx in args.layers):
        raise ValueError("layer index out of range")

    rows = []
    for layer_idx in args.layers:
        mlp = resolve_mlp(layers[layer_idx])
        activation = capture_last_intermediate(model, mlp, encoded)
        weight = mlp.down_proj.weight.detach().float().cpu()

        rows.append(
            {
                "layer": layer_idx,
                "examples": int(activation.shape[0]),
                "width": int(activation.shape[1]),
                "greedy_subset": greedy_subset_counts(
                    activation,
                    weight,
                    thresholds=args.thresholds,
                    block_size=args.block_size,
                    max_fraction=args.max_fraction,
                ),
                "naive_norm_prefix": naive_prefix_counts(
                    activation,
                    weight,
                    thresholds=args.thresholds,
                    block_size=args.block_size,
                    max_fraction=args.max_fraction,
                ),
            }
        )

    payload = {
        "experiment": "hf_greedy_subset_oracle_v0",
        "model": args.model,
        "rows": rows,
        "claim_boundary": (
            "Greedy subset search sees the exact down-projection weights and "
            "full dense activation, so it is an oracle for intrinsic fixed-"
            "coefficient neuron subset sparsity. It is not a deployable router."
        ),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        Path(args.json_out).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
