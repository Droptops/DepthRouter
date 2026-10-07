from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from depth_router import page_importance


DEFAULT_TEXTS = [
    "Explain why a cache miss can dominate inference latency.",
    "Write a short proof that a function-preserving neuron permutation keeps a dense MLP unchanged.",
    "Compare bandwidth-bound and compute-bound inference.",
    "A user asks the same question in several different ways. What latent structure could be shared?",
    "Summarize the difference between posterior uncertainty and model confidence.",
    "Describe how a query planner coalesces repeated I/O.",
    "What information should remain resident when the future computation is uncertain?",
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


def resolve_swiglu(layer: nn.Module) -> tuple[nn.Linear, nn.Linear, nn.Linear]:
    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        raise ValueError("layer has no .mlp")
    gate = getattr(mlp, "gate_proj", None)
    up = getattr(mlp, "up_proj", None)
    down = getattr(mlp, "down_proj", None)
    if not all(isinstance(module, nn.Linear) for module in (gate, up, down)):
        raise ValueError("layer is not a gate/up/down SwiGLU block")
    return gate, up, down


def load_texts(path: str | None) -> list[str]:
    if path is None:
        return list(DEFAULT_TEXTS)
    rows = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("{"):
            payload = json.loads(line)
            rows.append(str(payload["text"]))
        else:
            rows.append(line)
    if not rows:
        raise ValueError("text corpus is empty")
    return rows


def parse_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def counts_for_mass(importance: Tensor, mass_target: float) -> Tensor:
    if not 0 < mass_target <= 1:
        raise ValueError("mass_target must be in (0, 1]")
    normalized = importance / importance.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    ordered = normalized.sort(dim=-1, descending=True).values
    cumulative = ordered.cumsum(dim=-1)
    return (cumulative < mass_target).sum(dim=-1) + 1


@torch.inference_mode()
def behavioral_topk_probe(
    intermediate: Tensor,
    down: nn.Linear,
    fractions: list[float],
) -> dict[str, dict[str, float]]:
    """Approximate dense down-projection with per-token top-k neurons.

    Ranking uses |activation_i| * ||W_down[:, i]||. This is not an oracle; it is
    deliberately cheap enough to expose whether obvious neuron-level sparsity
    exists before building a more sophisticated address function.
    """

    u = intermediate.float()
    weight = down.weight.detach().float().cpu()
    bias = down.bias.detach().float().cpu() if down.bias is not None else None
    dense = torch.nn.functional.linear(u, weight, bias)
    denominator = dense.norm(dim=-1).clamp_min(1e-12)

    column_norm = weight.norm(dim=0)
    importance = u.abs() * column_norm[None, :]
    order = torch.argsort(importance, dim=-1, descending=True)

    result: dict[str, dict[str, float]] = {}
    width = u.shape[-1]
    for fraction in fractions:
        k = max(1, min(width, int(round(width * fraction))))
        selected = order[:, :k]
        sparse = torch.zeros_like(u)
        values = torch.gather(u, 1, selected)
        sparse.scatter_(1, selected, values)

        approx = torch.nn.functional.linear(sparse, weight, bias)
        relative = (dense - approx).norm(dim=-1) / denominator
        result[str(fraction)] = {
            "k": float(k),
            "mean_relative_l2": float(relative.mean().item()),
            "p50_relative_l2": float(torch.quantile(relative, 0.50).item()),
            "p95_relative_l2": float(torch.quantile(relative, 0.95).item()),
            "fraction_below_10pct_error": float((relative <= 0.10).float().mean().item()),
            "fraction_below_5pct_error": float((relative <= 0.05).float().mean().item()),
            "fraction_below_1pct_error": float((relative <= 0.01).float().mean().item()),
        }

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--text-file")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--max-trace-tokens", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument(
        "--mass-targets",
        nargs="+",
        type=float,
        default=[0.90, 0.95, 0.99],
    )
    parser.add_argument(
        "--probe-fractions",
        nargs="+",
        type=float,
        default=[0.10, 0.25, 0.50],
    )
    parser.add_argument("--behavioral-layers", type=int, default=3)
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

    texts = load_texts(args.text_file)
    encoded = tokenizer(
        texts,
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

    captures: dict[int, list[Tensor]] = {idx: [] for idx in range(len(layers))}
    handles = []
    for idx, layer in enumerate(layers):
        _, _, down = resolve_swiglu(layer)

        def prehook(
            _module: nn.Module,
            hook_args: tuple[Tensor, ...],
            *,
            layer_idx: int = idx,
        ) -> None:
            hidden = hook_args[0].detach().reshape(-1, hook_args[0].shape[-1])
            captures[layer_idx].append(hidden.cpu())

        handles.append(down.register_forward_pre_hook(prehook))

    try:
        with torch.inference_mode():
            model(**encoded)
    finally:
        for handle in handles:
            handle.remove()

    layer_rows = []
    intermediates: dict[int, Tensor] = {}
    for idx, layer in enumerate(layers):
        _, _, down = resolve_swiglu(layer)
        intermediate = torch.cat(captures[idx], dim=0)[: args.max_trace_tokens].float()
        intermediates[idx] = intermediate

        column_norm = down.weight.detach().float().cpu().norm(dim=0)
        importance = intermediate.abs() * column_norm[None, :]
        width = importance.shape[-1]

        row: dict[str, Any] = {
            "layer": idx,
            "tokens": int(intermediate.shape[0]),
            "intermediate_width": int(width),
        }
        for target in args.mass_targets:
            counts = counts_for_mass(importance, target).float()
            row[f"mass_{target}_mean_neuron_fraction"] = float(
                (counts / width).mean().item()
            )
            row[f"mass_{target}_p95_neuron_fraction"] = float(
                torch.quantile(counts / width, 0.95).item()
            )
        layer_rows.append(row)

    ranking_key = "mass_0.95_mean_neuron_fraction"
    if ranking_key not in layer_rows[0]:
        target = args.mass_targets[min(1, len(args.mass_targets) - 1)]
        ranking_key = f"mass_{target}_mean_neuron_fraction"

    best_layers = sorted(layer_rows, key=lambda row: row[ranking_key])[
        : args.behavioral_layers
    ]

    behavioral = {}
    for row in best_layers:
        idx = int(row["layer"])
        _, _, down = resolve_swiglu(layers[idx])
        behavioral[str(idx)] = behavioral_topk_probe(
            intermediates[idx],
            down.cpu(),
            args.probe_fractions,
        )

    payload = {
        "experiment": "hf_neuron_sparsity_scan_v0",
        "model": args.model,
        "num_layers": len(layers),
        "trace_tokens": int(next(iter(intermediates.values())).shape[0]),
        "mass_targets": args.mass_targets,
        "probe_fractions": args.probe_fractions,
        "layers": layer_rows,
        "best_structural_layers": [int(row["layer"]) for row in best_layers],
        "behavioral_topk_probe": behavioral,
        "claim_boundary": (
            "Structural mass uses a contribution-norm proxy. Behavioral top-k "
            "measures relative L2 error of the MLP down-projection on captured "
            "pre-down activations. It is not yet an end-to-end logit or hardware result."
        ),
    }

    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        Path(args.json_out).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
