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


def resolve_mlp(layer: nn.Module) -> nn.Module:
    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        raise ValueError("layer has no .mlp")
    for name in ("gate_proj", "up_proj", "down_proj"):
        if not isinstance(getattr(mlp, name, None), nn.Linear):
            raise TypeError("layer is not a gate/up/down SwiGLU block")
    return mlp


def load_texts(path: str | None) -> list[str]:
    if path is None:
        return list(DEFAULT_TEXTS)
    rows = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("{"):
            rows.append(str(json.loads(line)["text"]))
        else:
            rows.append(line)
    if len(rows) < 4:
        raise ValueError("need at least four texts")
    return rows


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


def encode(tokenizer, texts: list[str], device: torch.device, max_length: int) -> dict[str, Tensor]:
    batch = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return {
        key: value.to(device)
        for key, value in batch.items()
        if isinstance(value, Tensor)
    }


@torch.inference_mode()
def capture_mlp_io(
    model: nn.Module,
    layers: list[nn.Module],
    encoded: dict[str, Tensor],
    *,
    max_tokens: int,
) -> dict[int, tuple[Tensor, Tensor]]:
    inputs: dict[int, list[Tensor]] = {i: [] for i in range(len(layers))}
    outputs: dict[int, list[Tensor]] = {i: [] for i in range(len(layers))}
    handles = []

    for idx, layer in enumerate(layers):
        mlp = resolve_mlp(layer)

        def prehook(
            _module: nn.Module,
            args: tuple[Tensor, ...],
            *,
            layer_idx: int = idx,
        ) -> None:
            x = args[0].detach().reshape(-1, args[0].shape[-1]).cpu()
            inputs[layer_idx].append(x)

        def posthook(
            _module: nn.Module,
            _args: tuple[Tensor, ...],
            output: Tensor,
            *,
            layer_idx: int = idx,
        ) -> None:
            y = output.detach().reshape(-1, output.shape[-1]).cpu()
            outputs[layer_idx].append(y)

        handles.append(mlp.register_forward_pre_hook(prehook))
        handles.append(mlp.register_forward_hook(posthook))

    try:
        model(**encoded)
    finally:
        for handle in handles:
            handle.remove()

    result = {}
    for idx in range(len(layers)):
        x = torch.cat(inputs[idx], dim=0)[:max_tokens].float()
        y = torch.cat(outputs[idx], dim=0)[:max_tokens].float()
        result[idx] = (x, y)
    return result


def fit_ridge_operator(
    x: Tensor,
    y: Tensor,
    *,
    ridge: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Fit centered ridge map y ~= mean_y + (x - mean_x) @ W."""

    if ridge <= 0:
        raise ValueError("ridge must be positive")
    mean_x = x.mean(dim=0)
    mean_y = y.mean(dim=0)
    xc = x - mean_x
    yc = y - mean_y

    # Dual ridge is cheaper when trace count < hidden width.
    gram = xc @ xc.T
    scale = float(torch.trace(gram).item() / max(gram.shape[0], 1))
    regularization = ridge * max(scale, 1e-8)
    eye = torch.eye(gram.shape[0], dtype=gram.dtype)
    alpha = torch.linalg.solve(gram + regularization * eye, yc)
    weight = xc.T @ alpha
    return mean_x, mean_y, weight


def low_rank_predict_from_svd(
    x: Tensor,
    mean_x: Tensor,
    mean_y: Tensor,
    u: Tensor,
    s: Tensor,
    vh: Tensor,
    *,
    rank: int,
) -> Tensor:
    rank = min(rank, s.numel())
    left = (x - mean_x) @ u[:, :rank]
    left = left * s[:rank]
    return mean_y + left @ vh[:rank]


def summarize_error(reference: Tensor, candidate: Tensor) -> dict[str, float]:
    denominator = reference.norm(dim=-1).clamp_min(1e-12)
    relative = (reference - candidate).norm(dim=-1) / denominator
    return {
        "mean_relative_l2": float(relative.mean().item()),
        "p50_relative_l2": float(torch.quantile(relative, 0.50).item()),
        "p95_relative_l2": float(torch.quantile(relative, 0.95).item()),
        "fraction_below_20pct_error": float((relative <= 0.20).float().mean().item()),
        "fraction_below_10pct_error": float((relative <= 0.10).float().mean().item()),
        "fraction_below_5pct_error": float((relative <= 0.05).float().mean().item()),
    }


def mlp_parameter_count(mlp: nn.Module) -> int:
    total = 0
    for name in ("gate_proj", "up_proj", "down_proj"):
        module = getattr(mlp, name)
        total += module.weight.numel()
        if module.bias is not None:
            total += module.bias.numel()
    return int(total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--text-file")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--max-train-tokens", type=int, default=256)
    parser.add_argument("--max-test-tokens", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument(
        "--layer-indices",
        nargs="+",
        type=int,
        default=[],
        help="default scans five evenly spaced layers",
    )
    parser.add_argument(
        "--ranks",
        nargs="+",
        type=int,
        default=[8, 16, 32, 64, 128],
    )
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

    texts = load_texts(args.text_file)
    split = max(2, len(texts) // 2)
    train_texts = texts[:split]
    test_texts = texts[split:]
    if len(test_texts) < 2:
        raise ValueError("held-out split needs at least two texts")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    layers = resolve_layers(model)

    train = capture_mlp_io(
        model,
        layers,
        encode(tokenizer, train_texts, device, args.max_length),
        max_tokens=args.max_train_tokens,
    )
    test = capture_mlp_io(
        model,
        layers,
        encode(tokenizer, test_texts, device, args.max_length),
        max_tokens=args.max_test_tokens,
    )

    if args.layer_indices:
        selected_layers = args.layer_indices
    else:
        last = len(layers) - 1
        selected_layers = sorted(
            {
                0,
                last // 4,
                last // 2,
                (3 * last) // 4,
                last,
            }
        )
    if any(idx < 0 or idx >= len(layers) for idx in selected_layers):
        raise ValueError("layer index out of range")

    rows = []
    for idx in selected_layers:
        layer = layers[idx]
        x_train, y_train = train[idx]
        x_test, y_test = test[idx]
        mean_x, mean_y, weight = fit_ridge_operator(
            x_train,
            y_train,
            ridge=args.ridge,
        )

        u, s, vh = torch.linalg.svd(weight, full_matrices=False)

        mlp = resolve_mlp(layer)
        cold_parameters = mlp_parameter_count(mlp)
        cold_bytes_fp16 = cold_parameters * 2
        hidden = x_train.shape[-1]
        layer_result = {
            "layer": idx,
            "train_tokens": int(x_train.shape[0]),
            "test_tokens": int(x_test.shape[0]),
            "hidden_width": int(hidden),
            "cold_mlp_parameters": cold_parameters,
            "cold_mlp_bytes_fp16": cold_bytes_fp16,
            "ranks": {},
        }

        for rank in args.ranks:
            prediction = low_rank_predict_from_svd(
                x_test,
                mean_x,
                mean_y,
                u,
                s,
                vh,
                rank=rank,
            )
            # Runtime representation: two dense factors plus output bias/mean.
            metadata_parameters = 2 * hidden * min(rank, hidden) + hidden
            metadata_bytes_fp16 = metadata_parameters * 2
            layer_result["ranks"][str(rank)] = {
                "metadata_bytes_fp16": metadata_bytes_fp16,
                "metadata_fraction_of_cold": metadata_bytes_fp16 / cold_bytes_fp16,
                **summarize_error(y_test, prediction),
            }

        rows.append(layer_result)

    candidates = []
    for row in rows:
        for rank, metrics in row["ranks"].items():
            if metrics["metadata_fraction_of_cold"] <= 0.10:
                candidates.append(
                    (
                        metrics["mean_relative_l2"],
                        row["layer"],
                        int(rank),
                        metrics,
                    )
                )
    candidates.sort(key=lambda item: item[0])
    best = [
        {
            "layer": layer,
            "rank": rank,
            **metrics,
        }
        for _, layer, rank, metrics in candidates[:10]
    ]

    payload = {
        "experiment": "hf_hot_surrogate_scan_v0",
        "model": args.model,
        "ridge": args.ridge,
        "ranks": args.ranks,
        "layers": rows,
        "best_under_10pct_metadata": best,
        "claim_boundary": (
            "A low-rank linear hot surrogate is fit on one prompt split and "
            "evaluated on held-out prompts. This tests repeated-trajectory "
            "compressibility of local MLP behavior, not end-to-end generation "
            "quality or measured hardware speed."
        ),
    }

    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        Path(args.json_out).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
