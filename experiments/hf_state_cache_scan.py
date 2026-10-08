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
def capture_layer_io(
    model: nn.Module,
    layer: nn.Module,
    encoded: dict[str, Tensor],
    *,
    max_tokens: int,
) -> tuple[Tensor, Tensor]:
    mlp = resolve_mlp(layer)
    inputs: list[Tensor] = []
    outputs: list[Tensor] = []

    def prehook(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
        x = args[0].detach().reshape(-1, args[0].shape[-1]).cpu()
        inputs.append(x)

    def posthook(
        _module: nn.Module,
        _args: tuple[Tensor, ...],
        output: Tensor,
    ) -> None:
        y = output.detach().reshape(-1, output.shape[-1]).cpu()
        outputs.append(y)

    h1 = mlp.register_forward_pre_hook(prehook)
    h2 = mlp.register_forward_hook(posthook)
    try:
        model(**encoded)
    finally:
        h1.remove()
        h2.remove()

    x = torch.cat(inputs, dim=0)[:max_tokens].float()
    y = torch.cat(outputs, dim=0)[:max_tokens].float()
    return x, y


def farthest_point_prototypes(x: Tensor, count: int) -> Tensor:
    """Deterministic cosine-space farthest-point cache selection."""

    count = min(max(count, 1), x.shape[0])
    normalized = torch.nn.functional.normalize(x, dim=-1)
    chosen = [0]
    best_similarity = normalized @ normalized[0]
    for _ in range(1, count):
        # Select the point least similar to its nearest chosen prototype.
        candidate = torch.argmin(best_similarity).item()
        chosen.append(candidate)
        similarity = normalized @ normalized[candidate]
        best_similarity = torch.maximum(best_similarity, similarity)
    return torch.tensor(chosen, dtype=torch.long)


def cache_predict(
    x_train: Tensor,
    y_train: Tensor,
    x_test: Tensor,
    *,
    cache_entries: int,
) -> tuple[Tensor, Tensor]:
    indices = farthest_point_prototypes(x_train, cache_entries)
    keys = torch.nn.functional.normalize(x_train[indices], dim=-1)
    values = y_train[indices]

    queries = torch.nn.functional.normalize(x_test, dim=-1)
    similarity = queries @ keys.T
    nearest = similarity.argmax(dim=-1)
    confidence = similarity.max(dim=-1).values
    return values[nearest], confidence


def error_metrics(reference: Tensor, candidate: Tensor) -> dict[str, float]:
    relative = (
        (reference - candidate).norm(dim=-1)
        / reference.norm(dim=-1).clamp_min(1e-12)
    )
    return {
        "mean_relative_l2": float(relative.mean().item()),
        "p50_relative_l2": float(torch.quantile(relative, 0.50).item()),
        "p95_relative_l2": float(torch.quantile(relative, 0.95).item()),
        "fraction_below_20pct_error": float((relative <= 0.20).float().mean().item()),
        "fraction_below_10pct_error": float((relative <= 0.10).float().mean().item()),
        "fraction_below_5pct_error": float((relative <= 0.05).float().mean().item()),
    }


def parameter_count(mlp: nn.Module) -> int:
    return sum(parameter.numel() for parameter in mlp.parameters())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--text-file")
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--max-train-tokens", type=int, default=192)
    parser.add_argument("--max-test-tokens", type=int, default=192)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=[0, 6, 12, 18, 23],
    )
    parser.add_argument(
        "--cache-entries",
        nargs="+",
        type=int,
        default=[16, 32, 64, 128],
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
    train_encoded = encode(tokenizer, train_texts, device, args.max_length)
    test_encoded = encode(tokenizer, test_texts, device, args.max_length)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    layers = resolve_layers(model)
    if any(idx < 0 or idx >= len(layers) for idx in args.layers):
        raise ValueError("layer index out of range")

    rows = []
    for idx in args.layers:
        layer = layers[idx]
        x_train, y_train = capture_layer_io(
            model,
            layer,
            train_encoded,
            max_tokens=args.max_train_tokens,
        )
        x_test, y_test = capture_layer_io(
            model,
            layer,
            test_encoded,
            max_tokens=args.max_test_tokens,
        )
        mlp = resolve_mlp(layer)
        cold_bytes_fp16 = parameter_count(mlp) * 2
        hidden = x_train.shape[-1]

        mean_prediction = y_train.mean(dim=0, keepdim=True).expand_as(y_test)
        layer_result = {
            "layer": idx,
            "train_tokens": int(x_train.shape[0]),
            "test_tokens": int(x_test.shape[0]),
            "cold_mlp_bytes_fp16": cold_bytes_fp16,
            "mean_output_baseline": error_metrics(y_test, mean_prediction),
            "caches": {},
        }

        for entries in args.cache_entries:
            prediction, confidence = cache_predict(
                x_train,
                y_train,
                x_test,
                cache_entries=entries,
            )
            actual_entries = min(entries, x_train.shape[0])
            cache_bytes_fp16 = actual_entries * hidden * 2 * 2
            layer_result["caches"][str(entries)] = {
                "cache_bytes_fp16": cache_bytes_fp16,
                "cache_fraction_of_cold": cache_bytes_fp16 / cold_bytes_fp16,
                "mean_cosine_address_confidence": float(confidence.mean().item()),
                "p05_cosine_address_confidence": float(
                    torch.quantile(confidence, 0.05).item()
                ),
                **error_metrics(y_test, prediction),
            }

        rows.append(layer_result)

    payload = {
        "experiment": "hf_state_cache_scan_v0",
        "model": args.model,
        "layers": rows,
        "claim_boundary": (
            "Keys and cached MLP outputs are selected only from the calibration "
            "prompt split; metrics are on held-out prompts. This tests semantic "
            "memoization of recurring hidden states, not final-logit quality or "
            "measured cache latency."
        ),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        Path(args.json_out).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
