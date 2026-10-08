# ruff: noqa: I001
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


DEFAULT_TEXTS = [
    "Explain why a cache miss can dominate inference latency and how repeated requests create locality.",
    "Write a short proof that a function-preserving neuron permutation keeps a dense MLP unchanged.",
    "Compare bandwidth-bound and compute-bound inference over several generated tokens.",
    "A user asks the same question in several different ways. What latent structure could be shared?",
    "Summarize posterior uncertainty, predictive state, and model confidence over time.",
    "Describe how a query planner coalesces repeated I/O across adjacent requests.",
    "What information should remain resident when future computation is uncertain?",
    "Explain why parameter count is not the same thing as bytes moved through a hierarchy.",
    "Give an example of a compiler optimization that changes layout but not semantics.",
    "What is the difference between semantic similarity and computational co-demand?",
    "Explain a rate-distortion tradeoff and how it interacts with a physical cache.",
    "Why can adaptive routing fail if routing metadata is too expensive?",
    "Describe a memory hierarchy from registers through device memory and backing store.",
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


def resolve_down(layer: nn.Module) -> nn.Linear:
    mlp = getattr(layer, "mlp", None)
    down = getattr(mlp, "down_proj", None) if mlp is not None else None
    if not isinstance(down, nn.Linear):
        raise TypeError("layer does not expose a SwiGLU down projection")
    return down


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
def capture_intermediate(
    model: nn.Module,
    down: nn.Linear,
    encoded: dict[str, Tensor],
) -> tuple[Tensor, Tensor]:
    rows: list[Tensor] = []

    def prehook(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
        rows.append(args[0].detach().cpu())

    handle = down.register_forward_pre_hook(prehook)
    try:
        model(**encoded)
    finally:
        handle.remove()

    if len(rows) != 1:
        raise RuntimeError("expected one down projection invocation")
    return rows[0].float(), encoded["attention_mask"].bool().cpu()


def active_mask(intermediate: Tensor, down: nn.Linear, fraction: float) -> Tensor:
    valid_flat = intermediate
    column_norm = down.weight.detach().float().cpu().norm(dim=0)
    score = valid_flat.abs() * column_norm
    width = score.shape[-1]
    k = max(1, min(width, round(width * fraction)))
    selected = torch.topk(score, k=k, dim=-1, sorted=False).indices
    mask = torch.zeros_like(score, dtype=torch.bool)
    mask.scatter_(-1, selected, True)
    return mask


def balanced_binary_permutation(
    train_mask: Tensor,
    *,
    units_per_page: int,
    iterations: int,
    seed: int,
) -> Tensor:
    """Balanced spherical clustering of binary neuron-demand trajectories."""

    signatures = train_mask.float().T
    signatures = signatures / signatures.norm(dim=1, keepdim=True).clamp_min(1e-12)
    neurons = signatures.shape[0]
    pages = (neurons + units_per_page - 1) // units_per_page

    generator = torch.Generator().manual_seed(seed)
    seeds = torch.randperm(neurons, generator=generator)[:pages]
    centroids = signatures[seeds].clone()
    centroids = centroids / centroids.norm(dim=1, keepdim=True).clamp_min(1e-12)

    capacity = torch.full((pages,), units_per_page, dtype=torch.long)
    capacity[-1] = neurons - units_per_page * (pages - 1)
    assignment = torch.full((neurons,), -1, dtype=torch.long)

    for _ in range(iterations):
        similarity = signatures @ centroids.T
        confidence = similarity.max(dim=1).values
        order = torch.argsort(confidence, descending=True)
        preferences = torch.argsort(similarity, dim=1, descending=True)

        assignment.fill_(-1)
        remaining = capacity.clone()
        for neuron in order.tolist():
            for page in preferences[neuron].tolist():
                if int(remaining[page]) > 0:
                    assignment[neuron] = page
                    remaining[page] -= 1
                    break

        updated = []
        for page in range(pages):
            members = signatures[assignment == page]
            if members.numel() == 0:
                updated.append(centroids[page])
            else:
                center = members.mean(dim=0)
                updated.append(
                    center / center.norm().clamp_min(1e-12)
                )
        centroids = torch.stack(updated)

    similarity = signatures @ centroids.T
    own = similarity[torch.arange(neurons), assignment]
    composite = assignment.to(torch.float64) * 2.0 - own.to(torch.float64)
    return torch.argsort(composite)


def to_pages(mask: Tensor, permutation: Tensor, units_per_page: int) -> Tensor:
    reordered = mask[:, permutation]
    width = reordered.shape[-1]
    pages = (width + units_per_page - 1) // units_per_page
    padded = pages * units_per_page
    if padded > width:
        reordered = torch.cat(
            [
                reordered,
                torch.zeros(
                    reordered.shape[0],
                    padded - width,
                    dtype=torch.bool,
                ),
            ],
            dim=-1,
        )
    return reordered.reshape(-1, pages, units_per_page).any(dim=-1)


def page_metrics(pages: Tensor) -> dict[str, float]:
    fraction = pages.float().mean(dim=-1)
    return {
        "num_pages": pages.shape[-1],
        "mean_page_fraction": float(fraction.mean().item()),
        "p50_page_fraction": float(torch.quantile(fraction, 0.50).item()),
        "p95_page_fraction": float(torch.quantile(fraction, 0.95).item()),
        "mean_page_amplification_over_neuron_fraction": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--text-file")
    parser.add_argument("--max-length", type=int, default=48)
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--fraction", type=float, default=0.10)
    parser.add_argument(
        "--units-per-page",
        nargs="+",
        type=int,
        default=[16, 32, 64],
    )
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
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
    if not test_texts:
        raise ValueError("held-out text split is empty")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    layers = resolve_layers(model)
    if args.layer < 0 or args.layer >= len(layers):
        raise ValueError("layer index out of range")
    down = resolve_down(layers[args.layer])

    train_u, train_valid = capture_intermediate(
        model,
        down,
        encode(tokenizer, train_texts, device, args.max_length),
    )
    test_u, test_valid = capture_intermediate(
        model,
        down,
        encode(tokenizer, test_texts, device, args.max_length),
    )

    train_flat = train_u[train_valid]
    test_flat = test_u[test_valid]
    train_active = active_mask(train_flat, down.cpu(), args.fraction)
    test_active = active_mask(test_flat, down.cpu(), args.fraction)

    width = train_active.shape[-1]
    identity = torch.arange(width)
    rows = []

    for page_size in args.units_per_page:
        raw_pages = to_pages(test_active, identity, page_size)
        permutation = balanced_binary_permutation(
            train_active,
            units_per_page=page_size,
            iterations=args.iterations,
            seed=17 + page_size,
        )
        compiled_pages = to_pages(test_active, permutation, page_size)

        raw = page_metrics(raw_pages)
        compiled = page_metrics(compiled_pages)
        raw["mean_page_amplification_over_neuron_fraction"] = (
            raw["mean_page_fraction"] / args.fraction
        )
        compiled["mean_page_amplification_over_neuron_fraction"] = (
            compiled["mean_page_fraction"] / args.fraction
        )
        rows.append(
            {
                "units_per_page": page_size,
                "raw": raw,
                "compiled": compiled,
                "relative_page_fraction_reduction": (
                    1.0
                    - compiled["mean_page_fraction"]
                    / raw["mean_page_fraction"]
                ),
            }
        )

    payload = {
        "experiment": "hf_active_page_compile_v0",
        "model": args.model,
        "layer": args.layer,
        "active_neuron_fraction": args.fraction,
        "train_tokens": int(train_active.shape[0]),
        "test_tokens": int(test_active.shape[0]),
        "rows": rows,
        "claim_boundary": (
            "The physical neuron permutation is learned only from calibration "
            "prompts and tested on held-out prompts. Active sets still use exact "
            "dense activations. This measures whether demand locality can be "
            "compiled into contiguous pages, not measured HBM traffic."
        ),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        Path(args.json_out).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
