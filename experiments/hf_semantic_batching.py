# ruff: noqa: I001
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from depth_router import learn_simhash_codemand_permutation


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


@torch.inference_mode()
def capture(
    model: nn.Module,
    down: nn.Linear,
    encoded: dict[str, Tensor],
) -> tuple[Tensor, Tensor]:
    mlp_inputs: list[Tensor] = []
    intermediates: list[Tensor] = []
    mlp = None
    for module in model.modules():
        if getattr(module, "down_proj", None) is down:
            mlp = module
            break
    if mlp is None:
        raise RuntimeError("could not find owning MLP")

    def mlp_pre(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
        mlp_inputs.append(args[0].detach().cpu())

    def down_pre(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
        intermediates.append(args[0].detach().cpu())

    h1 = mlp.register_forward_pre_hook(mlp_pre)
    h2 = down.register_forward_pre_hook(down_pre)
    try:
        model(**encoded)
    finally:
        h1.remove()
        h2.remove()

    if len(mlp_inputs) != 1 or len(intermediates) != 1:
        raise RuntimeError("expected one layer invocation")
    return mlp_inputs[0].float(), intermediates[0].float()


def topk_neuron_mask(score: Tensor, fraction: float) -> Tensor:
    width = score.shape[-1]
    k = max(1, min(width, round(width * fraction)))
    indices = torch.topk(score, k=k, dim=-1, sorted=False).indices
    mask = torch.zeros_like(score, dtype=torch.bool)
    mask.scatter_(-1, indices, True)
    return mask


def to_pages(mask: Tensor, permutation: Tensor, units_per_page: int) -> Tensor:
    reordered = mask[:, permutation]
    width = reordered.shape[-1]
    num_pages = (width + units_per_page - 1) // units_per_page
    padded = num_pages * units_per_page
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
    return reordered.reshape(
        reordered.shape[0],
        num_pages,
        units_per_page,
    ).any(dim=-1)


def simhash_codes(x: Tensor, *, bits: int, seed: int) -> Tensor:
    if not 1 <= bits <= 30:
        raise ValueError("bits must be between 1 and 30")
    generator = torch.Generator().manual_seed(seed)
    projection = torch.randn(
        x.shape[-1],
        bits,
        generator=generator,
        dtype=x.dtype,
    )
    binary = (x @ projection) >= 0
    powers = (1 << torch.arange(bits, dtype=torch.int64))[None, :]
    return (binary.to(torch.int64) * powers).sum(dim=-1)


def group_union_stats(
    page_mask: Tensor,
    order: Tensor,
    *,
    batch_size: int,
) -> dict[str, float]:
    num_pages = page_mask.shape[-1]
    union_fractions = []
    independent_fractions = []

    ordered = page_mask[order]
    for start in range(0, ordered.shape[0], batch_size):
        group = ordered[start : start + batch_size]
        if group.shape[0] == 0:
            continue
        union = group.any(dim=0).float().mean()
        independent = group.float().mean(dim=-1).mean()
        union_fractions.append(float(union.item()))
        independent_fractions.append(float(independent.item()))

    union_tensor = torch.tensor(union_fractions)
    independent_tensor = torch.tensor(independent_fractions)
    return {
        "num_pages": num_pages,
        "groups": len(union_fractions),
        "mean_union_page_fraction": float(union_tensor.mean().item()),
        "p95_union_page_fraction": float(torch.quantile(union_tensor, 0.95).item()),
        "mean_per_token_page_fraction": float(independent_tensor.mean().item()),
        "dense_weight_fraction": 1.0,
        "sparse_vs_dense_union_reduction": float(
            1.0 - union_tensor.mean().item()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--text-file")
    parser.add_argument("--max-length", type=int, default=48)
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--fraction", type=float, default=0.25)
    parser.add_argument("--units-per-page", type=int, default=16)
    parser.add_argument("--scheduler-batch-size", type=int, default=16)
    parser.add_argument("--hash-bits", type=int, default=12)
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
    if args.layer < 0 or args.layer >= len(layers):
        raise ValueError("layer index out of range")
    down = resolve_down(layers[args.layer])

    hidden, intermediate = capture(model, down, encoded)
    valid = encoded["attention_mask"].bool().cpu()
    h = hidden[valid]
    u = intermediate[valid]

    column_norm = down.weight.detach().float().cpu().norm(dim=0)
    score = u.abs() * column_norm
    neurons = topk_neuron_mask(score, args.fraction)

    permutation = learn_simhash_codemand_permutation(
        u,
        down.cpu(),
        bits=16,
        max_trace_tokens=2048,
        seed=17,
    )
    pages = to_pages(neurons, permutation, args.units_per_page)

    count = pages.shape[0]
    sequential = torch.arange(count)
    generator = torch.Generator().manual_seed(29)
    random_order = torch.randperm(count, generator=generator)

    hidden_codes = simhash_codes(
        torch.nn.functional.normalize(h, dim=-1),
        bits=args.hash_bits,
        seed=31,
    )
    hidden_order = torch.argsort(hidden_codes)

    oracle_codes = simhash_codes(
        pages.float(),
        bits=args.hash_bits,
        seed=37,
    )
    oracle_order = torch.argsort(oracle_codes)

    payload = {
        "experiment": "hf_semantic_batching_v0",
        "model": args.model,
        "layer": args.layer,
        "active_neuron_fraction": args.fraction,
        "units_per_page": args.units_per_page,
        "tokens": count,
        "scheduler_batch_size": args.scheduler_batch_size,
        "schedules": {
            "sequence_order": group_union_stats(
                pages,
                sequential,
                batch_size=args.scheduler_batch_size,
            ),
            "random": group_union_stats(
                pages,
                random_order,
                batch_size=args.scheduler_batch_size,
            ),
            "hidden_state_hash": group_union_stats(
                pages,
                hidden_order,
                batch_size=args.scheduler_batch_size,
            ),
            "oracle_active_set_hash": group_union_stats(
                pages,
                oracle_order,
                batch_size=args.scheduler_batch_size,
            ),
        },
        "claim_boundary": (
            "Active neuron sets use exact dense activations. Oracle-active-set "
            "hashing is an upper-bound scheduler diagnostic; hidden-state hashing "
            "is deployable in principle. Union page fraction is structural and "
            "has not yet been validated as HBM traffic."
        ),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        Path(args.json_out).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
