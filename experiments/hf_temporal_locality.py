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
    "Explain why a cache miss can dominate inference latency and how temporal locality changes the answer.",
    "Write a short proof that a function-preserving neuron permutation keeps a dense MLP unchanged.",
    "Compare bandwidth-bound and compute-bound inference over several generated tokens.",
    "A user asks the same question in several different ways. What latent structure could be shared?",
    "Summarize the difference between posterior uncertainty and model confidence over time.",
    "Describe how a query planner coalesces repeated I/O across adjacent requests.",
    "What information should remain resident when future computation is uncertain?",
    "Explain why parameter count is not the same thing as bytes moved through a hierarchy.",
    "Give an example of a compiler optimization that changes layout but not semantics.",
    "What is the difference between semantic similarity and computational co-demand?",
    "Explain a rate-distortion tradeoff and how it interacts with a cache.",
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
def capture_intermediate(
    model: nn.Module,
    down: nn.Linear,
    encoded: dict[str, Tensor],
) -> Tensor:
    rows: list[Tensor] = []

    def prehook(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
        rows.append(args[0].detach().cpu())

    handle = down.register_forward_pre_hook(prehook)
    try:
        model(**encoded)
    finally:
        handle.remove()

    if len(rows) != 1:
        raise RuntimeError("expected one down-projection invocation")
    return rows[0].float()


def topk_mask(scores: Tensor, fraction: float) -> Tensor:
    width = scores.shape[-1]
    k = max(1, min(width, round(width * fraction)))
    indices = torch.topk(scores, k=k, dim=-1, sorted=False).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(-1, indices, True)
    return mask


def adjacent_metrics(mask: Tensor, valid: Tensor) -> dict[str, float]:
    previous = mask[:, :-1]
    current = mask[:, 1:]
    pair_valid = valid[:, :-1] & valid[:, 1:]

    intersection = (previous & current).sum(dim=-1).float()
    union = (previous | current).sum(dim=-1).float().clamp_min(1)
    current_size = current.sum(dim=-1).float().clamp_min(1)
    new = (current & ~previous).sum(dim=-1).float()

    retention = intersection / current_size
    jaccard = intersection / union
    new_fraction_of_active = new / current_size
    new_fraction_of_width = new / mask.shape[-1]

    def selected(values: Tensor) -> Tensor:
        return values[pair_valid]

    return {
        "pairs": int(pair_valid.sum().item()),
        "mean_retention": float(selected(retention).mean().item()),
        "p05_retention": float(torch.quantile(selected(retention), 0.05).item()),
        "mean_jaccard": float(selected(jaccard).mean().item()),
        "mean_new_fraction_of_active": float(
            selected(new_fraction_of_active).mean().item()
        ),
        "mean_new_fraction_of_width": float(
            selected(new_fraction_of_width).mean().item()
        ),
        "p95_new_fraction_of_width": float(
            torch.quantile(selected(new_fraction_of_width), 0.95).item()
        ),
    }


def page_request_mask(
    neuron_mask: Tensor,
    permutation: Tensor,
    *,
    units_per_page: int,
) -> Tensor:
    if units_per_page < 1:
        raise ValueError("units_per_page must be positive")

    reordered = neuron_mask[..., permutation]
    width = reordered.shape[-1]
    pages = (width + units_per_page - 1) // units_per_page
    padded_width = pages * units_per_page
    if padded_width != width:
        pad = torch.zeros(
            *reordered.shape[:-1],
            padded_width - width,
            dtype=torch.bool,
        )
        reordered = torch.cat([reordered, pad], dim=-1)
    return reordered.reshape(
        *reordered.shape[:-1],
        pages,
        units_per_page,
    ).any(dim=-1)


def lru_fault_metrics(
    page_requests: Tensor,
    valid: Tensor,
    *,
    cache_pages: int,
) -> dict[str, float]:
    if cache_pages < 1:
        raise ValueError("cache_pages must be >= 1")

    total_requested = 0
    total_faults = 0
    faults_per_token = []

    for batch in range(page_requests.shape[0]):
        cache: list[int] = []
        for token in range(page_requests.shape[1]):
            if not bool(valid[batch, token]):
                continue
            requested = torch.nonzero(
                page_requests[batch, token],
                as_tuple=False,
            ).flatten().tolist()
            token_faults = 0
            for page in requested:
                total_requested += 1
                if page in cache:
                    cache.remove(page)
                    cache.append(page)
                    continue
                total_faults += 1
                token_faults += 1
                if len(cache) >= cache_pages:
                    cache.pop(0)
                cache.append(page)
            faults_per_token.append(token_faults)

    fault_tensor = torch.tensor(faults_per_token, dtype=torch.float32)
    return {
        "total_requested_page_uses": total_requested,
        "total_page_faults": total_faults,
        "faults_over_requests": total_faults / max(total_requested, 1),
        "mean_faults_per_token": float(fault_tensor.mean().item()),
        "p95_faults_per_token": float(torch.quantile(fault_tensor, 0.95).item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--text-file")
    parser.add_argument("--max-length", type=int, default=48)
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
        default=[0, 12, 23],
    )
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=[0.10, 0.25, 0.50],
    )
    parser.add_argument("--units-per-page", type=int, default=32)
    parser.add_argument(
        "--cache-fractions",
        nargs="+",
        type=float,
        default=[0.25, 0.50, 0.75],
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
    valid = encoded["attention_mask"].bool().cpu()

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    layers = resolve_layers(model)
    if any(idx < 0 or idx >= len(layers) for idx in args.layers):
        raise ValueError("layer index out of range")

    rows = []
    for layer_index in args.layers:
        down = resolve_down(layers[layer_index])
        intermediate = capture_intermediate(model, down, encoded)
        column_norm = down.weight.detach().float().cpu().norm(dim=0)
        score = intermediate.abs() * column_norm

        flattened = intermediate[valid]
        permutation = learn_simhash_codemand_permutation(
            flattened,
            down.cpu(),
            bits=16,
            max_trace_tokens=2048,
            seed=layer_index + 17,
        )
        model.to(device)

        for fraction in args.fractions:
            neurons = topk_mask(score, fraction)
            temporal = adjacent_metrics(neurons, valid)
            pages = page_request_mask(
                neurons,
                permutation,
                units_per_page=args.units_per_page,
            )
            num_pages = pages.shape[-1]
            cache_results = {}
            for cache_fraction in args.cache_fractions:
                capacity = max(1, round(num_pages * cache_fraction))
                cache_results[str(cache_fraction)] = {
                    "cache_pages": capacity,
                    **lru_fault_metrics(
                        pages,
                        valid,
                        cache_pages=capacity,
                    ),
                }

            rows.append(
                {
                    "layer": layer_index,
                    "active_neuron_fraction": fraction,
                    "num_pages": num_pages,
                    "units_per_page": args.units_per_page,
                    **temporal,
                    "lru": cache_results,
                }
            )

    payload = {
        "experiment": "hf_temporal_locality_v0",
        "model": args.model,
        "rows": rows,
        "claim_boundary": (
            "Active sets are defined by exact post-gate activations, so this "
            "measures real temporal locality but not yet a deployable prefetch "
            "policy. LRU page faults are simulated over token order."
        ),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        Path(args.json_out).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
