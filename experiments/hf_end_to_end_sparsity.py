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
def last_logits(model: nn.Module, encoded: dict[str, Tensor]) -> Tensor:
    logits = model(**encoded).logits[:, -1, :]
    return logits.detach().float().cpu()


def distribution_metrics(reference: Tensor, candidate: Tensor) -> dict[str, float]:
    ref_logp = torch.log_softmax(reference, dim=-1)
    cand_logp = torch.log_softmax(candidate, dim=-1)
    p = ref_logp.exp()
    kl = (p * (ref_logp - cand_logp)).sum(dim=-1)

    ref_top = reference.argmax(dim=-1)
    cand_top = candidate.argmax(dim=-1)
    agreement = (ref_top == cand_top).float()

    ref_top_logp = torch.gather(ref_logp, 1, ref_top[:, None]).squeeze(1)
    cand_ref_top_logp = torch.gather(cand_logp, 1, ref_top[:, None]).squeeze(1)
    top_logp_drop = ref_top_logp - cand_ref_top_logp

    return {
        "mean_kl_nats": float(kl.mean().item()),
        "p95_kl_nats": float(torch.quantile(kl, 0.95).item()),
        "max_kl_nats": float(kl.max().item()),
        "top1_agreement": float(agreement.mean().item()),
        "mean_reference_top1_logp_drop": float(top_logp_drop.mean().item()),
        "p95_reference_top1_logp_drop": float(
            torch.quantile(top_logp_drop, 0.95).item()
        ),
    }


def install_topk_down_hook(
    down: nn.Linear,
    *,
    fraction: float,
):
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")

    column_norm = down.weight.detach().float().norm(dim=0)

    def prehook(_module: nn.Module, args: tuple[Tensor, ...]):
        activation = args[0]
        width = activation.shape[-1]
        k = max(1, min(width, round(width * fraction)))
        score = activation.detach().float().abs() * column_norm.to(activation.device)
        selected = torch.topk(score, k=k, dim=-1, sorted=False).indices
        sparse = torch.zeros_like(activation)
        values = torch.gather(activation, -1, selected)
        sparse.scatter_(-1, selected, values)
        return (sparse,)

    return down.register_forward_pre_hook(prehook)


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
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=[0, 6, 12, 18, 23],
    )
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=[0.10, 0.25, 0.50],
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
    if any(idx < 0 or idx >= len(layers) for idx in args.layers):
        raise ValueError("layer index out of range")

    baseline = last_logits(model, encoded)

    rows = []
    for layer_index in args.layers:
        down = resolve_down(layers[layer_index])
        for fraction in args.fractions:
            handle = install_topk_down_hook(down, fraction=fraction)
            try:
                candidate = last_logits(model, encoded)
            finally:
                handle.remove()

            rows.append(
                {
                    "layer": layer_index,
                    "active_neuron_fraction": fraction,
                    **distribution_metrics(baseline, candidate),
                }
            )

    payload = {
        "experiment": "hf_end_to_end_sparsity_v0",
        "model": args.model,
        "layers": args.layers,
        "fractions": args.fractions,
        "rows": rows,
        "claim_boundary": (
            "Top-k selection uses the full SwiGLU intermediate activation, so "
            "this is a behavioral sparsity oracle for the down projection, not "
            "yet a deployable bandwidth-saving route. Metrics compare final "
            "next-token logits against the untouched dense model."
        ),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        Path(args.json_out).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
