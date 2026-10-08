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


@torch.inference_mode()
def last_logits(model: nn.Module, encoded: dict[str, Tensor]) -> Tensor:
    return model(**encoded).logits[:, -1, :].detach().float().cpu()


def per_example_kl(reference: Tensor, candidate: Tensor) -> Tensor:
    ref_logp = torch.log_softmax(reference, dim=-1)
    cand_logp = torch.log_softmax(candidate, dim=-1)
    p = ref_logp.exp()
    return (p * (ref_logp - cand_logp)).sum(dim=-1)


def summarize(reference: Tensor, candidate: Tensor) -> dict[str, float]:
    kl = per_example_kl(reference, candidate)
    agreement = (reference.argmax(dim=-1) == candidate.argmax(dim=-1)).float()
    return {
        "mean_kl_nats": float(kl.mean().item()),
        "p95_kl_nats": float(torch.quantile(kl, 0.95).item()),
        "max_kl_nats": float(kl.max().item()),
        "top1_agreement": float(agreement.mean().item()),
    }


def zero_last_token_hook(active_skip: Tensor | None = None):
    def hook(
        _module: nn.Module,
        _args: tuple[Tensor, ...],
        output: Tensor,
    ) -> Tensor:
        replaced = output.clone()
        if active_skip is None:
            replaced[:, -1, :] = 0
        else:
            mask = active_skip.to(output.device)
            replaced[mask, -1, :] = 0
        return replaced

    return hook


def run_with_skip_matrix(
    model: nn.Module,
    layers: list[nn.Module],
    encoded: dict[str, Tensor],
    skip: Tensor,
) -> Tensor:
    if skip.shape != (encoded["input_ids"].shape[0], len(layers)):
        raise ValueError("skip matrix must have shape [batch, layers]")

    handles = []
    for layer_idx, layer in enumerate(layers):
        handles.append(
            resolve_mlp(layer).register_forward_hook(
                zero_last_token_hook(skip[:, layer_idx])
            )
        )
    try:
        return last_logits(model, encoded)
    finally:
        for handle in handles:
            handle.remove()


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
        "--retain-fractions",
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

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    layers = resolve_layers(model)
    baseline = last_logits(model, encoded)

    importance = []
    leave_one_out = []
    for layer_idx, layer in enumerate(layers):
        handle = resolve_mlp(layer).register_forward_hook(zero_last_token_hook())
        try:
            candidate = last_logits(model, encoded)
        finally:
            handle.remove()
        kl = per_example_kl(baseline, candidate)
        importance.append(kl)
        leave_one_out.append(
            {
                "layer": layer_idx,
                **summarize(baseline, candidate),
            }
        )

    importance_matrix = torch.stack(importance, dim=-1)
    rows = []
    for fraction in args.retain_fractions:
        retain_count = max(1, min(len(layers), round(len(layers) * fraction)))

        oracle_keep = torch.topk(
            importance_matrix,
            k=retain_count,
            dim=-1,
            largest=True,
        ).indices
        oracle_skip = torch.ones_like(importance_matrix, dtype=torch.bool)
        oracle_skip.scatter_(1, oracle_keep, False)
        oracle_candidate = run_with_skip_matrix(
            model,
            layers,
            encoded,
            oracle_skip,
        )

        global_score = importance_matrix.mean(dim=0)
        global_keep = torch.topk(
            global_score,
            k=retain_count,
            largest=True,
        ).indices
        global_skip = torch.ones_like(importance_matrix, dtype=torch.bool)
        global_skip[:, global_keep] = False
        global_candidate = run_with_skip_matrix(
            model,
            layers,
            encoded,
            global_skip,
        )

        rows.append(
            {
                "retain_fraction": fraction,
                "retain_layers": retain_count,
                "skip_fraction": 1.0 - retain_count / len(layers),
                "oracle_per_token": summarize(baseline, oracle_candidate),
                "global_static": summarize(baseline, global_candidate),
                "global_kept_layers": global_keep.tolist(),
            }
        )

    payload = {
        "experiment": "hf_vertical_sparsity_v0",
        "model": args.model,
        "num_layers": len(layers),
        "leave_one_out": leave_one_out,
        "rows": rows,
        "claim_boundary": (
            "Per-token oracle routing uses leave-one-MLP-out final-logit KL, "
            "which is unavailable without extra prediction machinery. Only the "
            "current decode token's MLP output is skipped; prior context remains "
            "dense. This is a behavioral sparsity probe, not a speed claim."
        ),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        Path(args.json_out).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
