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
    "Explain how branch prediction differs from semantic routing.",
    "Why can a large model have a small active working set?",
    "Describe a compiler pass that improves memory locality.",
    "How can held-out calibration prevent a routing policy from overclaiming?",
    "What does posterior concentration mean for future computation?",
    "Explain why a cache should track expected value per byte.",
    "How can repeated workloads create reusable computational trajectories?",
    "Describe the difference between a hot interpreter and cold program memory.",
    "Why is activation magnitude not necessarily predictive importance?",
    "What would falsify a neural virtual-memory hypothesis?",
    "Explain the role of a page table in an operating system.",
    "How could a neural model learn an address for cold weights?",
    "Why can distributed representations make neuron paging look dense?",
    "Describe prediction-conditioned sparsity.",
    "How can low-bit metadata be useful for routing without being useful for generation?",
    "Explain why final-logit behavior can tolerate local MLP approximation error.",
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


def act_fn(mlp: nn.Module):
    fn = getattr(mlp, "act_fn", None)
    return fn if callable(fn) else torch.nn.functional.silu


def quantized_row_sketch(weight: Tensor, bits: int) -> tuple[Tensor, Tensor]:
    if bits not in (1, 2):
        raise ValueError("bits must be 1 or 2")
    w = weight.detach().float()
    if bits == 1:
        scale = w.abs().mean(dim=1).clamp_min(1e-12)
        code = torch.where(w >= 0, torch.ones_like(w), -torch.ones_like(w))
        return code, scale

    # Ternary signed code: {-1, 0, 1}, stored under a 2-bit budget.
    scale = w.abs().mean(dim=1).clamp_min(1e-12)
    for _ in range(3):
        code = torch.round(w / scale[:, None]).clamp(-1, 1)
        numerator = (w * code).sum(dim=1)
        denominator = code.pow(2).sum(dim=1).clamp_min(1e-12)
        scale = (numerator / denominator).abs().clamp_min(1e-12)
    code = torch.round(w / scale[:, None]).clamp(-1, 1)
    return code, scale


def coarse_scores(hidden: Tensor, mlp: nn.Module, bits: int) -> Tensor:
    gate_code, gate_scale = quantized_row_sketch(mlp.gate_proj.weight, bits)
    up_code, up_scale = quantized_row_sketch(mlp.up_proj.weight, bits)
    gate = (hidden.float() @ gate_code.T.to(hidden.device)) * gate_scale.to(hidden.device)
    up = (hidden.float() @ up_code.T.to(hidden.device)) * up_scale.to(hidden.device)
    activation = act_fn(mlp)(gate) * up
    down_norm = mlp.down_proj.weight.detach().float().norm(dim=0).to(hidden.device)
    return activation.abs() * down_norm[None, :]


def cascaded_sparse_output(
    hidden: Tensor,
    mlp: nn.Module,
    *,
    bits: int,
    candidate_fraction: float,
    final_fraction: float,
) -> Tensor:
    width = mlp.down_proj.in_features
    candidate_k = max(1, min(width, round(width * candidate_fraction)))
    final_k = max(1, min(candidate_k, round(width * final_fraction)))

    candidate_score = coarse_scores(hidden, mlp, bits)
    candidate = torch.topk(
        candidate_score,
        k=candidate_k,
        dim=-1,
        sorted=False,
    ).indices
    down_norm = mlp.down_proj.weight.detach().float().norm(dim=0).to(hidden.device)
    fn = act_fn(mlp)

    rows = []
    for row in range(hidden.shape[0]):
        x = hidden[row]
        candidate_idx = candidate[row]

        gate = torch.nn.functional.linear(
            x,
            mlp.gate_proj.weight[candidate_idx],
            (
                mlp.gate_proj.bias[candidate_idx]
                if mlp.gate_proj.bias is not None
                else None
            ),
        )
        up = torch.nn.functional.linear(
            x,
            mlp.up_proj.weight[candidate_idx],
            (
                mlp.up_proj.bias[candidate_idx]
                if mlp.up_proj.bias is not None
                else None
            ),
        )
        candidate_activation = fn(gate) * up
        exact_score = candidate_activation.float().abs() * down_norm[candidate_idx]
        within = torch.topk(
            exact_score,
            k=final_k,
            dim=-1,
            sorted=False,
        ).indices
        final_idx = candidate_idx[within]
        final_activation = candidate_activation[within]

        rows.append(
            torch.nn.functional.linear(
                final_activation,
                mlp.down_proj.weight[:, final_idx],
                mlp.down_proj.bias,
            )
        )

    return torch.stack(rows)


@torch.inference_mode()
def last_logits(model: nn.Module, encoded: dict[str, Tensor]) -> Tensor:
    return model(**encoded).logits[:, -1, :].detach().float().cpu()


def quality(reference: Tensor, candidate: Tensor) -> dict[str, float]:
    ref_logp = torch.log_softmax(reference, dim=-1)
    cand_logp = torch.log_softmax(candidate, dim=-1)
    p = ref_logp.exp()
    kl = (p * (ref_logp - cand_logp)).sum(dim=-1)
    agreement = (reference.argmax(dim=-1) == candidate.argmax(dim=-1)).float()
    return {
        "mean_kl_nats": float(kl.mean().item()),
        "p95_kl_nats": float(torch.quantile(kl, 0.95).item()),
        "max_kl_nats": float(kl.max().item()),
        "top1_agreement": float(agreement.mean().item()),
    }


def byte_fraction(
    mlp: nn.Module,
    *,
    bits: int,
    candidate_fraction: float,
    final_fraction: float,
) -> dict[str, float]:
    gate = mlp.gate_proj
    up = mlp.up_proj
    down = mlp.down_proj

    gate_params = gate.weight.numel()
    up_params = up.weight.numel()
    down_params = down.weight.numel()
    cold_params = gate_params + up_params + down_params
    cold_bytes = cold_params * 2

    packed_bits = bits * (gate_params + up_params)
    metadata_bytes = (packed_bits + 7) // 8
    metadata_bytes += down.in_features * 3 * 2

    # Candidate refinement reads exact gate and up rows only.
    candidate_bytes = candidate_fraction * (gate_params + up_params) * 2
    # The final down columns are fetched only after refinement.
    final_down_bytes = final_fraction * down_params * 2

    total = metadata_bytes + candidate_bytes + final_down_bytes
    return {
        "address_metadata_fraction": metadata_bytes / cold_bytes,
        "candidate_gate_up_fraction": candidate_bytes / cold_bytes,
        "final_down_fraction": final_down_bytes / cold_bytes,
        "analytic_total_fraction": total / cold_bytes,
        "analytic_cold_byte_reduction_fraction": 1.0 - total / cold_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--text-file")
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--bits", nargs="+", type=int, default=[1, 2])
    parser.add_argument(
        "--candidate-fractions",
        nargs="+",
        type=float,
        default=[0.15, 0.20, 0.25, 0.30],
    )
    parser.add_argument(
        "--final-fractions",
        nargs="+",
        type=float,
        default=[0.05, 0.075, 0.10, 0.125],
    )
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
    mlp = resolve_mlp(layers[args.layer])
    baseline = last_logits(model, encoded)

    rows = []
    for bits in args.bits:
        for candidate_fraction in args.candidate_fractions:
            for final_fraction in args.final_fractions:
                if final_fraction > candidate_fraction:
                    continue

                def hook(
                    _module: nn.Module,
                    hook_args: tuple[Tensor, ...],
                    output: Tensor,
                ) -> Tensor:
                    hidden = hook_args[0][:, -1, :]
                    sparse = cascaded_sparse_output(
                        hidden,
                        mlp,
                        bits=bits,
                        candidate_fraction=candidate_fraction,
                        final_fraction=final_fraction,
                    )
                    replaced = output.clone()
                    replaced[:, -1, :] = sparse.to(replaced.dtype)
                    return replaced

                handle = mlp.register_forward_hook(hook)
                try:
                    candidate = last_logits(model, encoded)
                finally:
                    handle.remove()

                rows.append(
                    {
                        "bits": bits,
                        "candidate_fraction": candidate_fraction,
                        "final_fraction": final_fraction,
                        **byte_fraction(
                            mlp,
                            bits=bits,
                            candidate_fraction=candidate_fraction,
                            final_fraction=final_fraction,
                        ),
                        **quality(baseline, candidate),
                    }
                )

    payload = {
        "experiment": "hf_cascaded_address_v0",
        "model": args.model,
        "layer": args.layer,
        "rows": rows,
        "claim_boundary": (
            "A low-bit resident sketch proposes a coarse neuron shortlist. Exact "
            "gate/up rows are then fetched only for that shortlist and reused to "
            "choose the final down-projection columns. Byte fractions are analytic; "
            "the diagnostic hook still executes the dense MLP before replacement."
        ),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        Path(args.json_out).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
