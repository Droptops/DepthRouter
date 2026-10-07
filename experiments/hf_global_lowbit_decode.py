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
def last_logits(model: nn.Module, encoded: dict[str, Tensor]) -> Tensor:
    return model(**encoded).logits[:, -1, :].detach().float().cpu()


def distribution_metrics(reference: Tensor, candidate: Tensor) -> dict[str, float]:
    ref_logp = torch.log_softmax(reference, dim=-1)
    cand_logp = torch.log_softmax(candidate, dim=-1)
    p = ref_logp.exp()
    kl = (p * (ref_logp - cand_logp)).sum(dim=-1)
    agreement = (reference.argmax(dim=-1) == candidate.argmax(dim=-1)).float()

    ref_top = reference.argmax(dim=-1)
    ref_top_logp = torch.gather(ref_logp, 1, ref_top[:, None]).squeeze(1)
    cand_ref_top_logp = torch.gather(cand_logp, 1, ref_top[:, None]).squeeze(1)
    drop = ref_top_logp - cand_ref_top_logp

    return {
        "mean_kl_nats": float(kl.mean().item()),
        "p95_kl_nats": float(torch.quantile(kl, 0.95).item()),
        "max_kl_nats": float(kl.max().item()),
        "top1_agreement": float(agreement.mean().item()),
        "mean_reference_top1_logp_drop": float(drop.mean().item()),
        "p95_reference_top1_logp_drop": float(torch.quantile(drop, 0.95).item()),
    }


def quantized_weight(weight: Tensor, *, bits: int) -> tuple[Tensor, Tensor]:
    if bits not in (1, 2, 3, 4):
        raise ValueError("bits must be one of {1, 2, 3, 4}")
    w = weight.detach().float().cpu()

    if bits == 1:
        scale = w.abs().mean(dim=1).clamp_min(1e-12)
        code = torch.where(w >= 0, torch.ones_like(w), -torch.ones_like(w))
        return code, scale

    qmax = (1 << (bits - 1)) - 1
    if bits == 2:
        scale = w.abs().mean(dim=1).clamp_min(1e-12)
    else:
        scale = (w.abs().amax(dim=1) / qmax).clamp_min(1e-12)

    for _ in range(3):
        code = torch.round(w / scale[:, None]).clamp(-qmax, qmax)
        numerator = (w * code).sum(dim=1)
        denominator = code.pow(2).sum(dim=1).clamp_min(1e-12)
        scale = (numerator / denominator).abs().clamp_min(1e-12)

    code = torch.round(w / scale[:, None]).clamp(-qmax, qmax)
    return code, scale


def act_fn(mlp: nn.Module):
    fn = getattr(mlp, "act_fn", None)
    return fn if callable(fn) else torch.nn.functional.silu


class LowBitAddress:
    def __init__(self, mlp: nn.Module, bits: int) -> None:
        self.bits = bits
        gate_code, gate_scale = quantized_weight(
            mlp.gate_proj.weight,
            bits=bits,
        )
        up_code, up_scale = quantized_weight(
            mlp.up_proj.weight,
            bits=bits,
        )
        # Store the diagnostic reconstruction in fp32. Byte accounting below is
        # for the packed representation the runtime would actually keep.
        self.gate_hat = gate_code * gate_scale[:, None]
        self.up_hat = up_code * up_scale[:, None]
        self.down_norm = mlp.down_proj.weight.detach().float().cpu().norm(dim=0)


def sparse_exact_last_token(
    hidden: Tensor,
    mlp: nn.Module,
    address: LowBitAddress,
    *,
    fraction: float,
) -> Tensor:
    x = hidden[:, -1, :]
    gate_hat = torch.nn.functional.linear(
        x.float(),
        address.gate_hat.to(x.device),
        bias=None,
    )
    up_hat = torch.nn.functional.linear(
        x.float(),
        address.up_hat.to(x.device),
        bias=None,
    )
    approx_activation = act_fn(mlp)(gate_hat) * up_hat
    score = approx_activation.abs() * address.down_norm.to(x.device)[None, :]

    width = score.shape[-1]
    k = max(1, min(width, round(width * fraction)))
    selected = torch.topk(score, k=k, dim=-1, sorted=False).indices

    rows = []
    for row in range(x.shape[0]):
        index = selected[row]
        gate_bias = (
            mlp.gate_proj.bias[index]
            if mlp.gate_proj.bias is not None
            else None
        )
        up_bias = (
            mlp.up_proj.bias[index]
            if mlp.up_proj.bias is not None
            else None
        )
        gate = torch.nn.functional.linear(
            x[row],
            mlp.gate_proj.weight[index],
            gate_bias,
        )
        up = torch.nn.functional.linear(
            x[row],
            mlp.up_proj.weight[index],
            up_bias,
        )
        activation = act_fn(mlp)(gate) * up
        y = torch.nn.functional.linear(
            activation,
            mlp.down_proj.weight[:, index],
            mlp.down_proj.bias,
        )
        rows.append(y)
    return torch.stack(rows)


def install_hook(
    mlp: nn.Module,
    address: LowBitAddress,
    *,
    fraction: float,
):
    def hook(
        _module: nn.Module,
        args: tuple[Tensor, ...],
        output: Tensor,
    ) -> Tensor:
        sparse = sparse_exact_last_token(
            args[0],
            mlp,
            address,
            fraction=fraction,
        )
        replaced = output.clone()
        replaced[:, -1, :] = sparse.to(replaced.dtype)
        return replaced

    return mlp.register_forward_hook(hook)


def analytic_fraction(mlp: nn.Module, *, bits: int, selected_fraction: float) -> float:
    gate = mlp.gate_proj
    up = mlp.up_proj
    down = mlp.down_proj

    full_parameters = (
        gate.weight.numel()
        + up.weight.numel()
        + down.weight.numel()
    )
    cold_bytes = full_parameters * 2

    packed_bits = bits * (gate.weight.numel() + up.weight.numel())
    address_bytes = (packed_bits + 7) // 8
    width = down.in_features
    address_bytes += width * 3 * 2

    selected = max(1, min(width, round(width * selected_fraction)))
    selected_parameters = selected * gate.in_features * 3
    selected_bytes = selected_parameters * 2

    return (address_bytes + selected_bytes) / cold_bytes


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
    parser.add_argument("--address-bits", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.10, 0.25])
    parser.add_argument("--json-out")
    args = parser.parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit('Install with: pip install -e ".[hf]"') from exc

    if any(bits not in (1, 2, 4) for bits in args.address_bits):
        raise ValueError("address bits must be in {1, 2, 3, 4}")

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
    mlps = [resolve_mlp(layer) for layer in layers]
    baseline = last_logits(model, encoded)

    rows = []
    for bits in args.address_bits:
        addresses = [LowBitAddress(mlp, bits) for mlp in mlps]
        for fraction in args.fractions:
            handles = [
                install_hook(mlp, address, fraction=fraction)
                for mlp, address in zip(mlps, addresses, strict=True)
            ]
            try:
                candidate = last_logits(model, encoded)
            finally:
                for handle in handles:
                    handle.remove()

            layer_fraction = analytic_fraction(
                mlps[0],
                bits=bits,
                selected_fraction=fraction,
            )
            rows.append(
                {
                    "address_bits": bits,
                    "active_neuron_fraction": fraction,
                    "layers_sparsified": len(mlps),
                    "analytic_address_plus_exact_payload_fraction_per_mlp": layer_fraction,
                    "analytic_cold_byte_reduction_fraction": 1.0 - layer_fraction,
                    **distribution_metrics(baseline, candidate),
                }
            )

    payload = {
        "experiment": "hf_global_lowbit_decode_v0",
        "model": args.model,
        "num_layers": len(mlps),
        "rows": rows,
        "claim_boundary": (
            "Every MLP's current decode token is replaced by exact computation "
            "over neurons chosen by a low-bit approximation of gate/up weights. "
            "Earlier context tokens remain dense. PyTorch still executes the "
            "dense MLP before the replacement hook, so byte reductions are "
            "analytic until a fused selective kernel is profiled."
        ),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        Path(args.json_out).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
