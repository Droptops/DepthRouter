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
    return {
        "mean_kl_nats": float(kl.mean().item()),
        "p95_kl_nats": float(torch.quantile(kl, 0.95).item()),
        "max_kl_nats": float(kl.max().item()),
        "top1_agreement": float(agreement.mean().item()),
    }


def quantized_rows(weight: Tensor, *, bits: int) -> tuple[Tensor, Tensor]:
    if bits not in (2, 3, 4):
        raise ValueError("bits must be one of {2, 3, 4}")
    w = weight.detach().float().cpu()
    qmax = (1 << (bits - 1)) - 1
    scale = (w.abs().amax(dim=1) / qmax).clamp_min(1e-12)

    for _ in range(4):
        code = torch.round(w / scale[:, None]).clamp(-qmax, qmax)
        numerator = (w * code).sum(dim=1)
        denominator = code.square().sum(dim=1).clamp_min(1e-12)
        scale = (numerator / denominator).abs().clamp_min(1e-12)

    code = torch.round(w / scale[:, None]).clamp(-qmax, qmax)
    return code, scale


class QuantizedMLPAddress:
    def __init__(self, mlp: nn.Module, bits: int) -> None:
        self.bits = bits

        gate_code, gate_scale = quantized_rows(mlp.gate_proj.weight, bits=bits)
        up_code, up_scale = quantized_rows(mlp.up_proj.weight, bits=bits)
        # Quantize W_down by neuron/column, hence transpose first.
        down_code_t, down_scale = quantized_rows(
            mlp.down_proj.weight.T,
            bits=bits,
        )

        self.gate = gate_code * gate_scale[:, None]
        self.up = up_code * up_scale[:, None]
        self.down = (down_code_t * down_scale[:, None]).T
        self.down_norm_sq = self.down.square().sum(dim=0)


def act_fn(mlp: nn.Module):
    fn = getattr(mlp, "act_fn", None)
    return fn if callable(fn) else torch.nn.functional.silu


def approximate_activation(
    hidden: Tensor,
    mlp: nn.Module,
    address: QuantizedMLPAddress,
) -> Tensor:
    gate = torch.nn.functional.linear(
        hidden.float(),
        address.gate.to(hidden.device),
        bias=None,
    )
    up = torch.nn.functional.linear(
        hidden.float(),
        address.up.to(hidden.device),
        bias=None,
    )
    return act_fn(mlp)(gate) * up


def residual_pursuit(
    activation: Tensor,
    address: QuantizedMLPAddress,
    *,
    fraction: float,
    block_size: int,
) -> Tensor:
    """Greedy pursuit using only low-bit resident approximations."""

    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if block_size < 1:
        raise ValueError("block_size must be >= 1")

    down = address.down.to(activation.device)
    norm_sq = address.down_norm_sq.to(activation.device)
    target = activation.float() @ down.T
    residual = target.clone()

    batch, width = activation.shape
    k = max(1, min(width, round(width * fraction)))
    selected_mask = torch.zeros(batch, width, dtype=torch.bool, device=activation.device)
    selected_chunks: list[Tensor] = []

    chosen = 0
    while chosen < k:
        take = min(block_size, k - chosen)
        dot = residual @ down
        gain = (
            2.0 * activation.float() * dot
            - activation.float().square() * norm_sq[None, :]
        )
        gain = gain.masked_fill(selected_mask, float("-inf"))
        indices = torch.topk(gain, k=take, dim=-1, sorted=False).indices
        selected_mask.scatter_(1, indices, True)
        selected_chunks.append(indices)

        coeff = torch.gather(activation.float(), 1, indices)
        vectors = down.T[indices]
        residual = residual - (coeff[..., None] * vectors).sum(dim=1)
        chosen += take

    return torch.cat(selected_chunks, dim=1)


def norm_topk(
    activation: Tensor,
    address: QuantizedMLPAddress,
    *,
    fraction: float,
) -> Tensor:
    width = activation.shape[-1]
    k = max(1, min(width, round(width * fraction)))
    score = activation.float().abs() * address.down_norm_sq.sqrt().to(activation.device)
    return torch.topk(score, k=k, dim=-1, sorted=False).indices


def sparse_exact_output(
    hidden: Tensor,
    mlp: nn.Module,
    selected: Tensor,
) -> Tensor:
    rows = []
    for row in range(hidden.shape[0]):
        index = selected[row]
        x = hidden[row]
        gate_bias = mlp.gate_proj.bias[index] if mlp.gate_proj.bias is not None else None
        up_bias = mlp.up_proj.bias[index] if mlp.up_proj.bias is not None else None
        gate = torch.nn.functional.linear(
            x,
            mlp.gate_proj.weight[index],
            gate_bias,
        )
        up = torch.nn.functional.linear(
            x,
            mlp.up_proj.weight[index],
            up_bias,
        )
        activation = act_fn(mlp)(gate) * up
        output = torch.nn.functional.linear(
            activation,
            mlp.down_proj.weight[:, index],
            mlp.down_proj.bias,
        )
        rows.append(output)
    return torch.stack(rows)


def install_hook(
    mlp: nn.Module,
    address: QuantizedMLPAddress,
    *,
    fraction: float,
    block_size: int,
    mode: str,
):
    def hook(
        _module: nn.Module,
        args: tuple[Tensor, ...],
        output: Tensor,
    ) -> Tensor:
        hidden = args[0][:, -1, :]
        approx = approximate_activation(hidden, mlp, address)
        if mode == "pursuit":
            selected = residual_pursuit(
                approx,
                address,
                fraction=fraction,
                block_size=block_size,
            )
        elif mode == "topk":
            selected = norm_topk(
                approx,
                address,
                fraction=fraction,
            )
        else:
            raise ValueError(f"unknown mode: {mode}")

        sparse = sparse_exact_output(hidden, mlp, selected)
        replaced = output.clone()
        replaced[:, -1, :] = sparse.to(replaced.dtype)
        return replaced

    return mlp.register_forward_hook(hook)


def analytic_fraction(mlp: nn.Module, *, bits: int, selected_fraction: float) -> float:
    full_parameters = (
        mlp.gate_proj.weight.numel()
        + mlp.up_proj.weight.numel()
        + mlp.down_proj.weight.numel()
    )
    cold_bytes = full_parameters * 2

    # Entire MLP exists as a packed low-bit address model, plus one fp16 scale
    # per gate row, up row, and down column.
    packed_bits = bits * full_parameters
    scale_count = (
        mlp.gate_proj.out_features
        + mlp.up_proj.out_features
        + mlp.down_proj.in_features
    )
    address_bytes = (packed_bits + 7) // 8 + scale_count * 2

    width = mlp.down_proj.in_features
    selected = max(1, min(width, round(width * selected_fraction)))
    exact_parameters = selected * mlp.gate_proj.in_features * 3
    exact_bytes = exact_parameters * 2
    return (address_bytes + exact_bytes) / cold_bytes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--text-file")
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--address-bits", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=[0.03, 0.07, 0.17],
    )
    parser.add_argument("--block-size", type=int, default=32)
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
    mlp = resolve_mlp(layers[args.layer])
    baseline = last_logits(model, encoded)

    rows = []
    for bits in args.address_bits:
        address = QuantizedMLPAddress(mlp, bits)
        for fraction in args.fractions:
            for mode in ("topk", "pursuit"):
                handle = install_hook(
                    mlp,
                    address,
                    fraction=fraction,
                    block_size=args.block_size,
                    mode=mode,
                )
                try:
                    candidate = last_logits(model, encoded)
                finally:
                    handle.remove()

                rows.append(
                    {
                        "layer": args.layer,
                        "address_bits": bits,
                        "mode": mode,
                        "active_neuron_fraction": fraction,
                        "analytic_address_plus_exact_payload_fraction": analytic_fraction(
                            mlp,
                            bits=bits,
                            selected_fraction=fraction,
                        ),
                        **distribution_metrics(baseline, candidate),
                    }
                )

    payload = {
        "experiment": "hf_lowbit_residual_pursuit_v0",
        "model": args.model,
        "rows": rows,
        "claim_boundary": (
            "Selection uses a complete low-bit approximation of the MLP as "
            "resident address metadata, then exact weights only for selected "
            "neurons. PyTorch still executes the dense MLP before the diagnostic "
            "replacement hook, so the byte fraction is analytic until a selective "
            "kernel is profiled."
        ),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        Path(args.json_out).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
