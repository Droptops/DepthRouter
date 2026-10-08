# ruff: noqa: I001
from __future__ import annotations

import argparse
import json
from collections.abc import Callable
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


def metrics(reference: Tensor, candidate: Tensor) -> dict[str, float]:
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


def quantized_row_sketch(
    weight: Tensor,
    *,
    bits: int,
) -> tuple[Tensor, Tensor]:
    """Return dequantized integer codes plus one fp16 scale per row."""

    if bits not in (1, 2, 3, 4):
        raise ValueError("bits must be one of {1, 2, 3, 4}")

    w = weight.detach().float()
    if bits == 1:
        scale = w.abs().mean(dim=1).clamp_min(1e-12)
        code = torch.where(w >= 0, torch.ones_like(w), -torch.ones_like(w))
        return code, scale

    qmax = (1 << (bits - 1)) - 1
    if bits == 2:
        # Ternary initialization keeps moderate-magnitude weights instead of
        # letting one row outlier set the whole quantizer scale.
        scale = w.abs().mean(dim=1).clamp_min(1e-12)
    else:
        scale = (w.abs().amax(dim=1) / qmax).clamp_min(1e-12)

    # A few Lloyd-style least-squares refinements are cheap at compile time and
    # materially improve the address approximation without changing metadata.
    for _ in range(3):
        code = torch.round(w / scale[:, None]).clamp(-qmax, qmax)
        numerator = (w * code).sum(dim=1)
        denominator = code.pow(2).sum(dim=1).clamp_min(1e-12)
        scale = (numerator / denominator).abs().clamp_min(1e-12)

    code = torch.round(w / scale[:, None]).clamp(-qmax, qmax)
    return code, scale


def act_fn(mlp: nn.Module) -> Callable[[Tensor], Tensor]:
    fn = getattr(mlp, "act_fn", None)
    if callable(fn):
        return fn
    return torch.nn.functional.silu


def address_scores(
    hidden: Tensor,
    mlp: nn.Module,
    *,
    address_bits: int | None,
) -> tuple[Tensor, Tensor]:
    gate = mlp.gate_proj
    up = mlp.up_proj
    down = mlp.down_proj

    if address_bits is not None:
        gate_code, gate_scale = quantized_row_sketch(
            gate.weight,
            bits=address_bits,
        )
        up_code, up_scale = quantized_row_sketch(
            up.weight,
            bits=address_bits,
        )
        gate_hat = torch.matmul(hidden.float(), gate_code.T.to(hidden.device))
        up_hat = torch.matmul(hidden.float(), up_code.T.to(hidden.device))
        gate_hat = gate_hat * gate_scale.to(hidden.device)
        up_hat = up_hat * up_scale.to(hidden.device)
        activation = act_fn(mlp)(gate_hat) * up_hat
    else:
        gate_value = torch.nn.functional.linear(
            hidden,
            gate.weight,
            gate.bias,
        )
        up_value = torch.nn.functional.linear(
            hidden,
            up.weight,
            up.bias,
        )
        activation = act_fn(mlp)(gate_value) * up_value

    down_norm = down.weight.detach().float().norm(dim=0).to(hidden.device)
    score = activation.float().abs() * down_norm
    return score, activation


def sparse_exact_output(
    hidden: Tensor,
    mlp: nn.Module,
    selected: Tensor,
) -> Tensor:
    """Evaluate only selected exact SwiGLU neurons for each batch row."""

    gate = mlp.gate_proj
    up = mlp.up_proj
    down = mlp.down_proj
    activation_fn = act_fn(mlp)
    rows = []

    for row in range(hidden.shape[0]):
        indices = selected[row]
        x = hidden[row]
        gate_bias = gate.bias[indices] if gate.bias is not None else None
        up_bias = up.bias[indices] if up.bias is not None else None
        gate_value = torch.nn.functional.linear(
            x,
            gate.weight[indices],
            gate_bias,
        )
        up_value = torch.nn.functional.linear(
            x,
            up.weight[indices],
            up_bias,
        )
        activation = activation_fn(gate_value) * up_value
        y = torch.nn.functional.linear(
            activation,
            down.weight[:, indices],
            down.bias,
        )
        rows.append(y)

    return torch.stack(rows)


def install_sparse_last_token_hook(
    mlp: nn.Module,
    *,
    fraction: float,
    address_bits: int | None,
    overlap_sink: list[float],
):
    width = mlp.down_proj.in_features
    k = max(1, min(width, round(width * fraction)))

    def hook(
        _module: nn.Module,
        args: tuple[Tensor, ...],
        output: Tensor,
    ) -> Tensor:
        hidden = args[0][:, -1, :]
        predicted_score, _ = address_scores(
            hidden,
            mlp,
            address_bits=address_bits,
        )
        selected = torch.topk(
            predicted_score,
            k=k,
            dim=-1,
            sorted=False,
        ).indices

        if address_bits is not None:
            true_score, _ = address_scores(hidden, mlp, address_bits=None)
            true_selected = torch.topk(
                true_score,
                k=k,
                dim=-1,
                sorted=False,
            ).indices
            overlaps = []
            for row in range(hidden.shape[0]):
                predicted_set = set(selected[row].tolist())
                true_set = set(true_selected[row].tolist())
                overlaps.append(len(predicted_set & true_set) / k)
            overlap_sink.extend(overlaps)

        sparse = sparse_exact_output(hidden, mlp, selected)
        replaced = output.clone()
        replaced[:, -1, :] = sparse.to(replaced.dtype)
        return replaced

    return mlp.register_forward_hook(hook)


def analytic_bytes(
    mlp: nn.Module,
    fraction: float,
    *,
    address_bits: int,
) -> dict[str, float]:
    gate = mlp.gate_proj
    up = mlp.up_proj
    down = mlp.down_proj
    width = down.in_features
    hidden = gate.in_features
    selected = max(1, min(width, round(width * fraction)))

    # Compare everything at fp16/bf16 payload precision.
    full_parameters = (
        gate.weight.numel()
        + up.weight.numel()
        + down.weight.numel()
    )
    cold_bytes = full_parameters * 2

    # Quantized gate/up address weights plus fp16 row scales and down-column norms.
    packed_bits = address_bits * (
        gate.weight.numel() + up.weight.numel()
    )
    address_bytes = (packed_bits + 7) // 8
    address_bytes += width * 3 * 2

    selected_parameters = selected * hidden * 3
    selected_bytes = selected_parameters * 2

    return {
        "selected_neurons": selected,
        "cold_mlp_bytes_fp16": cold_bytes,
        "address_metadata_bytes": int(address_bytes),
        "address_metadata_fraction": address_bytes / cold_bytes,
        "selected_payload_bytes_fp16": selected_bytes,
        "selected_payload_fraction": selected_bytes / cold_bytes,
        "address_plus_selected_fraction": (
            address_bytes + selected_bytes
        )
        / cold_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--text-file")
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=[0.10, 0.25, 0.50],
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument(
        "--address-bits",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4],
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
    if any(bits not in (1, 2, 3, 4) for bits in args.address_bits):
        raise ValueError("--address-bits values must be in {1, 2, 3, 4}")

    for fraction in args.fractions:
        modes: list[int | None] = [None, *args.address_bits]
        for address_bits in modes:
            address_mode = "true" if address_bits is None else f"int{address_bits}"
            overlap: list[float] = []
            handle = install_sparse_last_token_hook(
                mlp,
                fraction=fraction,
                address_bits=address_bits,
                overlap_sink=overlap,
            )
            try:
                candidate = last_logits(model, encoded)
            finally:
                handle.remove()

            true_bytes = analytic_bytes(
                mlp,
                fraction,
                address_bits=1,
            )
            if address_bits is None:
                byte_stats = {
                    **true_bytes,
                    "address_metadata_bytes": 0,
                    "address_metadata_fraction": 0.0,
                    "address_plus_selected_fraction": true_bytes[
                        "selected_payload_fraction"
                    ],
                }
            else:
                byte_stats = analytic_bytes(
                    mlp,
                    fraction,
                    address_bits=address_bits,
                )

            rows.append(
                {
                    "layer": args.layer,
                    "fraction": fraction,
                    "address_mode": address_mode,
                    "mean_topk_overlap_with_true": (
                        sum(overlap) / len(overlap) if overlap else 1.0
                    ),
                    **byte_stats,
                    **metrics(baseline, candidate),
                }
            )

    payload = {
        "experiment": "hf_binary_address_v0",
        "model": args.model,
        "rows": rows,
        "claim_boundary": (
            "Low-bit addresses are evaluated from resident-style row-quantized gate/up sketches, "
            "then selected neurons use exact payload weights. PyTorch still "
            "executes the dense MLP before the diagnostic hook, so byte savings "
            "are analytic until a fused page-selective kernel is profiled."
        ),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        Path(args.json_out).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
