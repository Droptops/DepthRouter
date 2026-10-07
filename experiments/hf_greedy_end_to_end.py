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


def residual_greedy_indices(
    activation: Tensor,
    down_weight: Tensor,
    *,
    fraction: float,
    block_size: int,
) -> Tensor:
    """Select exact original neuron contributions by residual reduction."""

    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if block_size < 1:
        raise ValueError("block_size must be >= 1")

    u = activation.float()
    weight = down_weight.float()
    target = u @ weight.T
    residual = target.clone()
    norm_sq = weight.square().sum(dim=0)

    batch, width = u.shape
    k = max(1, min(width, round(width * fraction)))
    selected = torch.zeros(batch, width, dtype=torch.bool, device=u.device)
    chunks: list[Tensor] = []

    chosen = 0
    while chosen < k:
        take = min(block_size, k - chosen)
        dot = residual @ weight
        gain = 2.0 * u * dot - u.square() * norm_sq[None, :]
        gain = gain.masked_fill(selected, float("-inf"))
        index = torch.topk(gain, k=take, dim=-1, sorted=False).indices
        selected.scatter_(1, index, True)
        chunks.append(index)

        coeff = torch.gather(u, 1, index)
        vectors = weight.T[index]
        residual = residual - (coeff[..., None] * vectors).sum(dim=1)
        chosen += take

    return torch.cat(chunks, dim=1)


def norm_topk_indices(
    activation: Tensor,
    down_weight: Tensor,
    *,
    fraction: float,
) -> Tensor:
    width = activation.shape[-1]
    k = max(1, min(width, round(width * fraction)))
    score = activation.float().abs() * down_weight.float().norm(dim=0)[None, :]
    return torch.topk(score, k=k, dim=-1, sorted=False).indices


def exact_output_from_selected(
    activation: Tensor,
    down: nn.Linear,
    selected: Tensor,
) -> Tensor:
    rows = []
    for row in range(activation.shape[0]):
        index = selected[row]
        y = torch.nn.functional.linear(
            activation[row, index],
            down.weight[:, index],
            down.bias,
        )
        rows.append(y)
    return torch.stack(rows)


def install_hook(
    mlp: nn.Module,
    *,
    fraction: float,
    block_size: int,
    mode: str,
):
    down = mlp.down_proj

    def hook(
        _module: nn.Module,
        args: tuple[Tensor, ...],
        output: Tensor,
    ) -> Tensor:
        # The post-hook sees the MLP input, so obtain exact SwiGLU activation
        # from the already-resident dense weights. This is an oracle experiment.
        hidden = args[0][:, -1, :]
        gate = torch.nn.functional.linear(
            hidden,
            mlp.gate_proj.weight,
            mlp.gate_proj.bias,
        )
        up = torch.nn.functional.linear(
            hidden,
            mlp.up_proj.weight,
            mlp.up_proj.bias,
        )
        act = getattr(mlp, "act_fn", torch.nn.functional.silu)(gate) * up

        if mode == "greedy":
            selected = residual_greedy_indices(
                act,
                down.weight,
                fraction=fraction,
                block_size=block_size,
            )
        elif mode == "topk":
            selected = norm_topk_indices(
                act,
                down.weight,
                fraction=fraction,
            )
        else:
            raise ValueError(f"unknown mode: {mode}")

        sparse = exact_output_from_selected(act, down, selected)
        replaced = output.clone()
        replaced[:, -1, :] = sparse.to(replaced.dtype)
        return replaced

    return mlp.register_forward_hook(hook)


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
    for fraction in args.fractions:
        for mode in ("topk", "greedy"):
            handle = install_hook(
                mlp,
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
                    "mode": mode,
                    "active_neuron_fraction": fraction,
                    **metrics(baseline, candidate),
                }
            )

    payload = {
        "experiment": "hf_greedy_end_to_end_v0",
        "model": args.model,
        "rows": rows,
        "claim_boundary": (
            "Greedy selection sees exact full SwiGLU activations and exact down "
            "weights, so it is a behavioral oracle. It tests whether a tiny "
            "subset of the original neuron contributions can preserve final "
            "next-token logits; it is not yet a deployable route or byte saving."
        ),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        Path(args.json_out).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
