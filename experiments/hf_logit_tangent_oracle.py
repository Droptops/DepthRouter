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
def capture_final_layer(
    model: nn.Module,
    layer: nn.Module,
    encoded: dict[str, Tensor],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    mlp = layer.mlp
    down_inputs: list[Tensor] = []
    mlp_outputs: list[Tensor] = []
    layer_outputs: list[Tensor] = []

    def down_pre(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
        down_inputs.append(args[0][:, -1, :].detach())

    def mlp_post(
        _module: nn.Module,
        _args: tuple[Tensor, ...],
        output: Tensor,
    ) -> None:
        mlp_outputs.append(output[:, -1, :].detach())

    def layer_post(
        _module: nn.Module,
        _args: tuple[Tensor, ...],
        output,
    ) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        layer_outputs.append(hidden[:, -1, :].detach())

    handles = [
        mlp.down_proj.register_forward_pre_hook(down_pre),
        mlp.register_forward_hook(mlp_post),
        layer.register_forward_hook(layer_post),
    ]
    try:
        logits = model(**encoded).logits[:, -1, :].detach()
    finally:
        for handle in handles:
            handle.remove()

    if not (
        len(down_inputs) == len(mlp_outputs) == len(layer_outputs) == 1
    ):
        raise RuntimeError("expected one invocation of the final decoder layer")

    dense_hidden = layer_outputs[0]
    dense_mlp = mlp_outputs[0]
    base_hidden = dense_hidden - dense_mlp
    return (
        logits.float(),
        base_hidden.float(),
        dense_hidden.float(),
        down_inputs[0].float(),
    )


def rmsnorm_logit_jacobian(
    dense_hidden: Tensor,
    top_ids: Tensor,
    norm: nn.Module,
    lm_head: nn.Linear,
) -> Tensor:
    """Jacobian of selected final logits with respect to pre-norm hidden state."""

    h = dense_hidden.float()
    weight = norm.weight.detach().float().to(h.device)
    head = lm_head.weight.detach().float().to(h.device)
    selected_head = head[top_ids]  # [B, M, D]
    a = selected_head * weight[None, None, :]

    eps = float(
        getattr(
            norm,
            "variance_epsilon",
            getattr(norm, "eps", 1e-6),
        )
    )
    d = h.shape[-1]
    radius = torch.sqrt(h.square().mean(dim=-1) + eps)
    dot = torch.einsum("bmd,bd->bm", a, h)

    first = a / radius[:, None, None]
    second = (
        dot[:, :, None]
        * h[:, None, :]
        / (float(d) * radius[:, None, None].pow(3))
    )
    return first - second


def projected_neuron_effects(
    activation: Tensor,
    down_weight: Tensor,
    jacobian: Tensor,
) -> Tensor:
    """First-order selected-logit effect of every original MLP neuron."""

    basis = torch.einsum(
        "nd,bmd->bnm",
        down_weight.T.float(),
        jacobian.float(),
    )
    return activation.float()[:, :, None] * basis


def greedy_effect_indices(
    effects: Tensor,
    *,
    fraction: float,
    block_size: int,
) -> Tensor:
    """Greedily reconstruct the dense predictive tangent with neuron effects."""

    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if block_size < 1:
        raise ValueError("block_size must be >= 1")

    target = effects.sum(dim=1)
    residual = target.clone()
    batch, width, _ = effects.shape
    k = max(1, min(width, round(width * fraction)))
    selected = torch.zeros(batch, width, dtype=torch.bool, device=effects.device)
    chunks: list[Tensor] = []

    chosen = 0
    effect_norm_sq = effects.square().sum(dim=-1)
    while chosen < k:
        take = min(block_size, k - chosen)
        dot = torch.einsum("bm,bnm->bn", residual, effects)
        gain = 2.0 * dot - effect_norm_sq
        gain = gain.masked_fill(selected, float("-inf"))
        index = torch.topk(gain, k=take, dim=-1, sorted=False).indices
        selected.scatter_(1, index, True)
        chunks.append(index)

        picked = torch.gather(
            effects,
            1,
            index[:, :, None].expand(-1, -1, effects.shape[-1]),
        )
        residual = residual - picked.sum(dim=1)
        chosen += take

    return torch.cat(chunks, dim=1)


def sparse_hidden(
    base_hidden: Tensor,
    activation: Tensor,
    down_weight: Tensor,
    selected: Tensor,
) -> Tensor:
    rows = []
    wt = down_weight.T.float()
    for row in range(activation.shape[0]):
        index = selected[row]
        contribution = (
            activation[row, index, None] * wt[index]
        ).sum(dim=0)
        rows.append(base_hidden[row] + contribution)
    return torch.stack(rows)


def exact_logits_from_hidden(
    hidden: Tensor,
    norm: nn.Module,
    lm_head: nn.Linear,
) -> Tensor:
    normalized = norm(hidden.to(norm.weight.dtype))
    return lm_head(normalized).float()


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--text-file")
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument(
        "--top-logits",
        nargs="+",
        type=int,
        default=[8, 32, 128],
    )
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=[0.005, 0.01, 0.02, 0.05, 0.10],
    )
    parser.add_argument("--block-size", type=int, default=16)
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
    final_layer = layers[-1]
    mlp = final_layer.mlp
    norm = model.model.norm
    lm_head = model.lm_head

    reference_logits, base_hidden, dense_hidden, activation = capture_final_layer(
        model,
        final_layer,
        encoded,
    )
    reconstructed = base_hidden + torch.nn.functional.linear(
        activation,
        mlp.down_proj.weight,
        mlp.down_proj.bias,
    )
    reconstruction_error = float(
        (reconstructed - dense_hidden).abs().max().item()
    )
    direct_logits = exact_logits_from_hidden(dense_hidden, norm, lm_head)
    direct_logit_error = float(
        (direct_logits - reference_logits).abs().max().item()
    )

    rows = []
    for top_m in args.top_logits:
        top_ids = torch.topk(
            reference_logits,
            k=min(top_m, reference_logits.shape[-1]),
            dim=-1,
        ).indices
        jacobian = rmsnorm_logit_jacobian(
            dense_hidden,
            top_ids,
            norm,
            lm_head,
        )
        effects = projected_neuron_effects(
            activation,
            mlp.down_proj.weight,
            jacobian,
        )

        for fraction in args.fractions:
            selected = greedy_effect_indices(
                effects,
                fraction=fraction,
                block_size=args.block_size,
            )
            candidate_hidden = sparse_hidden(
                base_hidden,
                activation,
                mlp.down_proj.weight,
                selected,
            )
            candidate_logits = exact_logits_from_hidden(
                candidate_hidden,
                norm,
                lm_head,
            )
            rows.append(
                {
                    "top_logit_tangent_dim": top_m,
                    "active_neuron_fraction": fraction,
                    "selected_neurons": int(selected.shape[1]),
                    **distribution_metrics(
                        reference_logits,
                        candidate_logits,
                    ),
                }
            )

    payload = {
        "experiment": "hf_logit_tangent_oracle_v0",
        "model": args.model,
        "layer": len(layers) - 1,
        "dense_hidden_reconstruction_max_abs_error": reconstruction_error,
        "direct_logit_reconstruction_max_abs_error": direct_logit_error,
        "rows": rows,
        "claim_boundary": (
            "Selection is an oracle based on the local Jacobian of the dense "
            "next-token logits at the final hidden state. It tests whether "
            "predictive behavior is sparse in probability/logit tangent space, "
            "not whether that subset can yet be predicted cheaply or executed "
            "with lower measured HBM traffic."
        ),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        Path(args.json_out).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
