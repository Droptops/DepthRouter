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
    "What does a local Jacobian tell us about predictive sensitivity?",
    "Why might the important computation be sparse in logit tangent space?",
    "Explain why a prediction-aware router could outperform an activation router.",
    "How can a tiny address head be trained by a larger dense teacher?",
    "Describe a two-stage system with cheap addressing and exact sparse payload execution.",
    "Why is the addressability gap a distinct problem from sparsity itself?",
    "What is a sufficient statistic for choosing the next computation?",
    "How could expected information gain per byte guide inference?",
    "Explain why repeated semantic states might share future computation.",
    "What makes a cache policy falsifiable on hardware?",
    "Why does physical locality matter even when FLOP count is unchanged?",
    "How would you distinguish model compression from runtime working-set compression?",
    "What is the difference between a sparse model and sparse execution of a dense model?",
    "Why can a router's own memory footprint erase the savings it predicts?",
    "How could a compiler co-locate neurons with correlated predictive demand?",
    "Explain why a final-layer experiment is a useful first boundary test.",
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
    if len(rows) < 8:
        raise ValueError("need at least eight texts")
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


def act_fn(mlp: nn.Module):
    fn = getattr(mlp, "act_fn", None)
    return fn if callable(fn) else torch.nn.functional.silu


@torch.inference_mode()
def capture_trace(
    model: nn.Module,
    layer: nn.Module,
    encoded: dict[str, Tensor],
    *,
    max_tokens: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Capture valid-token MLP inputs, activations, final hidden, and logits."""

    mlp = layer.mlp
    mlp_inputs: list[Tensor] = []
    down_inputs: list[Tensor] = []
    layer_outputs: list[Tensor] = []

    def mlp_pre(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
        mlp_inputs.append(args[0].detach())

    def down_pre(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
        down_inputs.append(args[0].detach())

    def layer_post(_module: nn.Module, _args: tuple[Tensor, ...], output) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        layer_outputs.append(hidden.detach())

    handles = [
        mlp.register_forward_pre_hook(mlp_pre),
        mlp.down_proj.register_forward_pre_hook(down_pre),
        layer.register_forward_hook(layer_post),
    ]
    try:
        logits = model(**encoded).logits.detach()
    finally:
        for handle in handles:
            handle.remove()

    if len(mlp_inputs) != 1 or len(down_inputs) != 1 or len(layer_outputs) != 1:
        raise RuntimeError("expected one final-layer invocation")

    mask = encoded.get("attention_mask")
    if mask is None:
        valid = torch.ones(
            mlp_inputs[0].shape[:2],
            dtype=torch.bool,
            device=mlp_inputs[0].device,
        )
    else:
        valid = mask.to(torch.bool)

    hidden_in = mlp_inputs[0][valid][:max_tokens].float().cpu()
    activation = down_inputs[0][valid][:max_tokens].float().cpu()
    dense_hidden = layer_outputs[0][valid][:max_tokens].float().cpu()
    token_logits = logits[valid][:max_tokens].float().cpu()
    return hidden_in, activation, dense_hidden, token_logits


def rmsnorm_logit_jacobian(
    dense_hidden: Tensor,
    top_ids: Tensor,
    norm: nn.Module,
    lm_head: nn.Linear,
) -> Tensor:
    h = dense_hidden.float()
    weight = norm.weight.detach().float().cpu()
    head = lm_head.weight.detach().float().cpu()
    selected_head = head[top_ids]
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


def tangent_gain_scores(
    activation: Tensor,
    down_weight: Tensor,
    jacobian: Tensor,
) -> Tensor:
    """One-step predictive reconstruction gain for every MLP neuron."""

    basis = torch.einsum(
        "nd,bmd->bnm",
        down_weight.T.float().cpu(),
        jacobian.float(),
    )
    effects = activation.float()[:, :, None] * basis
    target = effects.sum(dim=1)
    dot = torch.einsum("bm,bnm->bn", target, effects)
    norm_sq = effects.square().sum(dim=-1)
    gain = 2.0 * dot - norm_sq
    # Ranking only; negative gains should not be rewarded.
    return gain.clamp_min(0)


class AddressHead(nn.Module):
    def __init__(self, hidden: int, width: int, rank: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(hidden, rank, bias=False)
        self.out_proj = nn.Linear(rank, width, bias=True)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.out_proj(torch.tanh(self.in_proj(hidden)))


def topk_labels(score: Tensor, fraction: float) -> Tensor:
    k = max(1, min(score.shape[-1], round(score.shape[-1] * fraction)))
    index = torch.topk(score, k=k, dim=-1, sorted=False).indices
    labels = torch.zeros_like(score)
    labels.scatter_(1, index, 1.0)
    return labels


def train_head(
    hidden: Tensor,
    score: Tensor,
    *,
    rank: int,
    teacher_fraction: float,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> AddressHead:
    torch.manual_seed(seed)
    labels = topk_labels(score, teacher_fraction)
    head = AddressHead(hidden.shape[-1], score.shape[-1], rank)
    positives = labels.sum()
    negatives = labels.numel() - positives
    pos_weight = (negatives / positives.clamp_min(1)).detach()
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed + 1)

    for _ in range(steps):
        index = torch.randint(
            hidden.shape[0],
            (min(batch_size, hidden.shape[0]),),
            generator=generator,
        )
        logits = head(hidden[index])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            labels[index],
            pos_weight=pos_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return head.eval()


def sparse_exact_output(hidden: Tensor, mlp: nn.Module, selected: Tensor) -> Tensor:
    rows = []
    fn = act_fn(mlp)
    for row in range(hidden.shape[0]):
        index = selected[row]
        x = hidden[row]
        gate = torch.nn.functional.linear(
            x,
            mlp.gate_proj.weight[index],
            mlp.gate_proj.bias[index] if mlp.gate_proj.bias is not None else None,
        )
        up = torch.nn.functional.linear(
            x,
            mlp.up_proj.weight[index],
            mlp.up_proj.bias[index] if mlp.up_proj.bias is not None else None,
        )
        u = fn(gate) * up
        y = torch.nn.functional.linear(
            u,
            mlp.down_proj.weight[:, index],
            mlp.down_proj.bias,
        )
        rows.append(y)
    return torch.stack(rows)


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


def metadata_stats(
    head: AddressHead,
    mlp: nn.Module,
    fraction: float,
) -> dict[str, float]:
    metadata_params = sum(p.numel() for p in head.parameters())
    metadata_bytes = metadata_params * 2
    cold_params = (
        mlp.gate_proj.weight.numel()
        + mlp.up_proj.weight.numel()
        + mlp.down_proj.weight.numel()
    )
    cold_bytes = cold_params * 2
    return {
        "address_metadata_bytes_fp16": int(metadata_bytes),
        "address_metadata_fraction": metadata_bytes / cold_bytes,
        "selected_payload_fraction": fraction,
        "address_plus_selected_fraction": metadata_bytes / cold_bytes + fraction,
    }


def evaluate_runtime_head(
    model: nn.Module,
    mlp: nn.Module,
    head: AddressHead,
    encoded: dict[str, Tensor],
    baseline: Tensor,
    *,
    fraction: float,
) -> dict[str, float]:
    width = mlp.down_proj.in_features
    k = max(1, min(width, round(width * fraction)))

    def hook(_module: nn.Module, args: tuple[Tensor, ...], output: Tensor) -> Tensor:
        hidden = args[0][:, -1, :]
        address = head(hidden.detach().float().cpu())
        selected = torch.topk(address, k=k, dim=-1, sorted=False).indices
        sparse = sparse_exact_output(
            hidden,
            mlp,
            selected.to(hidden.device),
        )
        replaced = output.clone()
        replaced[:, -1, :] = sparse.to(replaced.dtype)
        return replaced

    handle = mlp.register_forward_hook(hook)
    try:
        candidate = last_logits(model, encoded)
    finally:
        handle.remove()

    return {
        **metadata_stats(head, mlp, fraction),
        **distribution_metrics(baseline, candidate),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--text-file")
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--max-train-tokens", type=int, default=256)
    parser.add_argument("--top-logits", type=int, default=32)
    parser.add_argument("--teacher-fraction", type=float, default=0.10)
    parser.add_argument("--ranks", nargs="+", type=int, default=[8, 16, 32, 64])
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=[0.05, 0.10, 0.15, 0.20],
    )
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=73)
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

    texts = load_texts(args.text_file)
    split = len(texts) // 2
    train_texts = texts[:split]
    test_texts = texts[split:]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def encode(rows: list[str]) -> dict[str, Tensor]:
        batch = tokenizer(
            rows,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        )
        return {
            key: value.to(device)
            for key, value in batch.items()
            if isinstance(value, Tensor)
        }

    train_encoded = encode(train_texts)
    test_encoded = encode(test_texts)

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

    hidden, activation, dense_hidden, token_logits = capture_trace(
        model,
        final_layer,
        train_encoded,
        max_tokens=args.max_train_tokens,
    )
    top_ids = torch.topk(
        token_logits,
        k=min(args.top_logits, token_logits.shape[-1]),
        dim=-1,
    ).indices
    jacobian = rmsnorm_logit_jacobian(
        dense_hidden,
        top_ids,
        norm,
        lm_head,
    )
    teacher_score = tangent_gain_scores(
        activation,
        mlp.down_proj.weight,
        jacobian,
    )
    baseline = last_logits(model, test_encoded)

    rows = []
    for rank in args.ranks:
        head = train_head(
            hidden,
            teacher_score,
            rank=rank,
            teacher_fraction=args.teacher_fraction,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed + rank,
        )
        for fraction in args.fractions:
            rows.append(
                {
                    "rank": rank,
                    "fraction": fraction,
                    **evaluate_runtime_head(
                        model,
                        mlp,
                        head,
                        test_encoded,
                        baseline,
                        fraction=fraction,
                    ),
                }
            )

    payload = {
        "experiment": "hf_tangent_distilled_address_v0",
        "model": args.model,
        "layer": len(layers) - 1,
        "train_tokens": int(hidden.shape[0]),
        "test_prompts": len(test_texts),
        "top_logit_tangent_dim": args.top_logits,
        "teacher_fraction": args.teacher_fraction,
        "rows": rows,
        "claim_boundary": (
            "The teacher uses dense calibration traces plus the final-logit local "
            "Jacobian. The runtime address head sees only the incoming hidden state; "
            "selected neurons use exact cold weights. Byte savings remain analytic "
            "until page-selective execution is fused and profiled."
        ),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        Path(args.json_out).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
