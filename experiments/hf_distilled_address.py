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
    "Why is top-k activation magnitude not necessarily top-k predictive importance?",
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
    if len(rows) < 8:
        raise ValueError("need at least eight texts for train/test splits")
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


def act_fn(mlp: nn.Module):
    fn = getattr(mlp, "act_fn", None)
    return fn if callable(fn) else torch.nn.functional.silu


@torch.inference_mode()
def capture_teacher(
    model: nn.Module,
    mlp: nn.Module,
    encoded: dict[str, Tensor],
    *,
    max_tokens: int,
) -> tuple[Tensor, Tensor]:
    inputs: list[Tensor] = []
    activations: list[Tensor] = []

    def mlp_pre(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
        x = args[0].detach().reshape(-1, args[0].shape[-1])
        inputs.append(x.cpu())

    def down_pre(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
        u = args[0].detach().reshape(-1, args[0].shape[-1])
        activations.append(u.cpu())

    h1 = mlp.register_forward_pre_hook(mlp_pre)
    h2 = mlp.down_proj.register_forward_pre_hook(down_pre)
    try:
        model(**encoded)
    finally:
        h1.remove()
        h2.remove()

    hidden = torch.cat(inputs, dim=0)
    activation = torch.cat(activations, dim=0)
    mask = encoded.get("attention_mask")
    if mask is not None:
        valid = mask.reshape(-1).to(torch.bool).cpu()
        hidden = hidden[valid]
        activation = activation[valid]
    hidden = hidden[:max_tokens].float()
    activation = activation[:max_tokens].float()
    down_norm = mlp.down_proj.weight.detach().float().cpu().norm(dim=0)
    teacher_score = activation.abs() * down_norm[None, :]
    return hidden, teacher_score


class AddressHead(nn.Module):
    def __init__(self, hidden: int, width: int, rank: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(hidden, rank, bias=False)
        self.out_proj = nn.Linear(rank, width, bias=True)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.out_proj(torch.tanh(self.in_proj(hidden)))


def topk_labels(score: Tensor, fraction: float) -> Tensor:
    width = score.shape[-1]
    k = max(1, min(width, round(width * fraction)))
    indices = torch.topk(score, k=k, dim=-1, sorted=False).indices
    labels = torch.zeros_like(score)
    labels.scatter_(1, indices, 1.0)
    return labels


def train_head(
    hidden: Tensor,
    teacher_score: Tensor,
    *,
    rank: int,
    teacher_fraction: float,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> AddressHead:
    torch.manual_seed(seed)
    head = AddressHead(hidden.shape[-1], teacher_score.shape[-1], rank)
    labels = topk_labels(teacher_score, teacher_fraction)

    positives = labels.sum()
    negatives = labels.numel() - positives
    pos_weight = (negatives / positives.clamp_min(1)).detach()
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)

    generator = torch.Generator().manual_seed(seed + 1)
    for _ in range(steps):
        if hidden.shape[0] <= batch_size:
            index = torch.arange(hidden.shape[0])
        else:
            index = torch.randint(
                hidden.shape[0],
                (batch_size,),
                generator=generator,
            )
        logits = head(hidden[index])
        target = labels[index]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            target,
            pos_weight=pos_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return head.eval()


def sparse_exact_output(
    hidden: Tensor,
    mlp: nn.Module,
    selected: Tensor,
) -> Tensor:
    rows = []
    fn = act_fn(mlp)
    for row in range(hidden.shape[0]):
        idx = selected[row]
        x = hidden[row]
        gate = torch.nn.functional.linear(
            x,
            mlp.gate_proj.weight[idx],
            mlp.gate_proj.bias[idx] if mlp.gate_proj.bias is not None else None,
        )
        up = torch.nn.functional.linear(
            x,
            mlp.up_proj.weight[idx],
            mlp.up_proj.bias[idx] if mlp.up_proj.bias is not None else None,
        )
        activation = fn(gate) * up
        y = torch.nn.functional.linear(
            activation,
            mlp.down_proj.weight[:, idx],
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


def metadata_fraction(
    head: AddressHead,
    mlp: nn.Module,
    *,
    selected_fraction: float,
) -> dict[str, float]:
    metadata_params = sum(parameter.numel() for parameter in head.parameters())
    metadata_bytes = metadata_params * 2
    gate = mlp.gate_proj
    up = mlp.up_proj
    down = mlp.down_proj
    cold_params = gate.weight.numel() + up.weight.numel() + down.weight.numel()
    cold_bytes = cold_params * 2
    selected_payload = cold_bytes * selected_fraction
    return {
        "address_metadata_bytes_fp16": int(metadata_bytes),
        "address_metadata_fraction": metadata_bytes / cold_bytes,
        "selected_payload_fraction": selected_fraction,
        "address_plus_selected_fraction": (
            metadata_bytes + selected_payload
        )
        / cold_bytes,
    }


def evaluate_head(
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

    def hook(
        _module: nn.Module,
        args: tuple[Tensor, ...],
        output: Tensor,
    ) -> Tensor:
        hidden = args[0][:, -1, :].detach().float().cpu()
        score = head(hidden)
        selected = torch.topk(score, k=k, dim=-1, sorted=False).indices
        sparse = sparse_exact_output(
            args[0][:, -1, :],
            mlp,
            selected.to(args[0].device),
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
        **metadata_fraction(head, mlp, selected_fraction=fraction),
        **distribution_metrics(baseline, candidate),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--text-file")
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--max-train-tokens", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
    )
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
    parser.add_argument("--seed", type=int, default=41)
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
    if args.layer < 0 or args.layer >= len(layers):
        raise ValueError("layer index out of range")
    mlp = resolve_mlp(layers[args.layer])

    hidden, teacher = capture_teacher(
        model,
        mlp,
        train_encoded,
        max_tokens=args.max_train_tokens,
    )
    baseline = last_logits(model, test_encoded)

    rows = []
    for rank in args.ranks:
        head = train_head(
            hidden,
            teacher,
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
                    **evaluate_head(
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
        "experiment": "hf_distilled_address_v0",
        "model": args.model,
        "layer": args.layer,
        "train_tokens": int(hidden.shape[0]),
        "test_prompts": len(test_texts),
        "teacher_fraction": args.teacher_fraction,
        "rows": rows,
        "claim_boundary": (
            "The address head is trained only from calibration prompts to predict "
            "teacher top-neuron demand. Runtime selection uses only the incoming "
            "hidden state plus the tiny address head; selected neurons then use "
            "exact cold weights. PyTorch still computes the dense MLP before the "
            "replacement hook, so byte savings remain analytic until a fused "
            "selective kernel is profiled."
        ),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        Path(args.json_out).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
