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
    "Describe predictive coding as a baseline plus sparse residual corrections.",
    "Why might the residual after dominant neurons be easier to predict?",
    "How can a small hot operator and cold exact pages cooperate?",
    "What is the difference between approximating a full nonlinear map and its residual?",
    "Explain why exact sparse corrections can protect a cheap surrogate from worst cases.",
    "How would you account for resident metadata in a bandwidth budget?",
    "Why can a low-rank residual predictor be useful even when a low-rank full predictor fails?",
    "What would make a sparse correction architecture hardware efficient?",
    "Explain selective prediction and abstention.",
    "Why should an uncertain sparse router fall back to dense execution?",
    "How can two cheap predictors provide a disagreement signal?",
    "What is the difference between average byte savings and worst-case byte savings?",
    "How can a calibrated fallback preserve quality while reducing average traffic?",
    "Why is a conditional page fault safer than unconditional sparsification?",
    "Describe a confidence threshold calibrated on held-out data.",
    "What would make an uncertainty-triggered memory fault useful?",
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


def encode_batch(
    tokenizer,
    rows: list[str],
    device: torch.device,
    max_length: int,
) -> dict[str, Tensor]:
    batch = tokenizer(
        rows,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return {
        key: value.to(device)
        for key, value in batch.items()
        if isinstance(value, Tensor)
    }


@torch.inference_mode()
def capture_training_trace(
    model: nn.Module,
    mlp: nn.Module,
    encoded: dict[str, Tensor],
    *,
    max_tokens: int,
) -> tuple[Tensor, Tensor, Tensor]:
    inputs: list[Tensor] = []
    activations: list[Tensor] = []
    outputs: list[Tensor] = []

    def mlp_pre(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
        inputs.append(args[0].detach())

    def down_pre(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
        activations.append(args[0].detach())

    def mlp_post(
        _module: nn.Module,
        _args: tuple[Tensor, ...],
        output: Tensor,
    ) -> None:
        outputs.append(output.detach())

    handles = [
        mlp.register_forward_pre_hook(mlp_pre),
        mlp.down_proj.register_forward_pre_hook(down_pre),
        mlp.register_forward_hook(mlp_post),
    ]
    try:
        model(**encoded)
    finally:
        for handle in handles:
            handle.remove()

    hidden = inputs[0].reshape(-1, inputs[0].shape[-1]).cpu()
    activation = activations[0].reshape(-1, activations[0].shape[-1]).cpu()
    dense_output = outputs[0].reshape(-1, outputs[0].shape[-1]).cpu()
    mask = encoded.get("attention_mask")
    if mask is not None:
        valid = mask.reshape(-1).to(torch.bool).cpu()
        hidden = hidden[valid]
        activation = activation[valid]
        dense_output = dense_output[valid]

    return (
        hidden[:max_tokens].float().clone(),
        activation[:max_tokens].float().clone(),
        dense_output[:max_tokens].float().clone(),
    )


def teacher_score(activation: Tensor, mlp: nn.Module) -> Tensor:
    norm = mlp.down_proj.weight.detach().float().cpu().norm(dim=0)
    return activation.abs() * norm[None, :]


def topk_indices(score: Tensor, fraction: float) -> Tensor:
    width = score.shape[-1]
    k = max(1, min(width, round(width * fraction)))
    return torch.topk(score, k=k, dim=-1, sorted=False).indices


def selected_contribution_from_activation(
    activation: Tensor,
    mlp: nn.Module,
    selected: Tensor,
) -> Tensor:
    weight = mlp.down_proj.weight.detach().float().cpu()
    rows = []
    for row in range(activation.shape[0]):
        index = selected[row]
        rows.append(
            torch.nn.functional.linear(
                activation[row, index],
                weight[:, index],
                bias=None,
            )
        )
    return torch.stack(rows)


class AddressHead(nn.Module):
    def __init__(self, hidden: int, width: int, rank: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(hidden, rank, bias=False)
        self.out_proj = nn.Linear(rank, width, bias=True)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.out_proj(torch.tanh(self.in_proj(hidden)))


class ResidualHead(nn.Module):
    def __init__(self, hidden: int, rank: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(hidden, rank, bias=False)
        self.out_proj = nn.Linear(rank, hidden, bias=True)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.out_proj(torch.tanh(self.in_proj(hidden)))


def train_address(
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
    labels = torch.zeros_like(score)
    labels.scatter_(1, topk_indices(score, teacher_fraction), 1.0)
    positives = labels.sum()
    negatives = labels.numel() - positives
    pos_weight = (negatives / positives.clamp_min(1)).detach()

    torch.manual_seed(seed)
    head = AddressHead(hidden.shape[-1], score.shape[-1], rank)
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


def train_residual(
    hidden: Tensor,
    residual: Tensor,
    *,
    rank: int,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> ResidualHead:
    torch.manual_seed(seed)
    head = ResidualHead(hidden.shape[-1], rank)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed + 1)

    scale = residual.square().mean().sqrt().clamp_min(1e-8)
    target = residual / scale
    for _ in range(steps):
        index = torch.randint(
            hidden.shape[0],
            (min(batch_size, hidden.shape[0]),),
            generator=generator,
        )
        prediction = head(hidden[index]) / scale
        loss = torch.nn.functional.mse_loss(prediction, target[index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return head.eval()


def act_fn(mlp: nn.Module):
    fn = getattr(mlp, "act_fn", None)
    return fn if callable(fn) else torch.nn.functional.silu


def sparse_contribution(
    hidden: Tensor,
    mlp: nn.Module,
    selected: Tensor,
) -> Tensor:
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
        activation = fn(gate) * up
        rows.append(
            torch.nn.functional.linear(
                activation,
                mlp.down_proj.weight[:, index],
                bias=None,
            )
        )
    return torch.stack(rows)


def install_sparse_hook(
    mlp: nn.Module,
    address: AddressHead,
    residual: ResidualHead,
    *,
    fraction: float,
):
    width = mlp.down_proj.in_features
    k = max(1, min(width, round(width * fraction)))

    def hook(
        _module: nn.Module,
        args: tuple[Tensor, ...],
        output: Tensor,
    ) -> Tensor:
        hidden_device = args[0][:, -1, :]
        hidden = hidden_device.detach().float().cpu()
        selected = torch.topk(
            address(hidden),
            k=k,
            dim=-1,
            sorted=False,
        ).indices
        exact = sparse_contribution(
            hidden_device,
            mlp,
            selected.to(hidden_device.device),
        )
        hot = residual(hidden).to(hidden_device.device)
        replaced = output.clone()
        replaced[:, -1, :] = (exact + hot).to(replaced.dtype)
        return replaced

    return mlp.register_forward_hook(hook)


@torch.inference_mode()
def last_logits(model: nn.Module, input_ids: Tensor) -> Tensor:
    return model(input_ids=input_ids).logits[:, -1, :].detach().float().cpu()


def kl_and_agreement(
    reference: Tensor,
    candidate: Tensor,
) -> tuple[Tensor, Tensor]:
    ref_logp = torch.log_softmax(reference, dim=-1)
    cand_logp = torch.log_softmax(candidate, dim=-1)
    probability = ref_logp.exp()
    kl = (probability * (ref_logp - cand_logp)).sum(dim=-1)
    agreement = reference.argmax(dim=-1) == candidate.argmax(dim=-1)
    return kl, agreement


def metadata_fraction(
    address: AddressHead,
    residual: ResidualHead,
    mlp: nn.Module,
) -> float:
    metadata_parameters = sum(p.numel() for p in address.parameters())
    metadata_parameters += sum(p.numel() for p in residual.parameters())
    cold_parameters = (
        mlp.gate_proj.weight.numel()
        + mlp.up_proj.weight.numel()
        + mlp.down_proj.weight.numel()
    )
    return metadata_parameters / cold_parameters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--text-file")
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--max-train-tokens", type=int, default=512)
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--teacher-fraction", type=float, default=0.15)
    parser.add_argument("--address-rank", type=int, default=8)
    parser.add_argument("--residual-rank", type=int, default=32)
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=[0.10, 0.125, 0.15],
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=211)
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

    train_encoded = encode_batch(
        tokenizer,
        train_texts,
        device,
        args.max_length,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    layers = resolve_layers(model)
    if args.layer < 0 or args.layer >= len(layers):
        raise ValueError("layer index out of range")
    mlp = resolve_mlp(layers[args.layer])

    hidden, activation, dense_output = capture_training_trace(
        model,
        mlp,
        train_encoded,
        max_tokens=args.max_train_tokens,
    )
    score = teacher_score(activation, mlp)
    address = train_address(
        hidden,
        score,
        rank=args.address_rank,
        teacher_fraction=args.teacher_fraction,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )
    with torch.no_grad():
        teacher_selected = torch.topk(
            address(hidden),
            k=max(
                1,
                round(score.shape[-1] * args.teacher_fraction),
            ),
            dim=-1,
            sorted=False,
        ).indices
    exact = selected_contribution_from_activation(
        activation,
        mlp,
        teacher_selected,
    )
    residual = train_residual(
        hidden,
        dense_output - exact,
        rank=args.residual_rank,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed + 1,
    )

    rows = []
    for fraction in args.fractions:
        all_kl: list[Tensor] = []
        all_agreement: list[Tensor] = []
        per_prompt_exact: list[bool] = []
        first_mismatch_steps: list[int] = []

        for text in test_texts:
            prompt = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
            )["input_ids"].to(device)

            prefix = prompt
            prompt_exact = True
            first_mismatch = args.decode_steps
            for step in range(args.decode_steps):
                dense = last_logits(model, prefix)

                hook = install_sparse_hook(
                    mlp,
                    address,
                    residual,
                    fraction=fraction,
                )
                try:
                    sparse = last_logits(model, prefix)
                finally:
                    hook.remove()

                kl, agreement = kl_and_agreement(dense, sparse)
                all_kl.append(kl)
                all_agreement.append(agreement)

                if not bool(agreement.item()) and prompt_exact:
                    prompt_exact = False
                    first_mismatch = step

                next_token = dense.argmax(dim=-1).to(device)
                prefix = torch.cat(
                    [prefix, next_token[:, None]],
                    dim=1,
                )

            per_prompt_exact.append(prompt_exact)
            first_mismatch_steps.append(first_mismatch)

        kl = torch.cat(all_kl)
        agreement = torch.cat(all_agreement).float()
        meta = metadata_fraction(address, residual, mlp)
        rows.append(
            {
                "fraction": fraction,
                "states_evaluated": int(kl.numel()),
                "mean_kl_nats": float(kl.mean().item()),
                "p95_kl_nats": float(torch.quantile(kl, 0.95).item()),
                "max_kl_nats": float(kl.max().item()),
                "top1_agreement": float(agreement.mean().item()),
                "exact_prompt_rollout_fraction": (
                    sum(per_prompt_exact) / len(per_prompt_exact)
                ),
                "mean_first_mismatch_step": (
                    sum(first_mismatch_steps) / len(first_mismatch_steps)
                ),
                "resident_metadata_fraction": meta,
                "selected_payload_fraction": fraction,
                "total_metadata_plus_payload_fraction": meta + fraction,
            }
        )

    payload = {
        "experiment": "hf_decode_rollout_hot_residual_v0",
        "model": args.model,
        "layer": args.layer,
        "train_tokens": int(hidden.shape[0]),
        "test_prompts": len(test_texts),
        "decode_steps": args.decode_steps,
        "teacher_fraction": args.teacher_fraction,
        "address_rank": args.address_rank,
        "residual_rank": args.residual_rank,
        "rows": rows,
        "claim_boundary": (
            "Every evaluation state is a held-out autoregressive decode state on "
            "the dense model's greedy trajectory. The sparse/hot-residual MLP sees "
            "the same prefix, so KL and top-1 agreement are directly comparable. "
            "Only one MLP layer is modified and byte savings are analytic until a "
            "page-selective kernel is profiled."
        ),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        Path(args.json_out).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
