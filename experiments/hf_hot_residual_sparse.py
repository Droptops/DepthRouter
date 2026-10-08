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
        inputs.append(args[0].detach().reshape(-1, args[0].shape[-1]).cpu())

    def down_pre(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
        activations.append(args[0].detach().reshape(-1, args[0].shape[-1]).cpu())

    def mlp_post(_module: nn.Module, _args: tuple[Tensor, ...], output: Tensor) -> None:
        outputs.append(output.detach().reshape(-1, output.shape[-1]).cpu())

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

    hidden = torch.cat(inputs, dim=0)
    activation = torch.cat(activations, dim=0)
    dense_output = torch.cat(outputs, dim=0)
    mask = encoded.get("attention_mask")
    if mask is not None:
        valid = mask.reshape(-1).to(torch.bool).cpu()
        hidden = hidden[valid]
        activation = activation[valid]
        dense_output = dense_output[valid]
    hidden = hidden[:max_tokens].float()
    activation = activation[:max_tokens].float()
    dense_output = dense_output[:max_tokens].float()
    return hidden, activation, dense_output


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
        idx = selected[row]
        rows.append(
            torch.nn.functional.linear(
                activation[row, idx],
                weight[:, idx],
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
    width = score.shape[-1]
    selected = topk_indices(score, teacher_fraction)
    labels = torch.zeros_like(score)
    labels.scatter_(1, selected, 1.0)

    positives = labels.sum()
    negatives = labels.numel() - positives
    pos_weight = (negatives / positives.clamp_min(1)).detach()

    torch.manual_seed(seed)
    head = AddressHead(hidden.shape[-1], width, rank)
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
        rows.append(
            torch.nn.functional.linear(
                activation,
                mlp.down_proj.weight[:, idx],
                bias=None,
            )
        )
    return torch.stack(rows)


@torch.inference_mode()
def reference_logits(
    model: nn.Module,
    encoded: dict[str, Tensor],
    *,
    all_tokens: bool,
) -> Tensor:
    logits = model(**encoded).logits.detach().float().cpu()
    if not all_tokens:
        return logits[:, -1, :]

    flat = logits.reshape(-1, logits.shape[-1])
    mask = encoded.get("attention_mask")
    if mask is None:
        return flat
    valid = mask.detach().cpu().reshape(-1).to(torch.bool)
    return flat[valid]


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


def resident_fraction(
    address: AddressHead,
    residual: ResidualHead,
    mlp: nn.Module,
    *,
    selected_fraction: float,
) -> dict[str, float]:
    metadata_params = sum(p.numel() for p in address.parameters())
    metadata_params += sum(p.numel() for p in residual.parameters())
    metadata_bytes = metadata_params * 2

    cold_params = (
        mlp.gate_proj.weight.numel()
        + mlp.up_proj.weight.numel()
        + mlp.down_proj.weight.numel()
    )
    cold_bytes = cold_params * 2
    metadata_fraction = metadata_bytes / cold_bytes
    return {
        "resident_metadata_bytes_fp16": int(metadata_bytes),
        "resident_metadata_fraction": metadata_fraction,
        "selected_payload_fraction": selected_fraction,
        "total_address_plus_payload_fraction": metadata_fraction + selected_fraction,
    }


def evaluate(
    model: nn.Module,
    mlp: nn.Module,
    address: AddressHead,
    residual: ResidualHead,
    encoded: dict[str, Tensor],
    baseline: Tensor,
    *,
    fraction: float,
    all_tokens: bool,
) -> dict[str, float]:
    width = mlp.down_proj.in_features
    k = max(1, min(width, round(width * fraction)))

    def hook(_module: nn.Module, args: tuple[Tensor, ...], output: Tensor) -> Tensor:
        if all_tokens:
            shape = args[0].shape
            hidden_device = args[0].reshape(-1, shape[-1])
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
            replacement = (exact + hot).reshape(
                *shape[:-1],
                output.shape[-1],
            )
            return replacement.to(output.dtype)

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
        replacement = exact + hot
        replaced = output.clone()
        replaced[:, -1, :] = replacement.to(replaced.dtype)
        return replaced

    handle = mlp.register_forward_hook(hook)
    try:
        candidate = reference_logits(
            model,
            encoded,
            all_tokens=all_tokens,
        )
    finally:
        handle.remove()

    return {
        **resident_fraction(
            address,
            residual,
            mlp,
            selected_fraction=fraction,
        ),
        **distribution_metrics(baseline, candidate),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--text-file")
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--max-train-tokens", type=int, default=512)
    parser.add_argument("--teacher-fraction", type=float, default=0.15)
    parser.add_argument("--address-rank", type=int, default=8)
    parser.add_argument("--residual-ranks", nargs="+", type=int, default=[4, 8, 16, 32])
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=[0.10, 0.15, 0.20],
    )
    parser.add_argument("--steps", type=int, default=220)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument("--all-tokens", action="store_true")
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
    mlp = resolve_mlp(layers[args.layer])

    hidden, activation, dense_output = capture_training_trace(
        model,
        mlp,
        train_encoded,
        max_tokens=args.max_train_tokens,
    )
    hidden = hidden.detach().clone()
    activation = activation.detach().clone()
    dense_output = dense_output.detach().clone()

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

    # Train the hot residual against the *actual address head's* selected set,
    # not the teacher's ideal set. This makes the residual predictor absorb
    # systematic routing mistakes instead of assuming they disappear.
    with torch.no_grad():
        predicted_selected = torch.topk(
            address(hidden),
            k=max(
                1,
                min(
                    score.shape[-1],
                    round(score.shape[-1] * args.teacher_fraction),
                ),
            ),
            dim=-1,
            sorted=False,
        ).indices
    exact_predicted = selected_contribution_from_activation(
        activation,
        mlp,
        predicted_selected,
    )
    residual_target = dense_output - exact_predicted
    baseline = reference_logits(
        model,
        test_encoded,
        all_tokens=args.all_tokens,
    )

    rows = []
    for residual_rank in args.residual_ranks:
        residual = train_residual(
            hidden,
            residual_target,
            rank=residual_rank,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed + residual_rank,
        )
        for fraction in args.fractions:
            rows.append(
                {
                    "address_rank": args.address_rank,
                    "residual_rank": residual_rank,
                    "teacher_fraction": args.teacher_fraction,
                    "fraction": fraction,
                    **evaluate(
                        model,
                        mlp,
                        address,
                        residual,
                        test_encoded,
                        baseline,
                        fraction=fraction,
                        all_tokens=args.all_tokens,
                    ),
                }
            )

    payload = {
        "experiment": "hf_hot_residual_sparse_v0",
        "model": args.model,
        "layer": args.layer,
        "train_tokens": int(hidden.shape[0]),
        "test_prompts": len(test_texts),
        "all_tokens": args.all_tokens,
        "rows": rows,
        "claim_boundary": (
            "A tiny resident address head selects exact cold neurons while a tiny "
            "resident low-rank head predicts the omitted residual MLP output. Both "
            "heads are trained only on calibration prompts. PyTorch still executes "
            "the dense MLP before the diagnostic replacement hook, so byte savings "
            "remain analytic until a selective kernel is profiled."
        ),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        Path(args.json_out).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
