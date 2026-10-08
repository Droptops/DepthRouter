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
    "Explain why a final-layer correction can optimize the prediction rather than local MLP reconstruction.",
    "How can a normalized hidden state act as a sufficient target for a linear language-model head?",
    "Why might late-layer predictive residuals be much easier than reproducing the full MLP output?",
    "Describe a model that keeps a cheap hot predictor and faults exact cold neurons only for residual detail.",
    "What is the difference between reconstructing computation and preserving the answer distribution?",
    "Why should a compression target be measured after the final normalization?",
    "How could a tiny resident correction head exploit repeated probability trajectories?",
    "What would make a late-layer neural cache practically valuable?",
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
    if len(rows) < 12:
        raise ValueError("need at least twelve texts")
    return rows


def resolve_layers(model: nn.Module) -> list[nn.Module]:
    candidates: list[Any] = [
        getattr(getattr(model, "model", None), "layers", None),
        getattr(model, "layers", None),
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


def resolve_final_norm(model: nn.Module) -> nn.Module:
    inner = getattr(model, "model", None)
    norm = getattr(inner, "norm", None) if inner is not None else None
    if not isinstance(norm, nn.Module):
        raise TypeError("could not locate final model norm")
    return norm


def resolve_lm_head(model: nn.Module) -> nn.Linear:
    head = getattr(model, "lm_head", None)
    if not isinstance(head, nn.Linear):
        raise TypeError("could not locate linear lm_head")
    return head


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


def encode(tokenizer, rows: list[str], device: torch.device, max_length: int) -> dict[str, Tensor]:
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


class AddressHead(nn.Module):
    def __init__(self, hidden: int, width: int, rank: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(hidden, rank, bias=False)
        self.out_proj = nn.Linear(rank, width, bias=True)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.out_proj(torch.tanh(self.in_proj(hidden)))


class CorrectionHead(nn.Module):
    def __init__(self, hidden: int, rank: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(hidden, rank, bias=False)
        self.out_proj = nn.Linear(rank, hidden, bias=True)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.out_proj(torch.tanh(self.in_proj(hidden)))


@torch.inference_mode()
def capture_trace(
    model: nn.Module,
    layer: nn.Module,
    encoded: dict[str, Tensor],
    *,
    max_tokens: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    mlp = resolve_mlp(layer)
    mlp_inputs: list[Tensor] = []
    activations: list[Tensor] = []
    mlp_outputs: list[Tensor] = []
    layer_outputs: list[Tensor] = []

    def mlp_pre(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
        mlp_inputs.append(args[0].detach())

    def down_pre(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
        activations.append(args[0].detach())

    def mlp_post(_module: nn.Module, _args: tuple[Tensor, ...], output: Tensor) -> None:
        mlp_outputs.append(output.detach())

    def layer_post(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        layer_outputs.append(hidden.detach())

    handles = [
        mlp.register_forward_pre_hook(mlp_pre),
        mlp.down_proj.register_forward_pre_hook(down_pre),
        mlp.register_forward_hook(mlp_post),
        layer.register_forward_hook(layer_post),
    ]
    try:
        model(**encoded)
    finally:
        for handle in handles:
            handle.remove()

    x = mlp_inputs[0].reshape(-1, mlp_inputs[0].shape[-1]).cpu()
    u = activations[0].reshape(-1, activations[0].shape[-1]).cpu()
    m = mlp_outputs[0].reshape(-1, mlp_outputs[0].shape[-1]).cpu()
    h = layer_outputs[0].reshape(-1, layer_outputs[0].shape[-1]).cpu()
    mask = encoded.get("attention_mask")
    if mask is not None:
        valid = mask.reshape(-1).to(torch.bool).cpu()
        x, u, m, h = x[valid], u[valid], m[valid], h[valid]

    return (
        x[:max_tokens].float(),
        u[:max_tokens].float(),
        m[:max_tokens].float(),
        h[:max_tokens].float(),
    )


def teacher_score(activation: Tensor, mlp: nn.Module) -> Tensor:
    norm = mlp.down_proj.weight.detach().float().cpu().norm(dim=0)
    return activation.abs() * norm[None, :]


def topk_indices(score: Tensor, fraction: float) -> Tensor:
    width = score.shape[-1]
    k = max(1, min(width, round(width * fraction)))
    return torch.topk(score, k=k, dim=-1, sorted=False).indices


def train_address(
    hidden: Tensor,
    teacher: Tensor,
    *,
    rank: int,
    fraction: float,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> AddressHead:
    labels = torch.zeros_like(teacher)
    labels.scatter_(1, topk_indices(teacher, fraction), 1.0)
    positives = labels.sum()
    negatives = labels.numel() - positives
    pos_weight = (negatives / positives.clamp_min(1)).detach()

    torch.manual_seed(seed)
    head = AddressHead(hidden.shape[-1], teacher.shape[-1], rank)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed + 1)
    for _ in range(steps):
        idx = torch.randint(
            hidden.shape[0],
            (min(batch_size, hidden.shape[0]),),
            generator=generator,
        )
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            head(hidden[idx]),
            labels[idx],
            pos_weight=pos_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return head.eval()


def selected_contribution(
    activation: Tensor,
    down_weight: Tensor,
    selected: Tensor,
) -> Tensor:
    rows = []
    for row in range(activation.shape[0]):
        idx = selected[row]
        rows.append(
            torch.nn.functional.linear(
                activation[row, idx],
                down_weight[:, idx],
                bias=None,
            )
        )
    return torch.stack(rows)


def train_correction(
    hidden: Tensor,
    sparse_output: Tensor,
    base: Tensor,
    dense_layer_output: Tensor,
    final_norm: nn.Module,
    *,
    rank: int,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> CorrectionHead:
    torch.manual_seed(seed)
    head = CorrectionHead(hidden.shape[-1], rank)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed + 1)

    final_norm_cpu = final_norm.cpu()
    with torch.no_grad():
        target = final_norm_cpu(dense_layer_output).detach()

    for _ in range(steps):
        idx = torch.randint(
            hidden.shape[0],
            (min(batch_size, hidden.shape[0]),),
            generator=generator,
        )
        candidate_hidden = base[idx] + sparse_output[idx] + head(hidden[idx])
        candidate = final_norm_cpu(candidate_hidden)
        loss = torch.nn.functional.mse_loss(candidate, target[idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return head.eval()


@torch.inference_mode()
def evaluate_trace(
    hidden: Tensor,
    activation: Tensor,
    dense_mlp_output: Tensor,
    dense_layer_output: Tensor,
    mlp: nn.Module,
    final_norm: nn.Module,
    lm_head: nn.Linear,
    address: AddressHead,
    correction: CorrectionHead,
    *,
    fraction: float,
) -> dict[str, float]:
    selected = topk_indices(address(hidden), fraction)
    sparse = selected_contribution(
        activation,
        mlp.down_proj.weight.detach().float().cpu(),
        selected,
    )
    if mlp.down_proj.bias is not None:
        sparse = sparse + mlp.down_proj.bias.detach().float().cpu()
    base = dense_layer_output - dense_mlp_output
    candidate_hidden = base + sparse + correction(hidden)

    norm = final_norm.cpu()
    head = lm_head.cpu()
    dense_logits = head(norm(dense_layer_output))
    candidate_logits = head(norm(candidate_hidden))

    ref_logp = torch.log_softmax(dense_logits, dim=-1)
    cand_logp = torch.log_softmax(candidate_logits, dim=-1)
    p = ref_logp.exp()
    kl = (p * (ref_logp - cand_logp)).sum(dim=-1)
    agreement = dense_logits.argmax(dim=-1) == candidate_logits.argmax(dim=-1)

    normalized_error = (
        (norm(dense_layer_output) - norm(candidate_hidden)).norm(dim=-1)
        / norm(dense_layer_output).norm(dim=-1).clamp_min(1e-12)
    )
    return {
        "mean_kl_nats": float(kl.mean().item()),
        "p95_kl_nats": float(torch.quantile(kl, 0.95).item()),
        "max_kl_nats": float(kl.max().item()),
        "top1_agreement": float(agreement.float().mean().item()),
        "mean_final_normalized_hidden_error": float(normalized_error.mean().item()),
        "p95_final_normalized_hidden_error": float(
            torch.quantile(normalized_error, 0.95).item()
        ),
    }


def metadata_fraction(
    address: AddressHead,
    correction: CorrectionHead,
    mlp: nn.Module,
    fraction: float,
) -> dict[str, float]:
    metadata = sum(p.numel() for p in address.parameters())
    metadata += sum(p.numel() for p in correction.parameters())
    cold = (
        mlp.gate_proj.weight.numel()
        + mlp.up_proj.weight.numel()
        + mlp.down_proj.weight.numel()
    )
    meta_fraction = metadata / cold
    return {
        "resident_metadata_fraction": meta_fraction,
        "selected_payload_fraction": fraction,
        "metadata_plus_selected_fraction": meta_fraction + fraction,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--text-file")
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--fraction", type=float, default=0.10)
    parser.add_argument("--address-rank", type=int, default=8)
    parser.add_argument(
        "--correction-ranks",
        nargs="+",
        type=int,
        default=[4, 8, 16, 32],
    )
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=401)
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
    train_encoded = encode(tokenizer, train_texts, device, args.max_length)
    test_encoded = encode(tokenizer, test_texts, device, args.max_length)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    layers = resolve_layers(model)
    layer = layers[-1]
    mlp = resolve_mlp(layer)
    final_norm = resolve_final_norm(model)
    lm_head = resolve_lm_head(model)

    train_x, train_u, train_m, train_h = capture_trace(
        model,
        layer,
        train_encoded,
        max_tokens=args.max_tokens,
    )
    test_x, test_u, test_m, test_h = capture_trace(
        model,
        layer,
        test_encoded,
        max_tokens=args.max_tokens,
    )

    teacher = teacher_score(train_u, mlp)
    address = train_address(
        train_x,
        teacher,
        rank=args.address_rank,
        fraction=args.fraction,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )

    with torch.no_grad():
        train_selected = topk_indices(address(train_x), args.fraction)
        train_sparse = selected_contribution(
            train_u,
            mlp.down_proj.weight.detach().float().cpu(),
            train_selected,
        )
        if mlp.down_proj.bias is not None:
            train_sparse = train_sparse + mlp.down_proj.bias.detach().float().cpu()
        train_base = train_h - train_m

    rows = []
    for rank in args.correction_ranks:
        correction = train_correction(
            train_x,
            train_sparse,
            train_base,
            train_h,
            final_norm,
            rank=rank,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed + rank,
        )
        rows.append(
            {
                "correction_rank": rank,
                **metadata_fraction(address, correction, mlp, args.fraction),
                **evaluate_trace(
                    test_x,
                    test_u,
                    test_m,
                    test_h,
                    mlp,
                    final_norm,
                    lm_head,
                    address,
                    correction,
                    fraction=args.fraction,
                ),
            }
        )

    payload = {
        "experiment": "hf_predictive_correction_v0",
        "model": args.model,
        "layer": len(layers) - 1,
        "fraction": args.fraction,
        "address_rank": args.address_rank,
        "train_tokens": int(train_x.shape[0]),
        "test_tokens": int(test_x.shape[0]),
        "rows": rows,
        "claim_boundary": (
            "The last-layer correction head is trained on one prompt split to "
            "match the dense final normalized hidden state after exact selected "
            "neuron contributions. Held-out evaluation computes full vocabulary "
            "logit KL. Exact selected activations come from dense traces in this "
            "diagnostic; a deployable kernel must recompute only selected neurons."
        ),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        Path(args.json_out).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
