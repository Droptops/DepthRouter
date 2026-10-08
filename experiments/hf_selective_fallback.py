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
        inputs.append(args[0].detach())

    def down_pre(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
        activations.append(args[0].detach())

    h1 = mlp.register_forward_pre_hook(mlp_pre)
    h2 = mlp.down_proj.register_forward_pre_hook(down_pre)
    try:
        model(**encoded)
    finally:
        h1.remove()
        h2.remove()

    hidden = inputs[0].reshape(-1, inputs[0].shape[-1]).cpu()
    activation = activations[0].reshape(-1, activations[0].shape[-1]).cpu()
    mask = encoded.get("attention_mask")
    if mask is not None:
        valid = mask.reshape(-1).to(torch.bool).cpu()
        hidden = hidden[valid]
        activation = activation[valid]

    hidden = hidden[:max_tokens].float()
    activation = activation[:max_tokens].float()
    down_norm = mlp.down_proj.weight.detach().float().cpu().norm(dim=0)
    teacher = activation.abs() * down_norm[None, :]
    return hidden, teacher


class AddressHead(nn.Module):
    def __init__(self, hidden: int, width: int, rank: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(hidden, rank, bias=False)
        self.out_proj = nn.Linear(rank, width, bias=True)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.out_proj(torch.tanh(self.in_proj(hidden)))


def topk_indices(score: Tensor, fraction: float) -> Tensor:
    width = score.shape[-1]
    k = max(1, min(width, round(width * fraction)))
    return torch.topk(score, k=k, dim=-1, sorted=True).indices


def train_head(
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
    selected = topk_indices(teacher, fraction)
    labels = torch.zeros_like(teacher)
    labels.scatter_(1, selected, 1.0)
    positives = labels.sum()
    negatives = labels.numel() - positives
    pos_weight = (negatives / positives.clamp_min(1)).detach()

    torch.manual_seed(seed)
    head = AddressHead(hidden.shape[-1], teacher.shape[-1], rank)
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


def quantized_onebit_scores(hidden: Tensor, mlp: nn.Module) -> Tensor:
    def sketch(weight: Tensor) -> tuple[Tensor, Tensor]:
        w = weight.detach().float().cpu()
        scale = w.abs().mean(dim=1).clamp_min(1e-12)
        code = torch.where(w >= 0, torch.ones_like(w), -torch.ones_like(w))
        return code, scale

    gate_code, gate_scale = sketch(mlp.gate_proj.weight)
    up_code, up_scale = sketch(mlp.up_proj.weight)
    gate = (hidden.float() @ gate_code.T) * gate_scale
    up = (hidden.float() @ up_code.T) * up_scale
    fn = getattr(mlp, "act_fn", torch.nn.functional.silu)
    activation = fn(gate) * up
    norm = mlp.down_proj.weight.detach().float().cpu().norm(dim=0)
    return activation.abs() * norm[None, :]


def confidence_metrics(
    head_score: Tensor,
    lowbit_score: Tensor,
    *,
    fraction: float,
) -> dict[str, Tensor]:
    width = head_score.shape[-1]
    k = max(1, min(width - 1, round(width * fraction)))

    ordered = torch.topk(head_score, k=k + 1, dim=-1, sorted=True).values
    margin = ordered[:, k - 1] - ordered[:, k]

    head_idx = torch.topk(head_score, k=k, dim=-1, sorted=False).indices
    low_idx = torch.topk(lowbit_score, k=k, dim=-1, sorted=False).indices
    overlaps = []
    for row in range(head_idx.shape[0]):
        a = set(head_idx[row].tolist())
        b = set(low_idx[row].tolist())
        overlaps.append(len(a & b) / k)
    agreement = torch.tensor(overlaps, dtype=torch.float32)

    return {
        "head_margin": margin.float(),
        "address_agreement": agreement,
    }


def sparse_exact_output(hidden: Tensor, mlp: nn.Module, selected: Tensor) -> Tensor:
    fn = getattr(mlp, "act_fn", torch.nn.functional.silu)
    rows = []
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
                mlp.down_proj.bias,
            )
        )
    return torch.stack(rows)


@torch.inference_mode()
def dense_logits(model: nn.Module, encoded: dict[str, Tensor]) -> Tensor:
    return model(**encoded).logits.detach().float().cpu()


def flat_valid(logits: Tensor, encoded: dict[str, Tensor]) -> Tensor:
    flat = logits.reshape(-1, logits.shape[-1])
    mask = encoded.get("attention_mask")
    if mask is None:
        return flat
    valid = mask.detach().cpu().reshape(-1).to(torch.bool)
    return flat[valid]


def token_quality(reference: Tensor, candidate: Tensor) -> tuple[Tensor, Tensor]:
    ref_logp = torch.log_softmax(reference, dim=-1)
    cand_logp = torch.log_softmax(candidate, dim=-1)
    p = ref_logp.exp()
    kl = (p * (ref_logp - cand_logp)).sum(dim=-1)
    agree = reference.argmax(dim=-1) == candidate.argmax(dim=-1)
    return kl, agree


def run_sparse_collect(
    model: nn.Module,
    mlp: nn.Module,
    head: AddressHead,
    encoded: dict[str, Tensor],
    *,
    fraction: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    confidence_sink: dict[str, list[Tensor]] = {
        "head_margin": [],
        "address_agreement": [],
    }
    mask = encoded.get("attention_mask")

    def hook(_module: nn.Module, args: tuple[Tensor, ...], output: Tensor) -> Tensor:
        hidden_device = args[0]
        shape = hidden_device.shape
        hidden = hidden_device.detach().float().cpu().reshape(-1, shape[-1])
        head_score = head(hidden)
        lowbit = quantized_onebit_scores(hidden, mlp)
        metrics = confidence_metrics(head_score, lowbit, fraction=fraction)
        for key, value in metrics.items():
            confidence_sink[key].append(value)

        k = max(1, round(head_score.shape[-1] * fraction))
        selected = torch.topk(head_score, k=k, dim=-1, sorted=False).indices
        sparse = sparse_exact_output(
            hidden_device.reshape(-1, shape[-1]),
            mlp,
            selected.to(hidden_device.device),
        ).reshape(*shape[:-1], -1)

        replaced = output.clone()
        replaced[...] = sparse.to(replaced.dtype)
        return replaced

    handle = mlp.register_forward_hook(hook)
    try:
        logits = model(**encoded).logits.detach().float().cpu()
    finally:
        handle.remove()

    out = {}
    for key, chunks in confidence_sink.items():
        values = torch.cat(chunks)
        if mask is not None:
            valid = mask.detach().cpu().reshape(-1).to(torch.bool)
            values = values[valid]
        out[key] = values
    return logits, out


def calibrate_threshold(
    confidence: Tensor,
    safe: Tensor,
    *,
    max_failure_rate: float,
) -> tuple[float, float, float]:
    """Choose threshold maximizing acceptance under calibration failure budget."""

    candidates = torch.unique(confidence).sort().values
    best = (float("inf"), 0.0, 0.0)
    for threshold in candidates.tolist():
        accepted = confidence >= threshold
        count = int(accepted.sum().item())
        if count == 0:
            continue
        failure = float((~safe[accepted]).float().mean().item())
        rate = count / confidence.numel()
        if failure <= max_failure_rate and rate > best[1]:
            best = (float(threshold), rate, failure)
    return best


def run_selective(
    model: nn.Module,
    mlp: nn.Module,
    head: AddressHead,
    encoded: dict[str, Tensor],
    *,
    fraction: float,
    confidence_mode: str,
    threshold: float,
) -> tuple[Tensor, float]:
    accepted_sink: list[Tensor] = []
    mask = encoded.get("attention_mask")

    def hook(_module: nn.Module, args: tuple[Tensor, ...], output: Tensor) -> Tensor:
        hidden_device = args[0]
        shape = hidden_device.shape
        hidden = hidden_device.detach().float().cpu().reshape(-1, shape[-1])
        head_score = head(hidden)
        lowbit = quantized_onebit_scores(hidden, mlp)
        metrics = confidence_metrics(head_score, lowbit, fraction=fraction)
        confidence = metrics[confidence_mode]
        accepted = confidence >= threshold
        accepted_sink.append(accepted)

        k = max(1, round(head_score.shape[-1] * fraction))
        selected = torch.topk(head_score, k=k, dim=-1, sorted=False).indices
        sparse = sparse_exact_output(
            hidden_device.reshape(-1, shape[-1]),
            mlp,
            selected.to(hidden_device.device),
        )
        sparse = sparse.reshape(*shape[:-1], -1)

        replaced = output.clone()
        flat_replaced = replaced.reshape(-1, replaced.shape[-1])
        flat_sparse = sparse.reshape(-1, sparse.shape[-1])
        accepted_device = accepted.to(flat_replaced.device)
        flat_replaced[accepted_device] = flat_sparse[accepted_device].to(
            flat_replaced.dtype
        )
        return flat_replaced.reshape_as(replaced)

    handle = mlp.register_forward_hook(hook)
    try:
        logits = model(**encoded).logits.detach().float().cpu()
    finally:
        handle.remove()

    accepted = torch.cat(accepted_sink)
    if mask is not None:
        valid = mask.detach().cpu().reshape(-1).to(torch.bool)
        accepted = accepted[valid]
    return logits, float(accepted.float().mean().item())


def metadata_fraction(head: AddressHead, mlp: nn.Module) -> float:
    head_bytes = sum(p.numel() for p in head.parameters()) * 2
    gate = mlp.gate_proj
    up = mlp.up_proj
    down = mlp.down_proj
    cold_params = gate.weight.numel() + up.weight.numel() + down.weight.numel()
    cold_bytes = cold_params * 2

    onebit_bits = gate.weight.numel() + up.weight.numel()
    onebit_bytes = (onebit_bits + 7) // 8 + down.in_features * 3 * 2
    return (head_bytes + onebit_bytes) / cold_bytes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--text-file")
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--max-train-tokens", type=int, default=512)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--fraction", type=float, default=0.15)
    parser.add_argument("--steps", type=int, default=220)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--kl-tolerance", type=float, default=0.02)
    parser.add_argument("--max-calibration-failure", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=173)
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
    one = len(texts) // 3
    two = 2 * one
    train_texts = texts[:one]
    calibration_texts = texts[one:two]
    test_texts = texts[two:]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_encoded = encode(tokenizer, train_texts, device, args.max_length)
    calibration_encoded = encode(
        tokenizer,
        calibration_texts,
        device,
        args.max_length,
    )
    test_encoded = encode(tokenizer, test_texts, device, args.max_length)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    layers = resolve_layers(model)
    mlp = resolve_mlp(layers[args.layer])

    hidden, teacher = capture_teacher(
        model,
        mlp,
        train_encoded,
        max_tokens=args.max_train_tokens,
    )
    hidden = hidden.detach().clone()
    teacher = teacher.detach().clone()
    head = train_head(
        hidden,
        teacher,
        rank=args.rank,
        fraction=args.fraction,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )

    calibration_dense = dense_logits(model, calibration_encoded)
    calibration_sparse, calibration_confidence = run_sparse_collect(
        model,
        mlp,
        head,
        calibration_encoded,
        fraction=args.fraction,
    )
    cal_ref = flat_valid(calibration_dense, calibration_encoded)
    cal_sparse = flat_valid(calibration_sparse, calibration_encoded)
    cal_kl, cal_agree = token_quality(cal_ref, cal_sparse)
    safe = (cal_kl <= args.kl_tolerance) & cal_agree

    test_dense = dense_logits(model, test_encoded)
    rows = []
    meta = metadata_fraction(head, mlp)
    for confidence_mode in ("head_margin", "address_agreement"):
        threshold, cal_accept, cal_failure = calibrate_threshold(
            calibration_confidence[confidence_mode],
            safe,
            max_failure_rate=args.max_calibration_failure,
        )
        if not torch.isfinite(torch.tensor(threshold)):
            rows.append(
                {
                    "confidence_mode": confidence_mode,
                    "calibrated": False,
                }
            )
            continue

        test_mixed, test_accept = run_selective(
            model,
            mlp,
            head,
            test_encoded,
            fraction=args.fraction,
            confidence_mode=confidence_mode,
            threshold=threshold,
        )
        ref = flat_valid(test_dense, test_encoded)
        mixed = flat_valid(test_mixed, test_encoded)
        kl, agree = token_quality(ref, mixed)

        payload_fraction = (
            test_accept * args.fraction + (1.0 - test_accept)
        )
        rows.append(
            {
                "confidence_mode": confidence_mode,
                "calibrated": True,
                "threshold": threshold,
                "calibration_accept_rate": cal_accept,
                "calibration_failure_rate": cal_failure,
                "test_accept_rate": test_accept,
                "resident_metadata_fraction": meta,
                "average_payload_fraction": payload_fraction,
                "metadata_plus_average_payload_fraction": meta + payload_fraction,
                "analytic_cold_byte_reduction_fraction": 1.0 - payload_fraction,
                "mean_kl_nats": float(kl.mean().item()),
                "p95_kl_nats": float(torch.quantile(kl, 0.95).item()),
                "max_kl_nats": float(kl.max().item()),
                "top1_agreement": float(agree.float().mean().item()),
            }
        )

    payload = {
        "experiment": "hf_selective_fallback_v0",
        "model": args.model,
        "layer": args.layer,
        "rank": args.rank,
        "sparse_fraction": args.fraction,
        "kl_tolerance": args.kl_tolerance,
        "max_calibration_failure": args.max_calibration_failure,
        "train_prompts": len(train_texts),
        "calibration_prompts": len(calibration_texts),
        "test_prompts": len(test_texts),
        "rows": rows,
        "claim_boundary": (
            "A tiny address head proposes sparse execution. A confidence statistic "
            "calibrated on disjoint prompts decides whether to use the sparse path "
            "or retain dense execution. Reported byte reduction is analytic; the "
            "diagnostic hook still executes the dense MLP before replacement."
        ),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        Path(args.json_out).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
