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
    "How can a risk model estimate whether an approximate computation is safe?",
    "Why should risk prediction be trained on a disjoint split?",
    "What is selective classification with abstention?",
    "How can a model turn uncertainty into a memory fault?",
    "Why is average bandwidth under a quality constraint a useful objective?",
    "Explain how a tiny safety classifier can protect an aggressive sparse path.",
    "What does it mean to calibrate risk rather than confidence?",
    "Why can a cascade outperform one fixed sparse operating point?",
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
    if len(rows) < 16:
        raise ValueError("need at least sixteen texts")
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


class AddressHead(nn.Module):
    def __init__(self, hidden: int, width: int, rank: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(hidden, rank, bias=False)
        self.out_proj = nn.Linear(rank, width, bias=True)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.out_proj(torch.tanh(self.in_proj(hidden)))


class RiskHead(nn.Module):
    def __init__(self, hidden: int, rank: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(hidden, rank, bias=False)
        self.out_proj = nn.Linear(rank, 1, bias=True)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.out_proj(torch.tanh(self.in_proj(hidden))).squeeze(-1)


def topk_indices(score: Tensor, fraction: float) -> Tensor:
    width = score.shape[-1]
    k = max(1, min(width, round(width * fraction)))
    return torch.topk(score, k=k, dim=-1, sorted=False).indices


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
    norm = mlp.down_proj.weight.detach().float().cpu().norm(dim=0)
    teacher = activation.abs() * norm[None, :]
    return hidden, teacher


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


def train_risk(
    hidden: Tensor,
    unsafe: Tensor,
    *,
    rank: int,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> RiskHead:
    torch.manual_seed(seed)
    head = RiskHead(hidden.shape[-1], rank)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    unsafe = unsafe.float()
    positives = unsafe.sum()
    negatives = unsafe.numel() - positives
    pos_weight = (negatives / positives.clamp_min(1)).detach()
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
            unsafe[index],
            pos_weight=pos_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return head.eval()


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


def run_sparse(
    model: nn.Module,
    mlp: nn.Module,
    address: AddressHead,
    encoded: dict[str, Tensor],
    *,
    fraction: float,
) -> tuple[Tensor, Tensor]:
    hidden_sink: list[Tensor] = []
    mask = encoded.get("attention_mask")

    def hook(_module: nn.Module, args: tuple[Tensor, ...], output: Tensor) -> Tensor:
        hidden_device = args[0]
        shape = hidden_device.shape
        hidden = hidden_device.detach().float().cpu().reshape(-1, shape[-1])
        hidden_sink.append(hidden)
        selected = topk_indices(address(hidden), fraction)
        sparse = sparse_exact_output(
            hidden_device.reshape(-1, shape[-1]),
            mlp,
            selected.to(hidden_device.device),
        ).reshape(*shape[:-1], -1)
        return sparse.to(output.dtype)

    handle = mlp.register_forward_hook(hook)
    try:
        logits = model(**encoded).logits.detach().float().cpu()
    finally:
        handle.remove()

    hidden = torch.cat(hidden_sink)
    if mask is not None:
        valid = mask.detach().cpu().reshape(-1).to(torch.bool)
        hidden = hidden[valid]
    return logits, hidden


def calibrate_risk_threshold(
    risk: Tensor,
    safe: Tensor,
    *,
    max_failure_rate: float,
) -> tuple[float, float, float]:
    """Accept lowest predicted risk while respecting observed failure budget."""

    order = torch.argsort(risk)
    safe_ordered = safe[order]
    failures = (~safe_ordered).float().cumsum(dim=0)
    counts = torch.arange(1, risk.numel() + 1, dtype=torch.float32)
    rates = failures / counts
    valid = rates <= max_failure_rate
    if not bool(valid.any()):
        return float("-inf"), 0.0, 0.0
    accepted_count = int(torch.where(valid)[0][-1].item()) + 1
    threshold = float(risk[order[accepted_count - 1]].item())
    accept_rate = accepted_count / risk.numel()
    failure_rate = float(rates[accepted_count - 1].item())
    return threshold, accept_rate, failure_rate


def run_mixed(
    model: nn.Module,
    mlp: nn.Module,
    address: AddressHead,
    risk_head: RiskHead,
    encoded: dict[str, Tensor],
    *,
    fraction: float,
    risk_threshold: float,
) -> tuple[Tensor, float]:
    accepted_sink: list[Tensor] = []
    mask = encoded.get("attention_mask")

    def hook(_module: nn.Module, args: tuple[Tensor, ...], output: Tensor) -> Tensor:
        hidden_device = args[0]
        shape = hidden_device.shape
        hidden = hidden_device.detach().float().cpu().reshape(-1, shape[-1])
        risk = torch.sigmoid(risk_head(hidden))
        accepted = risk <= risk_threshold
        accepted_sink.append(accepted)

        selected = topk_indices(address(hidden), fraction)
        sparse = sparse_exact_output(
            hidden_device.reshape(-1, shape[-1]),
            mlp,
            selected.to(hidden_device.device),
        )
        sparse = sparse.reshape(*shape[:-1], -1)

        dense_flat = output.reshape(-1, output.shape[-1]).clone()
        sparse_flat = sparse.reshape(-1, sparse.shape[-1])
        accepted_device = accepted.to(dense_flat.device)
        dense_flat[accepted_device] = sparse_flat[accepted_device].to(
            dense_flat.dtype
        )
        return dense_flat.reshape_as(output)

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


def metadata_fraction(
    address: AddressHead,
    risk: RiskHead,
    mlp: nn.Module,
) -> float:
    metadata = sum(p.numel() for p in address.parameters())
    metadata += sum(p.numel() for p in risk.parameters())
    cold = (
        mlp.gate_proj.weight.numel()
        + mlp.up_proj.weight.numel()
        + mlp.down_proj.weight.numel()
    )
    return metadata / cold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--text-file")
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--max-train-tokens", type=int, default=512)
    parser.add_argument("--address-rank", type=int, default=8)
    parser.add_argument("--risk-rank", type=int, default=8)
    parser.add_argument("--fraction", type=float, default=0.15)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--kl-tolerance", type=float, default=0.01)
    parser.add_argument("--max-calibration-failure", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=307)
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
    n = len(texts)
    q1, q2, q3 = n // 4, n // 2, (3 * n) // 4
    address_texts = texts[:q1]
    risk_texts = texts[q1:q2]
    calibration_texts = texts[q2:q3]
    test_texts = texts[q3:]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def enc(rows: list[str]) -> dict[str, Tensor]:
        return encode(tokenizer, rows, device, args.max_length)

    address_encoded = enc(address_texts)
    risk_encoded = enc(risk_texts)
    calibration_encoded = enc(calibration_texts)
    test_encoded = enc(test_texts)

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
        address_encoded,
        max_tokens=args.max_train_tokens,
    )
    address = train_address(
        hidden.detach().clone(),
        teacher.detach().clone(),
        rank=args.address_rank,
        fraction=args.fraction,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )

    risk_dense = dense_logits(model, risk_encoded)
    risk_sparse, risk_hidden = run_sparse(
        model,
        mlp,
        address,
        risk_encoded,
        fraction=args.fraction,
    )
    risk_ref = flat_valid(risk_dense, risk_encoded)
    risk_candidate = flat_valid(risk_sparse, risk_encoded)
    risk_kl, risk_agree = token_quality(risk_ref, risk_candidate)
    unsafe = (risk_kl > args.kl_tolerance) | (~risk_agree)
    risk_head = train_risk(
        risk_hidden,
        unsafe,
        rank=args.risk_rank,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed + 1,
    )

    cal_dense = dense_logits(model, calibration_encoded)
    cal_sparse, cal_hidden = run_sparse(
        model,
        mlp,
        address,
        calibration_encoded,
        fraction=args.fraction,
    )
    cal_ref = flat_valid(cal_dense, calibration_encoded)
    cal_candidate = flat_valid(cal_sparse, calibration_encoded)
    cal_kl, cal_agree = token_quality(cal_ref, cal_candidate)
    cal_safe = (cal_kl <= args.kl_tolerance) & cal_agree
    with torch.no_grad():
        cal_risk = torch.sigmoid(risk_head(cal_hidden))
    threshold, cal_accept, cal_failure = calibrate_risk_threshold(
        cal_risk,
        cal_safe,
        max_failure_rate=args.max_calibration_failure,
    )

    test_dense = dense_logits(model, test_encoded)
    test_mixed, test_accept = run_mixed(
        model,
        mlp,
        address,
        risk_head,
        test_encoded,
        fraction=args.fraction,
        risk_threshold=threshold,
    )
    test_ref = flat_valid(test_dense, test_encoded)
    test_candidate = flat_valid(test_mixed, test_encoded)
    test_kl, test_agree = token_quality(test_ref, test_candidate)

    meta = metadata_fraction(address, risk_head, mlp)
    average_payload = test_accept * args.fraction + (1 - test_accept)
    payload = {
        "experiment": "hf_risk_controlled_fallback_v0",
        "model": args.model,
        "layer": args.layer,
        "sparse_fraction": args.fraction,
        "address_rank": args.address_rank,
        "risk_rank": args.risk_rank,
        "kl_tolerance": args.kl_tolerance,
        "max_calibration_failure": args.max_calibration_failure,
        "risk_train_unsafe_fraction": float(unsafe.float().mean().item()),
        "calibration_accept_rate": cal_accept,
        "calibration_failure_rate": cal_failure,
        "test_accept_rate": test_accept,
        "resident_metadata_fraction": meta,
        "average_payload_fraction": average_payload,
        "metadata_plus_average_payload_fraction": meta + average_payload,
        "analytic_cold_byte_reduction_fraction": 1.0 - average_payload,
        "mean_kl_nats": float(test_kl.mean().item()),
        "p95_kl_nats": float(torch.quantile(test_kl, 0.95).item()),
        "max_kl_nats": float(test_kl.max().item()),
        "top1_agreement": float(test_agree.float().mean().item()),
        "claim_boundary": (
            "Address training, risk training, threshold calibration, and testing "
            "use disjoint prompt splits. The learned risk head decides whether to "
            "accept sparse execution or fall back to dense execution. Byte savings "
            "are analytic until a fused selective kernel is profiled."
        ),
    }

    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        Path(args.json_out).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
