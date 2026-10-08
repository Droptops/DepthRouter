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


def act_fn(mlp: nn.Module):
    fn = getattr(mlp, "act_fn", None)
    return fn if callable(fn) else torch.nn.functional.silu


@torch.inference_mode()
def capture_state(
    model: nn.Module,
    mlp: nn.Module,
    input_ids: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    hidden_sink: list[Tensor] = []
    activation_sink: list[Tensor] = []
    output_sink: list[Tensor] = []

    def mlp_pre(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
        hidden_sink.append(args[0][:, -1, :].detach().cpu())

    def down_pre(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
        activation_sink.append(args[0][:, -1, :].detach().cpu())

    def mlp_post(_module: nn.Module, _args: tuple[Tensor, ...], output: Tensor) -> None:
        output_sink.append(output[:, -1, :].detach().cpu())

    handles = [
        mlp.register_forward_pre_hook(mlp_pre),
        mlp.down_proj.register_forward_pre_hook(down_pre),
        mlp.register_forward_hook(mlp_post),
    ]
    try:
        logits = model(input_ids=input_ids).logits[:, -1, :].detach().float().cpu()
    finally:
        for handle in handles:
            handle.remove()

    return (
        hidden_sink[0].float().clone(),
        activation_sink[0].float().clone(),
        output_sink[0].float().clone(),
        logits,
    )


@torch.inference_mode()
def collect_temporal_pairs(
    model: nn.Module,
    mlp: nn.Module,
    tokenizer,
    texts: list[str],
    *,
    decode_steps: int,
    max_length: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    delta_hidden_rows: list[Tensor] = []
    hidden_rows: list[Tensor] = []
    activation_rows: list[Tensor] = []
    delta_output_rows: list[Tensor] = []

    for text in texts:
        prefix = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )["input_ids"].to(device)

        previous_hidden: Tensor | None = None
        previous_output: Tensor | None = None

        for _ in range(decode_steps):
            hidden, activation, output, logits = capture_state(model, mlp, prefix)
            if previous_hidden is not None and previous_output is not None:
                delta_hidden_rows.append(hidden - previous_hidden)
                hidden_rows.append(hidden)
                activation_rows.append(activation)
                delta_output_rows.append(output - previous_output)

            previous_hidden = hidden
            previous_output = output
            token = logits.argmax(dim=-1).to(device)
            prefix = torch.cat([prefix, token[:, None]], dim=1)

    return (
        torch.cat(delta_hidden_rows, dim=0),
        torch.cat(hidden_rows, dim=0),
        torch.cat(activation_rows, dim=0),
        torch.cat(delta_output_rows, dim=0),
    )


def teacher_score(activation: Tensor, mlp: nn.Module) -> Tensor:
    norm = mlp.down_proj.weight.detach().float().cpu().norm(dim=0)
    return activation.abs() * norm[None, :]


def topk_indices(score: Tensor, fraction: float) -> Tensor:
    width = score.shape[-1]
    k = max(1, min(width, round(width * fraction)))
    return torch.topk(score, k=k, dim=-1, sorted=False).indices


def selected_contribution(
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


class DeltaHead(nn.Module):
    def __init__(self, hidden: int, rank: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(hidden, rank, bias=False)
        self.out_proj = nn.Linear(rank, hidden, bias=True)

    def forward(self, delta_hidden: Tensor) -> Tensor:
        return self.out_proj(torch.tanh(self.in_proj(delta_hidden)))


def train_address(
    hidden: Tensor,
    score: Tensor,
    *,
    rank: int,
    fraction: float,
    steps: int,
    lr: float,
    seed: int,
) -> AddressHead:
    labels = torch.zeros_like(score)
    labels.scatter_(1, topk_indices(score, fraction), 1.0)
    positives = labels.sum()
    negatives = labels.numel() - positives
    pos_weight = (negatives / positives.clamp_min(1)).detach()

    torch.manual_seed(seed)
    head = AddressHead(hidden.shape[-1], score.shape[-1], rank)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(steps):
        logits = head(hidden)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=pos_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return head.eval()


def train_delta(
    delta_hidden: Tensor,
    residual_delta: Tensor,
    *,
    rank: int,
    steps: int,
    lr: float,
    seed: int,
) -> DeltaHead:
    torch.manual_seed(seed)
    head = DeltaHead(delta_hidden.shape[-1], rank)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    scale = residual_delta.square().mean().sqrt().clamp_min(1e-8)
    target = residual_delta / scale
    for _ in range(steps):
        prediction = head(delta_hidden) / scale
        loss = torch.nn.functional.mse_loss(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return head.eval()


def sparse_contribution(
    hidden: Tensor,
    mlp: nn.Module,
    selected: Tensor,
) -> Tensor:
    fn = act_fn(mlp)
    rows = []
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


@torch.inference_mode()
def dense_logits(model: nn.Module, input_ids: Tensor) -> Tensor:
    return model(input_ids=input_ids).logits[:, -1, :].detach().float().cpu()


def distribution_metrics(reference: Tensor, candidate: Tensor) -> tuple[Tensor, Tensor]:
    ref_logp = torch.log_softmax(reference, dim=-1)
    cand_logp = torch.log_softmax(candidate, dim=-1)
    p = ref_logp.exp()
    kl = (p * (ref_logp - cand_logp)).sum(dim=-1)
    agreement = reference.argmax(dim=-1) == candidate.argmax(dim=-1)
    return kl, agreement


def metadata_fraction(
    address: AddressHead,
    delta: DeltaHead,
    mlp: nn.Module,
) -> tuple[float, float]:
    hot_params = sum(p.numel() for p in address.parameters())
    hot_params += sum(p.numel() for p in delta.parameters())
    cold_params = (
        mlp.gate_proj.weight.numel()
        + mlp.up_proj.weight.numel()
        + mlp.down_proj.weight.numel()
    )
    metadata = hot_params / cold_params
    cached_state = (2 * mlp.down_proj.out_features) / cold_params
    return metadata, cached_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--text-file")
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--train-decode-steps", type=int, default=8)
    parser.add_argument("--test-decode-steps", type=int, default=8)
    parser.add_argument("--teacher-fraction", type=float, default=0.10)
    parser.add_argument("--address-rank", type=int, default=8)
    parser.add_argument("--delta-rank", type=int, default=32)
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=[0.05, 0.075, 0.10, 0.125],
    )
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=419)
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

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    layers = resolve_layers(model)
    mlp = resolve_mlp(layers[args.layer])

    delta_hidden, hidden, activation, delta_output = collect_temporal_pairs(
        model,
        mlp,
        tokenizer,
        train_texts,
        decode_steps=args.train_decode_steps,
        max_length=args.max_length,
        device=device,
    )
    score = teacher_score(activation, mlp)
    address = train_address(
        hidden,
        score,
        rank=args.address_rank,
        fraction=args.teacher_fraction,
        steps=args.steps,
        lr=args.lr,
        seed=args.seed,
    )
    with torch.no_grad():
        selected = topk_indices(address(hidden), args.teacher_fraction)
    exact = selected_contribution(activation, mlp, selected)
    # Current delta output = sparse innovation + residual temporal delta.
    delta = train_delta(
        delta_hidden,
        delta_output - exact,
        rank=args.delta_rank,
        steps=args.steps,
        lr=args.lr,
        seed=args.seed + 1,
    )

    rows = []
    for fraction in args.fractions:
        all_kl: list[Tensor] = []
        all_agreement: list[Tensor] = []
        exact_rollouts = 0

        for text in test_texts:
            prefix = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
            )["input_ids"].to(device)

            # Dense warm-up establishes the temporal cache.
            previous_hidden, _, previous_sparse_output, warm_logits = capture_state(
                model,
                mlp,
                prefix,
            )
            token = warm_logits.argmax(dim=-1).to(device)
            prefix = torch.cat([prefix, token[:, None]], dim=1)
            prompt_exact = True

            for _ in range(1, args.test_decode_steps):
                dense_hidden_sink: list[Tensor] = []

                def hidden_pre(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
                    dense_hidden_sink.append(args[0][:, -1, :].detach().cpu())

                capture_handle = mlp.register_forward_pre_hook(hidden_pre)
                try:
                    dense = dense_logits(model, prefix)
                finally:
                    capture_handle.remove()
                current_hidden = dense_hidden_sink[0].float()

                k = max(
                    1,
                    round(mlp.down_proj.in_features * fraction),
                )

                state: dict[str, Tensor] = {}

                def sparse_hook(
                    _module: nn.Module,
                    args: tuple[Tensor, ...],
                    output: Tensor,
                ) -> Tensor:
                    hidden_device = args[0][:, -1, :]
                    hidden_cpu = hidden_device.detach().float().cpu()
                    selected_now = torch.topk(
                        address(hidden_cpu),
                        k=k,
                        dim=-1,
                        sorted=False,
                    ).indices
                    exact_now = sparse_contribution(
                        hidden_device,
                        mlp,
                        selected_now.to(hidden_device.device),
                    )
                    temporal = delta(
                        hidden_cpu - previous_hidden,
                    ).to(hidden_device.device)
                    prediction = (
                        previous_sparse_output.to(hidden_device.device)
                        + temporal
                        + exact_now
                    )
                    replaced = output.clone()
                    replaced[:, -1, :] = prediction.to(replaced.dtype)
                    state["hidden"] = hidden_cpu
                    state["output"] = prediction.detach().float().cpu()
                    return replaced

                handle = mlp.register_forward_hook(sparse_hook)
                try:
                    sparse = dense_logits(model, prefix)
                finally:
                    handle.remove()

                kl, agreement = distribution_metrics(dense, sparse)
                all_kl.append(kl)
                all_agreement.append(agreement)
                prompt_exact = prompt_exact and bool(agreement.item())

                previous_hidden = state["hidden"]
                previous_sparse_output = state["output"]
                token = dense.argmax(dim=-1).to(device)
                prefix = torch.cat([prefix, token[:, None]], dim=1)

            exact_rollouts += int(prompt_exact)

        kl = torch.cat(all_kl)
        agreement = torch.cat(all_agreement).float()
        metadata, state_fraction = metadata_fraction(address, delta, mlp)
        # One dense warm-up is amortized across the measured decode horizon.
        sparse_steps = max(args.test_decode_steps - 1, 1)
        payload_fraction = (
            1.0
            + sparse_steps * fraction
        ) / args.test_decode_steps
        rows.append(
            {
                "fraction": fraction,
                "states_evaluated": int(kl.numel()),
                "mean_kl_nats": float(kl.mean().item()),
                "p95_kl_nats": float(torch.quantile(kl, 0.95).item()),
                "max_kl_nats": float(kl.max().item()),
                "top1_agreement": float(agreement.mean().item()),
                "exact_rollout_fraction": exact_rollouts / len(test_texts),
                "resident_metadata_fraction": metadata,
                "cached_temporal_state_fraction": state_fraction,
                "steady_state_selected_payload_fraction": fraction,
                "amortized_payload_fraction_including_warmup": payload_fraction,
                "steady_state_total_fraction": metadata + state_fraction + fraction,
            }
        )

    payload = {
        "experiment": "hf_temporal_predictive_coding_v0",
        "model": args.model,
        "layer": args.layer,
        "train_prompts": len(train_texts),
        "test_prompts": len(test_texts),
        "train_decode_steps": args.train_decode_steps,
        "test_decode_steps": args.test_decode_steps,
        "training_transition_states": int(hidden.shape[0]),
        "teacher_fraction": args.teacher_fraction,
        "address_rank": args.address_rank,
        "delta_rank": args.delta_rank,
        "rows": rows,
        "claim_boundary": (
            "A dense first decode state warms a per-sequence cache containing the "
            "previous hidden and MLP output. Later states predict the temporal MLP "
            "delta from the hidden-state delta and add exact sparse current-neuron "
            "innovations. Evaluation uses held-out dense greedy prefixes. Only one "
            "MLP layer is modified; traffic is analytic until a selective kernel "
            "and real cache residency are profiled."
        ),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        Path(args.json_out).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
