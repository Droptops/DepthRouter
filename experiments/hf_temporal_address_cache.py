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
    "Describe a neural branch predictor for the next weight working set.",
    "Why can a cache miss be predicted from the change in hidden state?",
    "How can a persistence prior reduce address entropy?",
    "What is the difference between a logical active set and a physical fault set?",
    "Why should a router predict only the change in the working set?",
    "Describe an LRU cache for exact neural weight pages.",
    "How can temporal locality turn ten percent logical sparsity into three percent HBM traffic?",
    "Why can a small address predictor be worthwhile even if it does no generation?",
    "What does it mean to compile an autoregressive trajectory into cache transitions?",
    "How would you falsify the claim that hidden-state deltas predict future weight demand?",
    "Why is a stable active set useful even when the model's token distribution changes?",
    "Explain why physical bytes matter more than nominal parameter sparsity.",
    "How can retained exact weights protect quality while reducing cold transfers?",
    "What does a neural working-set transition matrix represent?",
    "Why should a prefetcher be judged on misses rather than hit rate alone?",
    "How can a predictive cache cooperate with exact sparse computation?",
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
def capture_state(
    model: nn.Module,
    mlp: nn.Module,
    input_ids: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    hidden_sink: list[Tensor] = []
    activation_sink: list[Tensor] = []

    def mlp_pre(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
        hidden_sink.append(args[0][:, -1, :].detach())

    def down_pre(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
        activation_sink.append(args[0][:, -1, :].detach())

    handles = [
        mlp.register_forward_pre_hook(mlp_pre),
        mlp.down_proj.register_forward_pre_hook(down_pre),
    ]
    try:
        logits = model(input_ids=input_ids).logits[:, -1, :].detach().float().cpu()
    finally:
        for handle in handles:
            handle.remove()

    return (
        hidden_sink[0].detach().float().cpu(),
        activation_sink[0].detach().float().cpu(),
        logits,
    )


def demand_score(activation: Tensor, mlp: nn.Module) -> Tensor:
    norm = mlp.down_proj.weight.detach().float().cpu().norm(dim=0)
    return activation.abs() * norm[None, :]


def topk_indices(score: Tensor, fraction: float) -> Tensor:
    width = score.shape[-1]
    k = max(1, min(width, round(width * fraction)))
    return torch.topk(score, k=k, dim=-1, sorted=False).indices


@torch.inference_mode()
def collect_pairs(
    model: nn.Module,
    mlp: nn.Module,
    tokenizer,
    texts: list[str],
    *,
    decode_steps: int,
    max_length: int,
    fraction: float,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    current_hidden: list[Tensor] = []
    delta_hidden: list[Tensor] = []
    current_labels: list[Tensor] = []
    previous_labels: list[Tensor] = []

    for text in texts:
        prefix = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )["input_ids"].to(device)

        prev_hidden: Tensor | None = None
        prev_selected: Tensor | None = None
        for _ in range(decode_steps):
            hidden, activation, logits = capture_state(model, mlp, prefix)
            selected = topk_indices(demand_score(activation, mlp), fraction)
            if prev_hidden is not None and prev_selected is not None:
                current_hidden.append(hidden)
                delta_hidden.append(hidden - prev_hidden)
                current_labels.append(selected)
                previous_labels.append(prev_selected)

            prev_hidden = hidden
            prev_selected = selected
            next_token = logits.argmax(dim=-1).to(device)
            prefix = torch.cat([prefix, next_token[:, None]], dim=1)

    return (
        torch.cat(current_hidden, dim=0).detach().clone(),
        torch.cat(delta_hidden, dim=0).detach().clone(),
        torch.cat(current_labels, dim=0).detach().clone(),
        torch.cat(previous_labels, dim=0).detach().clone(),
    )


class AddressHead(nn.Module):
    def __init__(self, hidden: int, width: int, rank: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(hidden, rank, bias=False)
        self.out_proj = nn.Linear(rank, width, bias=True)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.out_proj(torch.tanh(self.in_proj(hidden)))


def labels_from_indices(indices: Tensor, width: int) -> Tensor:
    labels = torch.zeros(indices.shape[0], width, dtype=torch.float32)
    labels.scatter_(1, indices, 1.0)
    return labels


def train_head(
    features: Tensor,
    selected: Tensor,
    *,
    width: int,
    rank: int,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> AddressHead:
    labels = labels_from_indices(selected, width)
    positives = labels.sum()
    negatives = labels.numel() - positives
    pos_weight = (negatives / positives.clamp_min(1)).detach()

    torch.manual_seed(seed)
    head = AddressHead(features.shape[-1], width, rank)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed + 1)
    for _ in range(steps):
        index = torch.randint(
            features.shape[0],
            (min(batch_size, features.shape[0]),),
            generator=generator,
        )
        logits = head(features[index])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            labels[index],
            pos_weight=pos_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return head.eval()


def tune_persistence_bias(
    delta_head: AddressHead,
    delta_hidden: Tensor,
    current: Tensor,
    previous: Tensor,
    *,
    width: int,
    fraction: float,
    biases: list[float],
) -> float:
    k = max(1, min(width, round(width * fraction)))
    current_labels = labels_from_indices(current, width)
    best_bias = biases[0]
    best_recall = -1.0
    with torch.no_grad():
        base = delta_head(delta_hidden)
        for bias in biases:
            score = base.clone()
            score.scatter_add_(
                1,
                previous,
                torch.full_like(previous, float(bias), dtype=score.dtype),
            )
            selected = torch.topk(score, k=k, dim=-1, sorted=False).indices
            labels = labels_from_indices(selected, width)
            recall = float(
                (labels * current_labels).sum(dim=1).div(k).mean().item()
            )
            if recall > best_recall:
                best_recall = recall
                best_bias = bias
    return best_bias


def sparse_exact_output(hidden: Tensor, mlp: nn.Module, selected: Tensor) -> Tensor:
    fn = getattr(mlp, "act_fn", torch.nn.functional.silu)
    rows = []
    for row in range(hidden.shape[0]):
        idx = selected[row].to(hidden.device)
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


def update_cache(cache: list[int], selected: Tensor, budget: int) -> tuple[list[int], int]:
    selected_list = selected.tolist()
    cache_set = set(cache)
    faults = sum(1 for index in selected_list if index not in cache_set)

    # Newest requested pages become MRU; retain older resident pages up to budget.
    ordered = []
    seen: set[int] = set()
    for index in selected_list + cache:
        if index in seen:
            continue
        seen.add(index)
        ordered.append(index)
        if len(ordered) >= budget:
            break
    return ordered, faults


def quality(reference: Tensor, candidate: Tensor) -> tuple[Tensor, Tensor]:
    ref_logp = torch.log_softmax(reference, dim=-1)
    cand_logp = torch.log_softmax(candidate, dim=-1)
    probability = ref_logp.exp()
    kl = (probability * (ref_logp - cand_logp)).sum(dim=-1)
    agree = reference.argmax(dim=-1) == candidate.argmax(dim=-1)
    return kl, agree


def set_recall(predicted: Tensor, target: Tensor) -> float:
    a = set(predicted.tolist())
    b = set(target.tolist())
    return len(a & b) / max(len(b), 1)


def metadata_fraction(
    full_head: AddressHead,
    delta_head: AddressHead,
    mlp: nn.Module,
) -> float:
    hot = sum(p.numel() for p in full_head.parameters())
    hot += sum(p.numel() for p in delta_head.parameters())
    cold = sum(p.numel() for p in mlp.parameters())
    return hot / cold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--text-file")
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--train-decode-steps", type=int, default=10)
    parser.add_argument("--test-decode-steps", type=int, default=10)
    parser.add_argument("--fraction", type=float, default=0.10)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--steps", type=int, default=350)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--cache-multipliers", nargs="+", type=float, default=[1.0, 1.5, 2.0])
    parser.add_argument("--persistence-biases", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--seed", type=int, default=809)
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
    if args.layer < 0 or args.layer >= len(layers):
        raise ValueError("layer index out of range")
    mlp = resolve_mlp(layers[args.layer])
    width = mlp.down_proj.in_features
    k = max(1, min(width, round(width * args.fraction)))

    train_hidden, train_delta, train_current, train_previous = collect_pairs(
        model,
        mlp,
        tokenizer,
        train_texts,
        decode_steps=args.train_decode_steps,
        max_length=args.max_length,
        fraction=args.fraction,
        device=device,
    )

    full_head = train_head(
        train_hidden,
        train_current,
        width=width,
        rank=args.rank,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )
    delta_head = train_head(
        train_delta,
        train_current,
        width=width,
        rank=args.rank,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed + 1,
    )
    persistence_bias = tune_persistence_bias(
        delta_head,
        train_delta,
        train_current,
        train_previous,
        width=width,
        fraction=args.fraction,
        biases=args.persistence_biases,
    )

    cache_stats = {
        multiplier: {"faults": 0, "requests": 0}
        for multiplier in args.cache_multipliers
    }
    all_kl: list[Tensor] = []
    all_agree: list[Tensor] = []
    active_recall: list[float] = []
    exact_rollouts = 0

    for text in test_texts:
        prefix = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
        )["input_ids"].to(device)

        previous_hidden: Tensor | None = None
        previous_selected: Tensor | None = None
        caches = {multiplier: [] for multiplier in args.cache_multipliers}
        prompt_exact = True

        for _ in range(args.test_decode_steps):
            _dense_hidden, dense_activation, dense_logits = capture_state(
                model,
                mlp,
                prefix,
            )
            teacher = topk_indices(demand_score(dense_activation, mlp), args.fraction)[0]

            state: dict[str, Tensor] = {}

            def hook(
                _module: nn.Module,
                hook_args: tuple[Tensor, ...],
                output: Tensor,
                *,
                _previous_hidden: Tensor | None = previous_hidden,
                _previous_selected: Tensor | None = previous_selected,
                _state: dict[str, Tensor] = state,
            ) -> Tensor:
                hidden_device = hook_args[0][:, -1, :]
                hidden_cpu = hidden_device.detach().float().cpu()
                if _previous_hidden is None or _previous_selected is None:
                    score = full_head(hidden_cpu)
                else:
                    score = delta_head(hidden_cpu - _previous_hidden)
                    bonus = torch.zeros_like(score)
                    bonus.scatter_(
                        1,
                        _previous_selected[None, :],
                        float(persistence_bias),
                    )
                    score = score + bonus
                selected = torch.topk(score, k=k, dim=-1, sorted=False).indices
                sparse = sparse_exact_output(hidden_device, mlp, selected)
                replaced = output.clone()
                replaced[:, -1, :] = sparse.to(replaced.dtype)
                _state["hidden"] = hidden_cpu[0]
                _state["selected"] = selected[0].detach().cpu()
                return replaced

            handle = mlp.register_forward_hook(hook)
            try:
                sparse_logits = model(input_ids=prefix).logits[:, -1, :].detach().float().cpu()
            finally:
                handle.remove()

            selected = state["selected"]
            active_recall.append(set_recall(selected, teacher))
            kl, agree = quality(dense_logits, sparse_logits)
            all_kl.append(kl)
            all_agree.append(agree)
            prompt_exact = prompt_exact and bool(agree.item())

            for multiplier in args.cache_multipliers:
                budget = max(k, min(width, round(k * multiplier)))
                caches[multiplier], faults = update_cache(
                    caches[multiplier],
                    selected,
                    budget,
                )
                cache_stats[multiplier]["faults"] += faults
                cache_stats[multiplier]["requests"] += k

            previous_hidden = state["hidden"]
            previous_selected = selected
            token = dense_logits.argmax(dim=-1).to(device)
            prefix = torch.cat([prefix, token[:, None]], dim=1)

        exact_rollouts += int(prompt_exact)

    kl = torch.cat(all_kl)
    agree = torch.cat(all_agree).float()
    metadata = metadata_fraction(full_head, delta_head, mlp)
    rows = []
    for multiplier in args.cache_multipliers:
        faults = cache_stats[multiplier]["faults"]
        requests = cache_stats[multiplier]["requests"]
        per_state_fault_fraction = faults / (len(all_kl) * width)
        rows.append(
            {
                "cache_multiplier": multiplier,
                "cache_fraction_of_mlp_width": min(1.0, args.fraction * multiplier),
                "selected_payload_fraction": args.fraction,
                "resident_address_fraction": metadata,
                "mean_predicted_active_set_recall": sum(active_recall) / len(active_recall),
                "mean_kl_nats": float(kl.mean().item()),
                "p95_kl_nats": float(torch.quantile(kl, 0.95).item()),
                "max_kl_nats": float(kl.max().item()),
                "top1_agreement": float(agree.mean().item()),
                "exact_rollout_fraction": exact_rollouts / len(test_texts),
                "physical_fault_fraction_of_full_mlp": per_state_fault_fraction,
                "fault_reduction_vs_stateless_selected_payload": (
                    1.0 - per_state_fault_fraction / args.fraction
                ),
                "cache_request_hit_rate": 1.0 - faults / max(requests, 1),
                "metadata_plus_physical_fault_fraction": metadata + per_state_fault_fraction,
            }
        )

    payload = {
        "experiment": "hf_temporal_address_cache_v0",
        "model": args.model,
        "layer": args.layer,
        "fraction": args.fraction,
        "rank": args.rank,
        "train_prompts": len(train_texts),
        "test_prompts": len(test_texts),
        "train_decode_steps": args.train_decode_steps,
        "test_decode_steps": args.test_decode_steps,
        "persistence_bias": persistence_bias,
        "rows": rows,
        "claim_boundary": (
            "The first decode state uses a tiny full address head; later states use "
            "a tiny hidden-delta address head plus the previous selected set as a "
            "persistence prior. Exact selected cold weights are evaluated, while an "
            "LRU-like resident cache accounts only newly missing selected pages as "
            "physical faults. The PyTorch hook still runs the dense MLP before "
            "replacement, so physical fault fractions are analytic until a true "
            "selective kernel and hardware profiler confirm them."
        ),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        Path(args.json_out).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
