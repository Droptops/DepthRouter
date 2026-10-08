# ruff: noqa: I001
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from hf_temporal_logit_tangent_cache import (
    greedy_effect_indices,
    projected_neuron_effects,
    rmsnorm_logit_jacobian,
    update_cache,
)


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
    "Why should routing be based on changes in the probability distribution?",
    "How can the same answer distribution arise from different internal activations?",
    "What does a local logit tangent say about which neurons matter for the next token?",
    "Why can predictive equivalence be more useful than activation similarity?",
    "How can a branch predictor learn the next exact neural working set?",
    "Why should the previous selected weights be a prior for the next token?",
    "How can hidden-state deltas expose changes in predictive demand?",
    "What would make a learned neural page table cheaper than the pages it selects?",
    "Why is distilling an oracle address different from distilling model outputs?",
    "How can a tiny router learn probability-important computation rather than large activations?",
    "What does it mean to predict the transition of a working set?",
    "Why can a persistence prior lower the entropy of a neural address?",
    "How should a learned prefetcher be evaluated on held-out autoregressive trajectories?",
    "What would count as a deployable version of a logit-tangent oracle?",
    "How can exact cold weights and an approximate hot address coexist safely?",
    "Why does quality per newly moved byte matter more than nominal sparsity?",
    "What is a neural branch predictor?",
    "How can a recurrent address state reduce routing metadata?",
    "Why is the first decode state a different problem from steady-state transitions?",
    "How can a hot address head exploit repeated user workloads?",
    "What does a state transition mean in probability space?",
    "Why might late transformer layers have more predictable working-set transitions?",
    "How do you distinguish a cache hit from a correct address prediction?",
    "What would falsify temporal tangent distillation?",
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
def capture_dense_state(
    model: nn.Module,
    layer: nn.Module,
    input_ids: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    mlp = layer.mlp
    mlp_inputs: list[Tensor] = []
    activations: list[Tensor] = []
    mlp_outputs: list[Tensor] = []
    layer_outputs: list[Tensor] = []

    def mlp_pre(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
        mlp_inputs.append(args[0][:, -1, :].detach())

    def down_pre(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
        activations.append(args[0][:, -1, :].detach())

    def mlp_post(
        _module: nn.Module,
        _args: tuple[Tensor, ...],
        output: Tensor,
    ) -> None:
        mlp_outputs.append(output[:, -1, :].detach())

    def layer_post(
        _module: nn.Module,
        _args: tuple[Tensor, ...],
        output: Any,
    ) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        layer_outputs.append(hidden[:, -1, :].detach())

    handles = [
        mlp.register_forward_pre_hook(mlp_pre),
        mlp.down_proj.register_forward_pre_hook(down_pre),
        mlp.register_forward_hook(mlp_post),
        layer.register_forward_hook(layer_post),
    ]
    try:
        logits = model(input_ids=input_ids).logits[:, -1, :].detach()
    finally:
        for handle in handles:
            handle.remove()

    dense_hidden = layer_outputs[0]
    dense_mlp = mlp_outputs[0]
    return (
        logits.float().cpu(),
        mlp_inputs[0].float().cpu(),
        (dense_hidden - dense_mlp).float().cpu(),
        dense_hidden.float().cpu(),
        activations[0].float().cpu(),
    )


def teacher_selected(
    reference: Tensor,
    dense_hidden: Tensor,
    activation: Tensor,
    mlp: nn.Module,
    norm: nn.Module,
    lm_head: nn.Linear,
    *,
    fraction: float,
    top_logits: int,
    block_size: int,
) -> Tensor:
    top_ids = torch.topk(
        reference,
        k=min(top_logits, reference.shape[-1]),
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
    return greedy_effect_indices(
        effects,
        fraction=fraction,
        block_size=block_size,
    )


class TemporalAddressHead(nn.Module):
    def __init__(self, hidden: int, width: int, rank: int) -> None:
        super().__init__()
        self.current_proj = nn.Linear(hidden, rank, bias=False)
        self.delta_proj = nn.Linear(hidden, rank, bias=False)
        self.out_proj = nn.Linear(rank, width, bias=True)
        self.persistence = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        current: Tensor,
        previous_hidden: Tensor,
        previous_mask: Tensor,
    ) -> Tensor:
        current_latent = torch.tanh(self.current_proj(current))
        delta_latent = torch.tanh(self.delta_proj(current - previous_hidden))
        logits = self.out_proj(current_latent + delta_latent)
        return logits + self.persistence * previous_mask


def labels_from_indices(indices: Tensor, width: int) -> Tensor:
    labels = torch.zeros(indices.shape[0], width, dtype=torch.float32)
    labels.scatter_(1, indices, 1.0)
    return labels


@torch.inference_mode()
def collect_training_states(
    model: nn.Module,
    layer: nn.Module,
    tokenizer,
    texts: list[str],
    *,
    decode_steps: int,
    max_length: int,
    fraction: float,
    top_logits: int,
    block_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    mlp = layer.mlp
    norm = model.model.norm
    lm_head = model.lm_head
    width = mlp.down_proj.in_features

    current_hidden: list[Tensor] = []
    previous_hidden: list[Tensor] = []
    previous_mask: list[Tensor] = []
    labels: list[Tensor] = []

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
            reference, hidden, _base, dense_hidden, activation = capture_dense_state(
                model,
                layer,
                prefix,
            )
            reference = reference.detach().clone()
            hidden = hidden.detach().clone()
            dense_hidden = dense_hidden.detach().clone()
            activation = activation.detach().clone()
            selected = teacher_selected(
                reference,
                dense_hidden,
                activation,
                mlp,
                norm,
                lm_head,
                fraction=fraction,
                top_logits=top_logits,
                block_size=block_size,
            )

            current_hidden.append(hidden)
            if prev_hidden is None or prev_selected is None:
                previous_hidden.append(hidden)
                previous_mask.append(torch.zeros(1, width))
            else:
                previous_hidden.append(prev_hidden)
                previous_mask.append(labels_from_indices(prev_selected, width))
            labels.append(labels_from_indices(selected, width))

            prev_hidden = hidden
            prev_selected = selected
            token = reference.argmax(dim=-1).to(device)
            prefix = torch.cat([prefix, token[:, None]], dim=1)

    return (
        torch.cat(current_hidden).detach().clone(),
        torch.cat(previous_hidden).detach().clone(),
        torch.cat(previous_mask).detach().clone(),
        torch.cat(labels).detach().clone(),
    )


def train_head(
    current: Tensor,
    previous: Tensor,
    previous_mask: Tensor,
    labels: Tensor,
    *,
    rank: int,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> TemporalAddressHead:
    torch.manual_seed(seed)
    head = TemporalAddressHead(current.shape[-1], labels.shape[-1], rank)
    positives = labels.sum()
    negatives = labels.numel() - positives
    pos_weight = (negatives / positives.clamp_min(1)).detach()
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed + 1)

    for _ in range(steps):
        index = torch.randint(
            current.shape[0],
            (min(batch_size, current.shape[0]),),
            generator=generator,
        )
        logits = head(
            current[index],
            previous[index],
            previous_mask[index],
        )
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
    fn = getattr(mlp, "act_fn", torch.nn.functional.silu)
    rows = []
    for row in range(hidden.shape[0]):
        index = selected[row].to(hidden.device)
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
                mlp.down_proj.bias,
            )
        )
    return torch.stack(rows)


def distribution_metrics(reference: Tensor, candidate: Tensor) -> tuple[Tensor, Tensor]:
    ref_logp = torch.log_softmax(reference, dim=-1)
    cand_logp = torch.log_softmax(candidate, dim=-1)
    probability = ref_logp.exp()
    kl = (probability * (ref_logp - cand_logp)).sum(dim=-1)
    agreement = reference.argmax(dim=-1) == candidate.argmax(dim=-1)
    return kl, agreement


def set_recall(predicted: Tensor, target: Tensor) -> float:
    a = set(predicted.tolist())
    b = set(target.tolist())
    return len(a & b) / max(len(b), 1)


def metadata_fraction(head: TemporalAddressHead, mlp: nn.Module) -> float:
    hot = sum(parameter.numel() for parameter in head.parameters())
    cold = sum(parameter.numel() for parameter in mlp.parameters())
    return hot / cold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--text-file")
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--train-decode-steps", type=int, default=8)
    parser.add_argument("--test-decode-steps", type=int, default=10)
    parser.add_argument("--max-train-prompts", type=int, default=20)
    parser.add_argument("--max-test-prompts", type=int, default=20)
    parser.add_argument("--fraction", type=float, default=0.30)
    parser.add_argument("--top-logits", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--cache-multipliers", nargs="+", type=float, default=[1.0, 1.5, 2.0])
    parser.add_argument("--seed", type=int, default=907)
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
    train_texts = texts[:split][: args.max_train_prompts]
    test_texts = texts[split:][: args.max_test_prompts]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    layers = resolve_layers(model)
    layer = layers[-1]
    mlp = layer.mlp
    width = mlp.down_proj.in_features
    k = max(1, min(width, round(width * args.fraction)))

    train_current, train_previous, train_previous_mask, train_labels = collect_training_states(
        model,
        layer,
        tokenizer,
        train_texts,
        decode_steps=args.train_decode_steps,
        max_length=args.max_length,
        fraction=args.fraction,
        top_logits=args.top_logits,
        block_size=args.block_size,
        device=device,
    )
    head = train_head(
        train_current,
        train_previous,
        train_previous_mask,
        train_labels,
        rank=args.rank,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )

    cache_stats = {
        multiplier: {"faults": 0, "steady_faults": 0, "requests": 0}
        for multiplier in args.cache_multipliers
    }
    all_kl: list[Tensor] = []
    all_agreement: list[Tensor] = []
    all_recall: list[float] = []
    exact_rollouts = 0
    states = 0

    for text in test_texts:
        prefix = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
        )["input_ids"].to(device)
        prev_hidden: Tensor | None = None
        prev_selected: Tensor | None = None
        caches = {multiplier: [] for multiplier in args.cache_multipliers}
        prompt_exact = True

        for step in range(args.test_decode_steps):
            reference, hidden, _base, dense_hidden, activation = capture_dense_state(
                model,
                layer,
                prefix,
            )
            reference = reference.detach().clone()
            hidden = hidden.detach().clone()
            dense_hidden = dense_hidden.detach().clone()
            activation = activation.detach().clone()
            target = teacher_selected(
                reference,
                dense_hidden,
                activation,
                mlp,
                model.model.norm,
                model.lm_head,
                fraction=args.fraction,
                top_logits=args.top_logits,
                block_size=args.block_size,
            )[0]

            if prev_hidden is None or prev_selected is None:
                previous_hidden = hidden
                previous_mask = torch.zeros(1, width)
            else:
                previous_hidden = prev_hidden
                previous_mask = labels_from_indices(prev_selected[None, :], width)

            with torch.no_grad():
                address = head(hidden, previous_hidden, previous_mask)
                selected = torch.topk(
                    address,
                    k=k,
                    dim=-1,
                    sorted=False,
                ).indices[0]

            def hook(
                _module: nn.Module,
                hook_args: tuple[Tensor, ...],
                output: Tensor,
                *,
                _selected: Tensor = selected,
            ) -> Tensor:
                hidden_device = hook_args[0][:, -1, :]
                sparse = sparse_exact_output(
                    hidden_device,
                    mlp,
                    _selected[None, :],
                )
                replaced = output.clone()
                replaced[:, -1, :] = sparse.to(replaced.dtype)
                return replaced

            handle = mlp.register_forward_hook(hook)
            try:
                candidate = model(input_ids=prefix).logits[:, -1, :].detach().float().cpu()
            finally:
                handle.remove()

            kl, agreement = distribution_metrics(reference, candidate)
            all_kl.append(kl)
            all_agreement.append(agreement)
            all_recall.append(set_recall(selected, target))
            prompt_exact = prompt_exact and bool(agreement.item())
            states += 1

            for multiplier in args.cache_multipliers:
                budget = max(k, min(width, round(k * multiplier)))
                caches[multiplier], faults = update_cache(
                    caches[multiplier],
                    selected,
                    budget,
                )
                cache_stats[multiplier]["faults"] += faults
                cache_stats[multiplier]["requests"] += k
                if step > 0:
                    cache_stats[multiplier]["steady_faults"] += faults

            prev_hidden = hidden
            prev_selected = selected
            token = reference.argmax(dim=-1).to(device)
            prefix = torch.cat([prefix, token[:, None]], dim=1)

        exact_rollouts += int(prompt_exact)

    kl = torch.cat(all_kl)
    agreement = torch.cat(all_agreement).float()
    metadata = metadata_fraction(head, mlp)
    rows = []
    for multiplier in args.cache_multipliers:
        stats = cache_stats[multiplier]
        steady_states = len(test_texts) * max(args.test_decode_steps - 1, 1)
        steady_fault_fraction = stats["steady_faults"] / (steady_states * width)
        amortized_fault_fraction = stats["faults"] / (states * width)
        rows.append(
            {
                "cache_multiplier": multiplier,
                "cache_fraction_of_mlp_width": min(1.0, args.fraction * multiplier),
                "selected_payload_fraction": args.fraction,
                "resident_address_fraction": metadata,
                "mean_oracle_set_recall": sum(all_recall) / len(all_recall),
                "amortized_physical_fault_fraction": amortized_fault_fraction,
                "steady_state_physical_fault_fraction": steady_fault_fraction,
                "metadata_plus_steady_fault_fraction": metadata + steady_fault_fraction,
                "fault_reduction_vs_stateless_sparse": (
                    1.0 - steady_fault_fraction / args.fraction
                ),
                "cache_request_hit_rate": (
                    1.0 - stats["faults"] / max(stats["requests"], 1)
                ),
                "mean_kl_nats": float(kl.mean().item()),
                "p95_kl_nats": float(torch.quantile(kl, 0.95).item()),
                "max_kl_nats": float(kl.max().item()),
                "top1_agreement": float(agreement.mean().item()),
                "exact_rollout_fraction": exact_rollouts / len(test_texts),
            }
        )

    payload = {
        "experiment": "hf_temporal_tangent_distilled_v0",
        "model": args.model,
        "layer": len(layers) - 1,
        "fraction": args.fraction,
        "top_logit_tangent_dim": args.top_logits,
        "rank": args.rank,
        "train_prompts": len(train_texts),
        "test_prompts": len(test_texts),
        "train_decode_steps": args.train_decode_steps,
        "test_decode_steps": args.test_decode_steps,
        "address_persistence": float(head.persistence.detach().item()),
        "rows": rows,
        "claim_boundary": (
            "The training teacher uses dense local logit-tangent working sets. "
            "At held-out runtime, the address head sees only the current MLP input, "
            "the previous hidden state, and the previous predicted working-set mask. "
            "Selected neurons use exact cold weights. Cache accounting charges only "
            "newly missing selected rows. PyTorch hooks still execute the dense MLP "
            "before replacement, so physical byte savings remain analytic until a "
            "selective kernel and hardware profiler confirm them."
        ),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        Path(args.json_out).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
