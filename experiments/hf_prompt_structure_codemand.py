# ruff: noqa: I001
from __future__ import annotations

import argparse
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


SEMANTIC_CASES = [
    {
        "id": "backend_language",
        "a": "Python",
        "b": "Rust",
        "goal": "a latency-sensitive backend service",
        "criteria": "runtime performance, memory safety, ecosystem maturity, and developer productivity",
    },
    {
        "id": "database",
        "a": "PostgreSQL",
        "b": "DynamoDB",
        "goal": "a transactional application expected to scale quickly",
        "criteria": "consistency, operational burden, query flexibility, and horizontal scaling",
    },
    {
        "id": "vehicle",
        "a": "an electric vehicle",
        "b": "a hybrid vehicle",
        "goal": "a long daily commute",
        "criteria": "energy cost, refueling or charging convenience, maintenance, and reliability",
    },
    {
        "id": "energy",
        "a": "solar plus batteries",
        "b": "grid electricity",
        "goal": "powering a remote cabin",
        "criteria": "upfront cost, reliability, maintenance, and long-term operating cost",
    },
    {
        "id": "portfolio",
        "a": "a broad index fund",
        "b": "a bond ladder",
        "goal": "preserving capital over a medium-term horizon",
        "criteria": "volatility, liquidity, inflation exposure, and expected return",
    },
    {
        "id": "workstation_cooling",
        "a": "air cooling",
        "b": "liquid cooling",
        "goal": "a high-performance workstation",
        "criteria": "thermals, acoustic noise, reliability, maintenance, and cost",
    },
    {
        "id": "travel",
        "a": "a train",
        "b": "a flight",
        "goal": "a 500-mile business trip",
        "criteria": "door-to-door time, reliability, cost, and ability to work while traveling",
    },
    {
        "id": "cooking",
        "a": "induction cooking",
        "b": "gas cooking",
        "goal": "a home kitchen",
        "criteria": "temperature control, safety, cleanup, compatibility, and operating cost",
    },
    {
        "id": "battery",
        "a": "lithium nickel-manganese-cobalt batteries",
        "b": "lithium iron phosphate batteries",
        "goal": "stationary energy storage",
        "criteria": "cycle life, safety, energy density, cost, and thermal stability",
    },
    {
        "id": "api",
        "a": "REST",
        "b": "GraphQL",
        "goal": "a mobile application API",
        "criteria": "bandwidth efficiency, caching, client flexibility, operational complexity, and tooling",
    },
]


STRUCTURES = {
    "direct": (
        "Choose between {a} and {b} for {goal}. Consider {criteria}. "
        "State the better choice and give one concise reason."
    ),
    "compare_then_choose": (
        "Compare {a} versus {b} for {goal} across these criteria: {criteria}. "
        "After the comparison, choose one option and give one concise reason."
    ),
    "multiple_choice": (
        "For {goal}, which option is better when considering {criteria}? "
        "A) {a}  B) {b}. Answer A or B first, then give one concise reason."
    ),
    "json_schema": (
        "Evaluate {a} and {b} for {goal} using {criteria}. "
        'Return only JSON with keys "choice" and "reason"; "choice" must be one of the two options.'
    ),
    "ordered_fields": (
        "Decision task for {goal}. Option 1: {a}. Option 2: {b}. "
        "Criteria: {criteria}. Respond in exactly this order: "
        "Criteria assessment -> Comparison -> Choice -> One-sentence reason."
    ),
    "role_instruction": (
        "Act as a neutral decision evaluator. Your task is to choose between {a} and {b} "
        "for {goal}. Use only these criteria: {criteria}. "
        "Give the choice first and then one concise reason."
    ),
}


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
def capture_final_layer(
    model: nn.Module,
    layer: nn.Module,
    input_ids: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    mlp = layer.mlp
    down_inputs: list[Tensor] = []
    mlp_outputs: list[Tensor] = []
    layer_outputs: list[Tensor] = []

    def down_pre(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
        down_inputs.append(args[0][:, -1, :].detach())

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
        (dense_hidden - dense_mlp).float().cpu(),
        dense_hidden.float().cpu(),
        down_inputs[0].float().cpu(),
    )


def rmsnorm_logit_jacobian(
    dense_hidden: Tensor,
    top_ids: Tensor,
    norm: nn.Module,
    lm_head: nn.Linear,
) -> Tensor:
    h = dense_hidden.float()
    norm_weight = norm.weight.detach().float().cpu()
    head = lm_head.weight.detach().float().cpu()
    selected_head = head[top_ids]
    a = selected_head * norm_weight[None, None, :]

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
    return (
        a / radius[:, None, None]
        - dot[:, :, None]
        * h[:, None, :]
        / (float(d) * radius[:, None, None].pow(3))
    )


def projected_neuron_effects(
    activation: Tensor,
    down_weight: Tensor,
    jacobian: Tensor,
) -> Tensor:
    basis = torch.einsum(
        "nd,bmd->bnm",
        down_weight.T.float().cpu(),
        jacobian.float(),
    )
    return activation.float()[:, :, None] * basis


def greedy_effect_indices(
    effects: Tensor,
    *,
    fraction: float,
    block_size: int,
) -> Tensor:
    target = effects.sum(dim=1)
    residual = target.clone()
    batch, width, _ = effects.shape
    k = max(1, min(width, round(width * fraction)))
    selected = torch.zeros(batch, width, dtype=torch.bool)
    chunks: list[Tensor] = []
    effect_norm_sq = effects.square().sum(dim=-1)

    chosen = 0
    while chosen < k:
        take = min(block_size, k - chosen)
        dot = torch.einsum("bm,bnm->bn", residual, effects)
        gain = 2.0 * dot - effect_norm_sq
        gain = gain.masked_fill(selected, float("-inf"))
        index = torch.topk(gain, k=take, dim=-1, sorted=False).indices
        selected.scatter_(1, index, True)
        chunks.append(index)
        picked = torch.gather(
            effects,
            1,
            index[:, :, None].expand(-1, -1, effects.shape[-1]),
        )
        residual = residual - picked.sum(dim=1)
        chosen += take

    return torch.cat(chunks, dim=1)


def sparse_hidden(
    base_hidden: Tensor,
    activation: Tensor,
    down_weight: Tensor,
    selected: Tensor,
) -> Tensor:
    wt = down_weight.T.detach().float().cpu()
    rows = []
    for row in range(activation.shape[0]):
        index = selected[row]
        contribution = (activation[row, index, None] * wt[index]).sum(dim=0)
        rows.append(base_hidden[row] + contribution)
    return torch.stack(rows)


def exact_logits_from_hidden(
    hidden: Tensor,
    norm: nn.Module,
    lm_head: nn.Linear,
) -> Tensor:
    norm_cpu = norm.cpu()
    head_cpu = lm_head.cpu()
    normalized = norm_cpu(hidden.to(norm_cpu.weight.dtype))
    return head_cpu(normalized).float()


def distribution_metrics(reference: Tensor, candidate: Tensor) -> tuple[float, float]:
    ref_logp = torch.log_softmax(reference, dim=-1)
    cand_logp = torch.log_softmax(candidate, dim=-1)
    probability = ref_logp.exp()
    kl = (probability * (ref_logp - cand_logp)).sum(dim=-1)
    agreement = reference.argmax(dim=-1) == candidate.argmax(dim=-1)
    return float(kl.mean().item()), float(agreement.float().mean().item())


def jaccard(a: Tensor, b: Tensor) -> float:
    sa = set(a.tolist())
    sb = set(b.tolist())
    return len(sa & sb) / max(len(sa | sb), 1)


def build_prefetch_set(
    rows: list[dict[str, Any]],
    *,
    width: int,
    budget_fraction: float,
) -> Tensor:
    budget = max(1, min(width, round(width * budget_fraction)))
    counts = torch.zeros(width, dtype=torch.float32)
    for row in rows:
        counts[row["selected"]] += 1.0
    return torch.topk(counts, k=budget, sorted=False).indices


def recall_of_prefetch(selected: Tensor, prefetched: Tensor) -> float:
    target = set(selected.tolist())
    resident = set(prefetched.tolist())
    return len(target & resident) / max(len(target), 1)


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def pairwise_overlap_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for left, right in itertools.combinations(rows, 2):
        same_semantic = left["semantic_id"] == right["semantic_id"]
        same_structure = left["structure"] == right["structure"]
        if same_semantic and not same_structure:
            key = "same_semantic_different_structure"
        elif same_structure and not same_semantic:
            key = "same_structure_different_semantic"
        elif not same_semantic and not same_structure:
            key = "different_semantic_different_structure"
        else:
            continue
        buckets[key].append(jaccard(left["selected"], right["selected"]))

    return {
        key: mean(values)
        for key, values in buckets.items()
    }


def structure_prefetch_metrics(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    *,
    width: int,
    budget_fraction: float,
) -> dict[str, float]:
    global_prefetch = build_prefetch_set(
        train_rows,
        width=width,
        budget_fraction=budget_fraction,
    )
    by_structure = {
        structure: build_prefetch_set(
            [row for row in train_rows if row["structure"] == structure],
            width=width,
            budget_fraction=budget_fraction,
        )
        for structure in STRUCTURES
    }

    structure_recall = []
    global_recall = []
    for row in test_rows:
        structure_recall.append(
            recall_of_prefetch(
                row["selected"],
                by_structure[row["structure"]],
            )
        )
        global_recall.append(
            recall_of_prefetch(row["selected"], global_prefetch)
        )

    active_fraction = len(test_rows[0]["selected"]) / width
    struct_mean = mean(structure_recall)
    global_mean = mean(global_recall)
    return {
        "budget_fraction": budget_fraction,
        "structure_conditioned_recall": struct_mean,
        "global_prior_recall": global_mean,
        "structure_recall_lift_over_global": struct_mean - global_mean,
        "structure_residual_fault_fraction_of_full_width": (
            active_fraction * (1.0 - struct_mean)
        ),
        "global_residual_fault_fraction_of_full_width": (
            active_fraction * (1.0 - global_mean)
        ),
        "random_expected_recall": budget_fraction,
    }


def permutation_p_value(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    *,
    width: int,
    budget_fraction: float,
    permutations: int,
    seed: int,
) -> tuple[float, float]:
    observed = structure_prefetch_metrics(
        train_rows,
        test_rows,
        width=width,
        budget_fraction=budget_fraction,
    )["structure_conditioned_recall"]

    rng = random.Random(seed)
    null_scores = []
    semantic_ids = sorted({row["semantic_id"] for row in train_rows})
    for _ in range(permutations):
        shuffled = [dict(row) for row in train_rows]
        for semantic_id in semantic_ids:
            indices = [
                index
                for index, row in enumerate(shuffled)
                if row["semantic_id"] == semantic_id
            ]
            labels = [shuffled[index]["structure"] for index in indices]
            rng.shuffle(labels)
            for index, label in zip(indices, labels, strict=True):
                shuffled[index]["structure"] = label

        null_scores.append(
            structure_prefetch_metrics(
                shuffled,
                test_rows,
                width=width,
                budget_fraction=budget_fraction,
            )["structure_conditioned_recall"]
        )

    exceed = sum(score >= observed for score in null_scores)
    p_value = (exceed + 1) / (permutations + 1)
    return observed, p_value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--top-logits", type=int, default=128)
    parser.add_argument("--fraction", type=float, default=0.30)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--train-semantics", type=int, default=6)
    parser.add_argument(
        "--prefetch-budgets",
        nargs="+",
        type=float,
        default=[0.10, 0.20, 0.30],
    )
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1103)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument("--json-out")
    args = parser.parse_args()

    if not 1 <= args.train_semantics < len(SEMANTIC_CASES):
        raise ValueError("train-semantics must leave at least one held-out semantic case")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit('Install with: pip install -e ".[hf]"') from exc

    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype)
    if device.type == "cpu" and dtype == torch.float16:
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

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
    width = mlp.down_proj.in_features

    rows: list[dict[str, Any]] = []
    for semantic in SEMANTIC_CASES:
        for structure, template in STRUCTURES.items():
            prompt = template.format(**semantic)
            input_ids = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
            )["input_ids"].to(device)

            reference, base_hidden, dense_hidden, activation = capture_final_layer(
                model,
                final_layer,
                input_ids,
            )
            reference = reference.detach().clone()
            base_hidden = base_hidden.detach().clone()
            dense_hidden = dense_hidden.detach().clone()
            activation = activation.detach().clone()

            top_ids = torch.topk(
                reference,
                k=min(args.top_logits, reference.shape[-1]),
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
            selected = greedy_effect_indices(
                effects,
                fraction=args.fraction,
                block_size=args.block_size,
            )[0]
            candidate_hidden = sparse_hidden(
                base_hidden,
                activation,
                mlp.down_proj.weight,
                selected[None, :],
            )
            candidate = exact_logits_from_hidden(candidate_hidden, norm, lm_head)
            kl, top1 = distribution_metrics(reference, candidate)

            rows.append(
                {
                    "semantic_id": semantic["id"],
                    "structure": structure,
                    "selected": selected.detach().cpu(),
                    "kl_nats": kl,
                    "top1_agreement": top1,
                }
            )

    train_ids = {
        semantic["id"]
        for semantic in SEMANTIC_CASES[: args.train_semantics]
    }
    train_rows = [row for row in rows if row["semantic_id"] in train_ids]
    test_rows = [row for row in rows if row["semantic_id"] not in train_ids]

    overlap = pairwise_overlap_summary(rows)
    same_structure = overlap.get("same_structure_different_semantic", 0.0)
    same_semantic = overlap.get("same_semantic_different_structure", 0.0)
    baseline = overlap.get("different_semantic_different_structure", 0.0)

    prefetch = []
    for budget in args.prefetch_budgets:
        metrics = structure_prefetch_metrics(
            train_rows,
            test_rows,
            width=width,
            budget_fraction=budget,
        )
        _observed, p_value = permutation_p_value(
            train_rows,
            test_rows,
            width=width,
            budget_fraction=budget,
            permutations=args.permutations,
            seed=args.seed + round(budget * 1000),
        )
        metrics["structure_label_permutation_p_value"] = p_value
        prefetch.append(metrics)

    quality_kl = [float(row["kl_nats"]) for row in rows]
    quality_top1 = [float(row["top1_agreement"]) for row in rows]

    payload = {
        "experiment": "hf_prompt_structure_codemand_v0",
        "model": args.model,
        "layer": len(layers) - 1,
        "semantic_cases": len(SEMANTIC_CASES),
        "structures": list(STRUCTURES),
        "train_semantics": args.train_semantics,
        "test_semantics": len(SEMANTIC_CASES) - args.train_semantics,
        "oracle_selected_fraction": args.fraction,
        "top_logit_tangent_dim": args.top_logits,
        "oracle_quality": {
            "mean_kl_nats": mean(quality_kl),
            "top1_agreement": mean(quality_top1),
        },
        "pairwise_working_set_jaccard": overlap,
        "structure_overlap_lift_over_unmatched": same_structure - baseline,
        "semantic_overlap_lift_over_unmatched": same_semantic - baseline,
        "structure_minus_semantic_overlap": same_structure - same_semantic,
        "heldout_structure_prefetch": prefetch,
        "claim_boundary": (
            "Prediction-important neuron sets are oracle selections derived from "
            "dense final-layer activations and a local final-logit Jacobian. "
            "Prompt structure is crossed with semantic topic, and structure priors "
            "are trained only on semantic cases disjoint from evaluation cases. "
            "A positive result shows that prompt form contains information about "
            "future exact working-set demand; it does not yet establish a deployable "
            "parser, selective kernel, or measured HBM reduction."
        ),
    }

    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        Path(args.json_out).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
