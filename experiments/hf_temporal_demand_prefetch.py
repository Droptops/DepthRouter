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
    "How can the previous token's active set predict the next token's memory demand?",
    "What does semantic branch prediction mean for neural inference?",
    "Why can temporal locality matter more than static sparsity?",
    "Explain how a cache can turn repeated logical demand into few physical transfers.",
    "What is working-set churn?",
    "Why might autoregressive decode states move smoothly through computational space?",
    "How should a neural prefetcher be evaluated?",
    "What does it mean to cache a probability trajectory physically?",
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
    if len(rows) < 4:
        raise ValueError("need at least four texts")
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
def capture_demand(
    model: nn.Module,
    mlp: nn.Module,
    input_ids: Tensor,
) -> tuple[Tensor, Tensor]:
    activations: list[Tensor] = []

    def down_pre(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
        activations.append(args[0][:, -1, :].detach())

    handle = mlp.down_proj.register_forward_pre_hook(down_pre)
    try:
        logits = model(input_ids=input_ids).logits[:, -1, :].detach().float().cpu()
    finally:
        handle.remove()

    activation = activations[0].float().cpu()
    down_norm = mlp.down_proj.weight.detach().float().cpu().norm(dim=0)
    score = activation.abs() * down_norm[None, :]
    score = score / score.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return score, logits


def top_indices(score: Tensor, fraction: float) -> Tensor:
    width = score.shape[-1]
    k = max(1, min(width, round(width * fraction)))
    return torch.topk(score, k=k, dim=-1, sorted=False).indices


def set_recall(target: Tensor, cached: Tensor) -> float:
    target_set = set(target.tolist())
    cached_set = set(cached.tolist())
    return len(target_set & cached_set) / max(len(target_set), 1)


def union_recent(history: list[Tensor], budget: int) -> Tensor:
    """LRU-like union ordered newest-first, capped to neuron budget."""

    seen: set[int] = set()
    result: list[int] = []
    for selected in reversed(history):
        for index in selected.tolist():
            if index in seen:
                continue
            seen.add(index)
            result.append(index)
            if len(result) >= budget:
                return torch.tensor(result, dtype=torch.long)
    return torch.tensor(result, dtype=torch.long)


def summarize(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float32)
    return {
        "mean": float(tensor.mean().item()),
        "p05": float(torch.quantile(tensor, 0.05).item()),
        "p50": float(torch.quantile(tensor, 0.50).item()),
        "p95": float(torch.quantile(tensor, 0.95).item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--text-file")
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--decode-steps", type=int, default=12)
    parser.add_argument("--layers", nargs="+", type=int, default=[18, 23])
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=[0.05, 0.10, 0.15, 0.20],
    )
    parser.add_argument(
        "--cache-multipliers",
        nargs="+",
        type=float,
        default=[1.0, 1.5, 2.0, 3.0],
    )
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

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    layers = resolve_layers(model)
    if any(index < 0 or index >= len(layers) for index in args.layers):
        raise ValueError("layer index out of range")

    texts = load_texts(args.text_file)
    rows = []

    for layer_index in args.layers:
        mlp = resolve_mlp(layers[layer_index])
        width = mlp.down_proj.in_features
        score_sequences: list[list[Tensor]] = []

        for text in texts:
            prefix = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
            )["input_ids"].to(device)
            sequence: list[Tensor] = []

            for _ in range(args.decode_steps):
                score, logits = capture_demand(model, mlp, prefix)
                sequence.append(score[0])
                token = logits.argmax(dim=-1).to(device)
                prefix = torch.cat([prefix, token[:, None]], dim=1)

            score_sequences.append(sequence)

        for fraction in args.fractions:
            k = max(1, min(width, round(width * fraction)))
            one_step_recall: list[float] = []
            one_step_churn: list[float] = []
            score_cosine: list[float] = []

            multiplier_metrics: dict[str, dict[str, list[float]]] = {
                str(multiplier): {"recall": [], "fault_fraction": []}
                for multiplier in args.cache_multipliers
            }

            for sequence in score_sequences:
                selections = [top_indices(score[None, :], fraction)[0] for score in sequence]
                for step in range(1, len(sequence)):
                    previous = selections[step - 1]
                    current = selections[step]
                    recall = set_recall(current, previous)
                    one_step_recall.append(recall)
                    one_step_churn.append(1.0 - recall)
                    score_cosine.append(
                        float(
                            torch.nn.functional.cosine_similarity(
                                sequence[step - 1][None, :],
                                sequence[step][None, :],
                            ).item()
                        )
                    )

                    history = selections[:step]
                    for multiplier in args.cache_multipliers:
                        budget = max(k, round(k * multiplier))
                        cached = union_recent(history, budget)
                        cached_recall = set_recall(current, cached)
                        multiplier_metrics[str(multiplier)]["recall"].append(
                            cached_recall
                        )
                        # Only missing target neurons require a cold fault.
                        missing = k * (1.0 - cached_recall)
                        multiplier_metrics[str(multiplier)]["fault_fraction"].append(
                            missing / width
                        )

            metrics = {
                "layer": layer_index,
                "active_fraction": fraction,
                "active_neurons": k,
                "score_cosine": summarize(score_cosine),
                "previous_step_recall": summarize(one_step_recall),
                "working_set_churn": summarize(one_step_churn),
                "cache_budgets": {},
            }
            for multiplier in args.cache_multipliers:
                key = str(multiplier)
                metrics["cache_budgets"][key] = {
                    "cache_fraction_of_mlp_width": min(
                        1.0,
                        fraction * multiplier,
                    ),
                    "next_active_set_recall": summarize(
                        multiplier_metrics[key]["recall"]
                    ),
                    "cold_fault_fraction_of_full_width": summarize(
                        multiplier_metrics[key]["fault_fraction"]
                    ),
                    "fault_reduction_vs_stateless_sparse": (
                        1.0
                        - (
                            sum(multiplier_metrics[key]["fault_fraction"])
                            / len(multiplier_metrics[key]["fault_fraction"])
                        )
                        / fraction
                    ),
                }
            rows.append(metrics)

    payload = {
        "experiment": "hf_temporal_demand_prefetch_v0",
        "model": args.model,
        "decode_steps": args.decode_steps,
        "prompts": len(texts),
        "rows": rows,
        "claim_boundary": (
            "The demanded neuron set is defined with exact dense SwiGLU "
            "activations, so this is a temporal-locality oracle. The prefetch "
            "policy itself uses only previously observed active sets. Reported "
            "fault fractions are neuron-payload proxies, not measured HBM bytes. "
            "A deployable system still needs a selective kernel and a way to "
            "establish or predict the first active set."
        ),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        Path(args.json_out).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
