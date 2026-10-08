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
    "Why should physical cache faults be measured separately from logical sparsity?",
    "How can an oracle establish a lower bound on unavoidable HBM traffic?",
    "Why is a negative oracle result enough to kill an addressing idea?",
    "Describe the difference between active weights and newly transferred weights.",
    "How does cache capacity change a sparse model's physical byte frontier?",
    "Why should a first-token warmup be separated from steady-state decode traffic?",
    "What would make exact sparse compute useful even when logical sparsity is modest?",
    "Explain how a resident working set can turn repeated exact computation into cache hits.",
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
def dense_state(
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
    return activations[0].detach().float().cpu(), logits


def selected_indices(activation: Tensor, mlp: nn.Module, fraction: float) -> Tensor:
    width = activation.shape[-1]
    k = max(1, min(width, round(width * fraction)))
    down_norm = mlp.down_proj.weight.detach().float().cpu().norm(dim=0)
    score = activation.abs() * down_norm[None, :]
    return torch.topk(score, k=k, dim=-1, sorted=False).indices


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


@torch.inference_mode()
def sparse_logits(
    model: nn.Module,
    mlp: nn.Module,
    input_ids: Tensor,
    selected: Tensor,
) -> Tensor:
    def hook(_module: nn.Module, args: tuple[Tensor, ...], output: Tensor) -> Tensor:
        hidden = args[0][:, -1, :]
        sparse = sparse_exact_output(hidden, mlp, selected)
        replaced = output.clone()
        replaced[:, -1, :] = sparse.to(replaced.dtype)
        return replaced

    handle = mlp.register_forward_hook(hook)
    try:
        return model(input_ids=input_ids).logits[:, -1, :].detach().float().cpu()
    finally:
        handle.remove()


def update_cache(cache: list[int], selected: Tensor, budget: int) -> tuple[list[int], int]:
    requested = selected.tolist()
    resident = set(cache)
    faults = sum(1 for index in requested if index not in resident)

    # Requested pages become most-recently-used. Older resident pages remain
    # until capacity is exhausted.
    ordered: list[int] = []
    seen: set[int] = set()
    for index in requested + cache:
        if index in seen:
            continue
        seen.add(index)
        ordered.append(index)
        if len(ordered) >= budget:
            break
    return ordered, faults


def distribution_metrics(reference: Tensor, candidate: Tensor) -> tuple[Tensor, Tensor]:
    ref_logp = torch.log_softmax(reference, dim=-1)
    cand_logp = torch.log_softmax(candidate, dim=-1)
    probability = ref_logp.exp()
    kl = (probability * (ref_logp - cand_logp)).sum(dim=-1)
    agreement = reference.argmax(dim=-1) == candidate.argmax(dim=-1)
    return kl, agreement


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--text-file")
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--decode-steps", type=int, default=10)
    parser.add_argument("--max-prompts", type=int, default=16)
    parser.add_argument("--layers", nargs="+", type=int, default=[18, 23])
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=[0.10, 0.15, 0.20, 0.25, 0.30, 0.40],
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

    texts = load_texts(args.text_file)[: args.max_prompts]
    rows = []

    for layer_index in args.layers:
        mlp = resolve_mlp(layers[layer_index])
        width = mlp.down_proj.in_features

        for fraction in args.fractions:
            k = max(1, min(width, round(width * fraction)))
            cache_state = {
                multiplier: {"faults": 0, "steady_faults": 0, "requests": 0}
                for multiplier in args.cache_multipliers
            }
            all_kl: list[Tensor] = []
            all_agreement: list[Tensor] = []
            exact_rollouts = 0
            total_states = 0

            for text in texts:
                prefix = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=args.max_length,
                )["input_ids"].to(device)
                caches = {multiplier: [] for multiplier in args.cache_multipliers}
                prompt_exact = True

                for step in range(args.decode_steps):
                    activation, reference = dense_state(model, mlp, prefix)
                    selected = selected_indices(activation, mlp, fraction)
                    candidate = sparse_logits(model, mlp, prefix, selected)
                    kl, agreement = distribution_metrics(reference, candidate)
                    all_kl.append(kl)
                    all_agreement.append(agreement)
                    prompt_exact = prompt_exact and bool(agreement.item())
                    total_states += 1

                    for multiplier in args.cache_multipliers:
                        budget = max(k, min(width, round(k * multiplier)))
                        caches[multiplier], faults = update_cache(
                            caches[multiplier],
                            selected[0],
                            budget,
                        )
                        cache_state[multiplier]["faults"] += faults
                        cache_state[multiplier]["requests"] += k
                        if step > 0:
                            cache_state[multiplier]["steady_faults"] += faults

                    token = reference.argmax(dim=-1).to(device)
                    prefix = torch.cat([prefix, token[:, None]], dim=1)

                exact_rollouts += int(prompt_exact)

            kl = torch.cat(all_kl)
            agreement = torch.cat(all_agreement).float()
            quality = {
                "mean_kl_nats": float(kl.mean().item()),
                "p95_kl_nats": float(torch.quantile(kl, 0.95).item()),
                "max_kl_nats": float(kl.max().item()),
                "top1_agreement": float(agreement.mean().item()),
                "exact_rollout_fraction": exact_rollouts / len(texts),
            }

            for multiplier in args.cache_multipliers:
                stats = cache_state[multiplier]
                amortized_fault_fraction = stats["faults"] / (total_states * width)
                steady_states = len(texts) * max(args.decode_steps - 1, 1)
                steady_fault_fraction = stats["steady_faults"] / (steady_states * width)
                rows.append(
                    {
                        "layer": layer_index,
                        "selected_payload_fraction": fraction,
                        "cache_multiplier": multiplier,
                        "cache_fraction_of_mlp_width": min(1.0, fraction * multiplier),
                        "amortized_physical_fault_fraction": amortized_fault_fraction,
                        "steady_state_physical_fault_fraction": steady_fault_fraction,
                        "fault_reduction_vs_stateless_sparse": (
                            1.0 - steady_fault_fraction / fraction
                        ),
                        "cache_request_hit_rate": (
                            1.0 - stats["faults"] / max(stats["requests"], 1)
                        ),
                        **quality,
                    }
                )

    payload = {
        "experiment": "hf_oracle_physical_cache_frontier_v0",
        "model": args.model,
        "prompts": len(texts),
        "decode_steps": args.decode_steps,
        "rows": rows,
        "claim_boundary": (
            "Current active neurons are selected with exact dense activations, so "
            "this is an oracle lower bound rather than a deployable address path. "
            "The sparse candidate evaluates exact selected weights and quality is "
            "measured along held-out dense greedy trajectories. Cache faults count "
            "only selected pages absent from an LRU-like resident working set. "
            "Reported fractions remain analytic until a selective kernel and "
            "hardware profiler measure real HBM/cache traffic."
        ),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        Path(args.json_out).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
