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
    if not rows:
        raise ValueError("text corpus is empty")
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
        output,
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
    weight = norm.weight.detach().float().cpu()
    head = lm_head.weight.detach().float().cpu()
    selected_head = head[top_ids]
    a = selected_head * weight[None, None, :]

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
    rows = []
    wt = down_weight.T.detach().float().cpu()
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


def distribution_metrics(reference: Tensor, candidate: Tensor) -> tuple[Tensor, Tensor]:
    ref_logp = torch.log_softmax(reference, dim=-1)
    cand_logp = torch.log_softmax(candidate, dim=-1)
    probability = ref_logp.exp()
    kl = (probability * (ref_logp - cand_logp)).sum(dim=-1)
    agreement = reference.argmax(dim=-1) == candidate.argmax(dim=-1)
    return kl, agreement


def set_overlap(current: Tensor, previous: Tensor) -> float:
    a = set(current.tolist())
    b = set(previous.tolist())
    return len(a & b) / max(len(a), 1)


def update_cache(cache: list[int], selected: Tensor, budget: int) -> tuple[list[int], int]:
    requested = selected.tolist()
    resident = set(cache)
    faults = sum(1 for index in requested if index not in resident)

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--text-file")
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--decode-steps", type=int, default=6)
    parser.add_argument("--max-prompts", type=int, default=8)
    parser.add_argument("--top-logits", type=int, default=128)
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.05, 0.10, 0.15])
    parser.add_argument("--block-size", type=int, default=16)
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
    final_layer = layers[-1]
    mlp = final_layer.mlp
    norm = model.model.norm
    lm_head = model.lm_head
    width = mlp.down_proj.in_features

    texts = load_texts(args.text_file)[: args.max_prompts]
    rows = []

    for fraction in args.fractions:
        all_kl: list[Tensor] = []
        all_agreement: list[Tensor] = []
        one_step_overlap: list[float] = []
        exact_rollouts = 0
        total_states = 0
        cache_stats = {
            multiplier: {"faults": 0, "steady_faults": 0, "requests": 0}
            for multiplier in args.cache_multipliers
        }

        for text in texts:
            prefix = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
            )["input_ids"].to(device)
            previous_selected: Tensor | None = None
            caches = {multiplier: [] for multiplier in args.cache_multipliers}
            prompt_exact = True

            for step in range(args.decode_steps):
                reference, base_hidden, dense_hidden, activation = capture_final_layer(
                    model,
                    final_layer,
                    prefix,
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
                    fraction=fraction,
                    block_size=args.block_size,
                )
                candidate_hidden = sparse_hidden(
                    base_hidden,
                    activation,
                    mlp.down_proj.weight,
                    selected,
                )
                candidate = exact_logits_from_hidden(candidate_hidden, norm, lm_head)
                kl, agreement = distribution_metrics(reference, candidate)
                all_kl.append(kl)
                all_agreement.append(agreement)
                prompt_exact = prompt_exact and bool(agreement.item())
                total_states += 1

                if previous_selected is not None:
                    one_step_overlap.append(
                        set_overlap(selected[0], previous_selected)
                    )

                k = selected.shape[1]
                for multiplier in args.cache_multipliers:
                    budget = max(k, min(width, round(k * multiplier)))
                    caches[multiplier], faults = update_cache(
                        caches[multiplier],
                        selected[0],
                        budget,
                    )
                    cache_stats[multiplier]["faults"] += faults
                    cache_stats[multiplier]["requests"] += k
                    if step > 0:
                        cache_stats[multiplier]["steady_faults"] += faults

                previous_selected = selected[0].detach().clone()
                token = reference.argmax(dim=-1).to(device)
                prefix = torch.cat([prefix, token[:, None]], dim=1)

            exact_rollouts += int(prompt_exact)

        kl = torch.cat(all_kl)
        agreement = torch.cat(all_agreement).float()
        selected_fraction = fraction
        quality = {
            "states_evaluated": int(agreement.numel()),
            "top1_mismatches": int((1.0 - agreement).sum().item()),
            "mean_kl_nats": float(kl.mean().item()),
            "p95_kl_nats": float(torch.quantile(kl, 0.95).item()),
            "max_kl_nats": float(kl.max().item()),
            "top1_agreement": float(agreement.mean().item()),
            "exact_rollout_fraction": exact_rollouts / len(texts),
            "mean_previous_step_overlap": (
                sum(one_step_overlap) / len(one_step_overlap)
                if one_step_overlap
                else 0.0
            ),
        }

        for multiplier in args.cache_multipliers:
            stats = cache_stats[multiplier]
            steady_states = len(texts) * max(args.decode_steps - 1, 1)
            amortized_fault_fraction = stats["faults"] / (total_states * width)
            steady_fault_fraction = stats["steady_faults"] / (steady_states * width)
            rows.append(
                {
                    "top_logit_tangent_dim": args.top_logits,
                    "selected_payload_fraction": selected_fraction,
                    "cache_multiplier": multiplier,
                    "cache_fraction_of_mlp_width": min(
                        1.0,
                        fraction * multiplier,
                    ),
                    "amortized_physical_fault_fraction": amortized_fault_fraction,
                    "steady_state_physical_fault_fraction": steady_fault_fraction,
                    "fault_reduction_vs_stateless_sparse": (
                        1.0 - steady_fault_fraction / selected_fraction
                    ),
                    "cache_request_hit_rate": (
                        1.0 - stats["faults"] / max(stats["requests"], 1)
                    ),
                    **quality,
                }
            )

    payload = {
        "experiment": "hf_temporal_logit_tangent_cache_v0",
        "model": args.model,
        "layer": len(layers) - 1,
        "prompts": len(texts),
        "decode_steps": args.decode_steps,
        "rows": rows,
        "claim_boundary": (
            "Selection is an oracle derived from the local final-logit Jacobian, "
            "so this test is not deployable. It asks one narrow question: whether "
            "probability-optimal exact working sets both preserve dense next-token "
            "behavior and exhibit enough temporal locality that only a small set "
            "of new exact weight rows must move per decode step. Physical fault "
            "fractions are analytic until confirmed with a selective kernel and "
            "hardware profiler."
        ),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        Path(args.json_out).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
