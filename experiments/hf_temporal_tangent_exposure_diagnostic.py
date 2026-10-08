# ruff: noqa: I001
"""Held-out exposure-bias diagnostic for the temporal tangent address head.

The head from hf_temporal_tangent_distilled.py is trained with the previous
*oracle* working set as its history input, but at runtime it only sees its own
previous *prediction*. This script trains the same head and evaluates every
held-out decode state under several history arms:

- student_history: previous predicted mask (the deployable configuration)
- oracle_history: previous oracle mask (teacher forcing, privileged)
- no_history: zero mask and zero hidden delta
- copy_previous: reuse the previous oracle set verbatim (persistence-only
  baseline, privileged; step 0 falls back to no_history)
- oracle_selection: the current oracle set (privileged ceiling)

Dense-generated prefixes are used at every step, so this is not a free-running
sparse rollout, and no hardware traffic is measured.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import Tensor, nn

from hf_temporal_logit_tangent_cache import update_cache
from hf_temporal_tangent_distilled import (
    capture_dense_state,
    collect_training_states,
    distribution_metrics,
    labels_from_indices,
    load_texts,
    metadata_fraction,
    parse_dtype,
    resolve_device,
    resolve_layers,
    set_recall,
    sparse_exact_output,
    teacher_selected,
    train_head,
)


ARMS = ("student_history", "oracle_history", "no_history", "copy_previous", "oracle_selection")


@torch.inference_mode()
def sparse_final_logits(
    base_hidden: Tensor,
    mlp_input: Tensor,
    selected: Tensor,
    mlp: nn.Module,
    norm: nn.Module,
    lm_head: nn.Linear,
) -> Tensor:
    """Final-layer logits with only the selected MLP rows, without a second model forward.

    Only valid for the last decoder layer: base_hidden is that layer's output
    minus its dense MLP contribution, so adding the sparse MLP output and
    applying the final norm and LM head reproduces the model's logits.
    """
    device = mlp.down_proj.weight.device
    dtype = mlp.down_proj.weight.dtype
    sparse = sparse_exact_output(
        mlp_input.to(device=device, dtype=dtype),
        mlp,
        selected[None, :],
    )
    final_hidden = base_hidden.to(device=device, dtype=dtype) + sparse
    return lm_head(norm(final_hidden)).float().cpu()


def empty_samples() -> dict[str, dict[str, list[float]]]:
    return {arm: {"kl": [], "top1": [], "recall": []} for arm in ARMS}


def summarize(samples: dict[str, list[float]]) -> dict[str, float]:
    if not samples["kl"]:
        return {}
    kl = torch.tensor(samples["kl"])
    return {
        "states": len(samples["kl"]),
        "mean_kl_nats": float(kl.mean().item()),
        "p95_kl_nats": float(torch.quantile(kl, 0.95).item()),
        "top1_agreement": sum(samples["top1"]) / len(samples["top1"]),
        "mean_oracle_set_recall": sum(samples["recall"]) / len(samples["recall"]),
    }


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
        default="float32",
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
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.eval()
    layers = resolve_layers(model)
    layer = layers[-1]
    mlp = layer.mlp
    norm = model.model.norm
    lm_head = model.lm_head
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

    all_steps = empty_samples()
    first_step = empty_samples()
    steady_steps = empty_samples()
    exact_prompts = {arm: 0 for arm in ARMS}
    faults = {m: {"amortized": 0, "steady": 0} for m in args.cache_multipliers}
    zero_mask = torch.zeros(1, width)

    for text in test_texts:
        prefix = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
        )["input_ids"].to(device)
        prev_hidden: Tensor | None = None
        prev_student: Tensor | None = None
        prev_oracle: Tensor | None = None
        caches: dict[float, list[int]] = {m: [] for m in args.cache_multipliers}
        prompt_exact = {arm: True for arm in ARMS}

        for step in range(args.test_decode_steps):
            reference, hidden, base_hidden, dense_hidden, activation = capture_dense_state(
                model,
                layer,
                prefix,
            )
            reference = reference.detach().clone()
            hidden = hidden.detach().clone()
            base_hidden = base_hidden.detach().clone()
            dense_hidden = dense_hidden.detach().clone()
            activation = activation.detach().clone()
            target = teacher_selected(
                reference,
                dense_hidden,
                activation,
                mlp,
                norm,
                lm_head,
                fraction=args.fraction,
                top_logits=args.top_logits,
                block_size=args.block_size,
            )[0]

            if prev_hidden is None or prev_student is None or prev_oracle is None:
                previous_hidden = hidden
                student_mask = zero_mask
                oracle_mask = zero_mask
            else:
                previous_hidden = prev_hidden
                student_mask = labels_from_indices(prev_student[None, :], width)
                oracle_mask = labels_from_indices(prev_oracle[None, :], width)

            with torch.no_grad():
                addresses = {
                    "student_history": head(hidden, previous_hidden, student_mask),
                    "oracle_history": head(hidden, previous_hidden, oracle_mask),
                    "no_history": head(hidden, hidden, zero_mask),
                }
            selections = {
                arm: torch.topk(logits, k=k, dim=-1, sorted=False).indices[0]
                for arm, logits in addresses.items()
            }
            selections["copy_previous"] = (
                prev_oracle if prev_oracle is not None else selections["no_history"]
            )
            selections["oracle_selection"] = target

            for arm, selected in selections.items():
                candidate = sparse_final_logits(
                    base_hidden,
                    hidden,
                    selected,
                    mlp,
                    norm,
                    lm_head,
                )
                kl, agreement = distribution_metrics(reference, candidate)
                sample = {
                    "kl": float(kl.item()),
                    "top1": float(agreement.item()),
                    "recall": set_recall(selected, target),
                }
                bucket = first_step if step == 0 else steady_steps
                for key, value in sample.items():
                    all_steps[arm][key].append(value)
                    bucket[arm][key].append(value)
                prompt_exact[arm] = prompt_exact[arm] and bool(agreement.item())

            student = selections["student_history"]
            for multiplier in args.cache_multipliers:
                budget = max(k, min(width, round(k * multiplier)))
                caches[multiplier], missing = update_cache(caches[multiplier], student, budget)
                faults[multiplier]["amortized"] += missing
                if step > 0:
                    faults[multiplier]["steady"] += missing

            prev_hidden = hidden
            prev_student = student
            prev_oracle = target
            token = reference.argmax(dim=-1).to(device)
            prefix = torch.cat([prefix, token[:, None]], dim=1)

        for arm in ARMS:
            exact_prompts[arm] += int(prompt_exact[arm])

    metadata = metadata_fraction(head, mlp)
    states = len(test_texts) * args.test_decode_steps
    steady_states = len(test_texts) * max(args.test_decode_steps - 1, 1)
    cache_rows = []
    for multiplier, count in faults.items():
        steady = count["steady"] / (steady_states * width)
        cache_rows.append(
            {
                "cache_multiplier": multiplier,
                "cache_fraction_of_mlp_width": min(1.0, args.fraction * multiplier),
                "resident_address_fraction": metadata,
                "student_amortized_fault_fraction": count["amortized"] / (states * width),
                "student_steady_fault_fraction": steady,
                "metadata_plus_student_steady_fault_fraction": metadata + steady,
            }
        )

    payload = {
        "experiment": "hf_temporal_tangent_exposure_diagnostic_v0",
        "model": args.model,
        "layer": len(layers) - 1,
        "fraction": args.fraction,
        "top_logit_tangent_dim": args.top_logits,
        "rank": args.rank,
        "train_prompts": len(train_texts),
        "test_prompts": len(test_texts),
        "train_states": int(train_current.shape[0]),
        "test_states": states,
        "address_head_params": sum(p.numel() for p in head.parameters()),
        "address_persistence": float(head.persistence.detach().item()),
        "diagnostic_all_steps": {arm: summarize(all_steps[arm]) for arm in ARMS},
        "diagnostic_first_step": {arm: summarize(first_step[arm]) for arm in ARMS},
        "diagnostic_steady_steps": {arm: summarize(steady_steps[arm]) for arm in ARMS},
        "dense_prefix_exact_prompt_fraction": {
            arm: exact_prompts[arm] / len(test_texts) for arm in ARMS
        },
        "cache_rows": cache_rows,
        "claim_boundary": (
            "oracle_history, copy_previous and oracle_selection use the dense oracle "
            "working set and are privileged diagnostic references, not deployable arms. "
            "Every step decodes from the dense-generated prefix; this is not a "
            "free-running sparse rollout. Fault fractions assume ideal neuron-granular "
            "transfers, and no kernel, PCIe/HBM/L2 profiler, or latency test is included."
        ),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        Path(args.json_out).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
