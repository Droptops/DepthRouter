# ruff: noqa: I001
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from hf_block_structured_causal_selector import (
    LowBitBlockScorer,
    blocks_to_neurons,
    capture_dense_state,
    collect_evaluation_states,
    load_texts_from_url,
    lowbit_metadata_bytes,
    lowbit_metadata_fraction,
    parse_dtype,
    passes_strict_gate,
    resolve_device,
    resolve_layers,
    run_sparse_logits,
)
from hf_global_lowbit_decode import LowBitAddress, load_texts


@dataclass
class CandidateOutcomes:
    kl: Tensor
    error: Tensor
    logit_margin: Tensor
    probability_margin: Tensor
    block_gap: Tensor


def wilson_interval(
    successes: int,
    total: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if not 0 <= successes <= total or total <= 0:
        raise ValueError("successes must be in [0, total] and total must be positive")
    probability = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (probability + z2 / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        (probability * (1.0 - probability) + z2 / (4.0 * total)) / total
    ) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def text_set_sha256(texts: list[str]) -> str:
    payload = "\n\x1e\n".join(texts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def candidate_confidence(
    candidate: Tensor,
    block_score: Tensor,
    *,
    k: int,
    mode: str,
) -> Tensor:
    if mode == "block_gap":
        ranked = torch.topk(
            block_score,
            k=min(k + 1, block_score.shape[-1]),
            dim=-1,
            sorted=True,
        ).values
        if ranked.shape[-1] <= k:
            return torch.full(
                (block_score.shape[0],),
                torch.inf,
                device=block_score.device,
            )
        return ranked[:, k - 1] - ranked[:, k]

    top2 = torch.topk(candidate, k=2, dim=-1, sorted=True).values
    if mode == "logit_margin":
        return top2[:, 0] - top2[:, 1]
    if mode == "probability_margin":
        probability = torch.exp(
            top2 - torch.logsumexp(candidate, dim=-1, keepdim=True)
        )
        return probability[:, 0] - probability[:, 1]
    raise ValueError(f"unsupported confidence mode: {mode}")


def fallback_metrics(
    outcomes: CandidateOutcomes,
    confidence: Tensor,
    *,
    threshold: float,
    sparse_execution_fraction: float,
    metadata_fraction: float,
) -> dict[str, float | bool | int]:
    accepted = confidence >= threshold
    count = int(accepted.sum().item())
    total = int(confidence.numel())
    acceptance = count / max(total, 1)
    execution = 1.0 - acceptance * (1.0 - sparse_execution_fraction)
    effective_kl = torch.where(accepted, outcomes.kl, torch.zeros_like(outcomes.kl))
    mean_kl = float(effective_kl.mean().item())
    errors = int(outcomes.error[accepted].sum().item())
    successes = total - errors
    top1 = successes / max(total, 1)
    top1_lower, top1_upper = wilson_interval(successes, total)
    return {
        "states": total,
        "sparse_acceptance_fraction": acceptance,
        "dense_fallback_fraction": 1.0 - acceptance,
        "average_execution_fraction": execution,
        "resident_selector_fraction": metadata_fraction,
        "metadata_plus_average_execution_fraction": execution
        + metadata_fraction,
        "mean_kl_nats": mean_kl,
        "p95_kl_nats": float(torch.quantile(effective_kl, 0.95).item()),
        "max_kl_nats": float(effective_kl.max().item()),
        "top1_agreement": top1,
        "top1_wilson_95_lower": top1_lower,
        "top1_wilson_95_upper": top1_upper,
        "sparse_errors": errors,
        "passes_strict_gate": passes_strict_gate(
            top1_agreement=top1,
            mean_kl_nats=mean_kl,
            execution_fraction=execution,
            metadata_fraction=metadata_fraction,
        ),
    }


def calibrate_threshold(
    outcomes: CandidateOutcomes,
    confidence: Tensor,
    *,
    sparse_execution_fraction: float,
    metadata_fraction: float,
    minimum_calibration_top1: float = 0.99,
    maximum_calibration_kl: float = 0.01,
) -> tuple[float, dict[str, float | bool | int]] | None:
    best: tuple[float, dict[str, float | bool | int]] | None = None
    for threshold_tensor in torch.unique(confidence).sort(descending=True).values:
        threshold = float(threshold_tensor.item())
        metrics = fallback_metrics(
            outcomes,
            confidence,
            threshold=threshold,
            sparse_execution_fraction=sparse_execution_fraction,
            metadata_fraction=metadata_fraction,
        )
        if (
            not metrics["passes_strict_gate"]
            or metrics["top1_agreement"] < minimum_calibration_top1
            or metrics["mean_kl_nats"] >= maximum_calibration_kl
        ):
            continue
        if best is None or metrics["sparse_acceptance_fraction"] > best[1][
            "sparse_acceptance_fraction"
        ]:
            best = threshold, metrics
    return best


@torch.inference_mode()
def evaluate_candidates(
    model,
    mlp,
    states,
    scorer: LowBitBlockScorer,
    *,
    k: int,
    block_size: int,
    width: int,
    device: torch.device,
) -> CandidateOutcomes:
    kl_rows: list[Tensor] = []
    error_rows: list[Tensor] = []
    logit_margin_rows: list[Tensor] = []
    probability_margin_rows: list[Tensor] = []
    block_gap_rows: list[Tensor] = []

    for state in states:
        block_score = scorer(state.current)
        ranked = torch.topk(
            block_score,
            k=min(k + 1, block_score.shape[-1]),
            dim=-1,
            sorted=True,
        )
        selected = blocks_to_neurons(
            ranked.indices[0, :k],
            block_size=block_size,
            width=width,
        )
        candidate = run_sparse_logits(
            model,
            mlp,
            state.input_ids.to(device),
            selected,
        )
        reference_logp = torch.log_softmax(state.reference_logits, dim=-1)
        candidate_logp = torch.log_softmax(candidate, dim=-1)
        probability = reference_logp.exp()
        kl_rows.append(
            (probability * (reference_logp - candidate_logp)).sum(dim=-1)
        )
        error_rows.append(
            state.reference_logits.argmax(dim=-1) != candidate.argmax(dim=-1)
        )
        top2 = torch.topk(candidate, k=2, dim=-1, sorted=True).values
        logit_margin_rows.append(top2[:, 0] - top2[:, 1])
        top2_probability = torch.exp(
            top2 - torch.logsumexp(candidate, dim=-1, keepdim=True)
        )
        probability_margin_rows.append(
            top2_probability[:, 0] - top2_probability[:, 1]
        )
        if ranked.values.shape[-1] > k:
            block_gap_rows.append(ranked.values[:, k - 1] - ranked.values[:, k])
        else:
            block_gap_rows.append(torch.full((1,), torch.inf))

    return CandidateOutcomes(
        kl=torch.cat(kl_rows).cpu(),
        error=torch.cat(error_rows).cpu(),
        logit_margin=torch.cat(logit_margin_rows).cpu(),
        probability_margin=torch.cat(probability_margin_rows).cpu(),
        block_gap=torch.cat(block_gap_rows).cpu(),
    )


@torch.inference_mode()
def evaluate_free_running(
    model,
    layer,
    tokenizer,
    texts: list[str],
    dense_tokens: list[list[int]],
    scorer: LowBitBlockScorer,
    *,
    k: int,
    block_size: int,
    width: int,
    confidence_mode: str,
    threshold: float,
    sparse_execution_fraction: float,
    metadata_fraction: float,
    max_length: int,
    device: torch.device,
) -> dict[str, float | bool | int]:
    exact_prompts = 0
    matching_tokens = 0
    total_tokens = 0
    sparse_tokens = 0
    mlp = layer.mlp

    for text, reference_tokens in zip(texts, dense_tokens, strict=True):
        prefix = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )["input_ids"].to(device)
        prompt_exact = True
        for reference_token in reference_tokens:
            reference, hidden, _base, _dense, _active = capture_dense_state(
                model,
                layer,
                prefix,
            )
            block_score = scorer(hidden)
            chosen = torch.topk(
                block_score,
                k=k,
                dim=-1,
                sorted=False,
            ).indices[0]
            selected = blocks_to_neurons(
                chosen,
                block_size=block_size,
                width=width,
            )
            candidate = run_sparse_logits(model, mlp, prefix, selected)
            confidence = candidate_confidence(
                candidate,
                block_score,
                k=k,
                mode=confidence_mode,
            )
            use_sparse = float(confidence.item()) >= threshold
            logits = candidate if use_sparse else reference
            token = int(logits.argmax(dim=-1).item())
            sparse_tokens += int(use_sparse)
            matching_tokens += int(token == reference_token)
            total_tokens += 1
            prompt_exact = prompt_exact and token == reference_token
            prefix = torch.cat(
                [prefix, torch.tensor([[token]], device=device)],
                dim=1,
            )
        exact_prompts += int(prompt_exact)

    acceptance = sparse_tokens / max(total_tokens, 1)
    execution = 1.0 - acceptance * (1.0 - sparse_execution_fraction)
    return {
        "prompts": len(texts),
        "tokens": total_tokens,
        "exact_prompt_fraction": exact_prompts / max(len(texts), 1),
        "token_agreement": matching_tokens / max(total_tokens, 1),
        "sparse_acceptance_fraction": acceptance,
        "average_execution_fraction": execution,
        "metadata_plus_average_execution_fraction": execution
        + metadata_fraction,
        "within_execution_gate": execution <= 0.90
        and execution + metadata_fraction < 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--revision")
    parser.add_argument("--text-file")
    parser.add_argument("--text-url")
    parser.add_argument("--deduplicate-texts", action="store_true")
    parser.add_argument("--split-seed", type=int, default=20261012)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--decode-steps", type=int, default=3)
    parser.add_argument("--max-calibration-prompts", type=int, default=200)
    parser.add_argument("--max-test-prompts", type=int, default=200)
    parser.add_argument("--block-sizes", nargs="+", type=int, default=[16])
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=[0.75, 0.80, 0.85],
    )
    parser.add_argument("--address-bits", type=int, choices=[1, 2], default=2)
    parser.add_argument("--minimum-calibration-top1", type=float, default=0.998)
    parser.add_argument("--maximum-calibration-kl", type=float, default=0.0075)
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument("--json-out")
    parser.add_argument("--free-running", action="store_true")
    args = parser.parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit('Install with: pip install -e ".[hf]"') from exc

    if args.text_file and args.text_url:
        raise ValueError("choose only one of --text-file and --text-url")
    if any(not 0.0 < fraction <= 0.90 for fraction in args.fractions):
        raise ValueError("every sparse execution fraction must be in (0, 0.90]")

    texts = (
        load_texts_from_url(args.text_url)
        if args.text_url
        else load_texts(args.text_file)
    )
    if args.deduplicate_texts:
        texts = list(dict.fromkeys(texts))
    source_prompt_sha256 = text_set_sha256(texts)
    generator = torch.Generator().manual_seed(args.split_seed)
    order = torch.randperm(len(texts), generator=generator).tolist()
    texts = [texts[index] for index in order]
    requested = args.max_calibration_prompts + args.max_test_prompts
    if requested > len(texts):
        raise ValueError("calibration/test prompts exceed available disjoint texts")
    calibration_texts = texts[: args.max_calibration_prompts]
    test_texts = texts[-args.max_test_prompts :]
    if set(calibration_texts) & set(test_texts):
        raise ValueError("calibration and test prompts must be disjoint")

    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype)
    if device.type == "cpu" and dtype == torch.float16:
        dtype = torch.float32
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=dtype,
    ).to(device)
    model.eval().requires_grad_(False)
    layers = resolve_layers(model)
    layer = layers[-1]
    mlp = layer.mlp
    width = mlp.down_proj.in_features
    address = LowBitAddress(mlp, args.address_bits)
    metadata = lowbit_metadata_fraction(mlp, bits=args.address_bits)
    address_bytes, dense_mlp_bytes = lowbit_metadata_bytes(
        mlp,
        bits=args.address_bits,
    )

    calibration_states, _calibration_tokens = collect_evaluation_states(
        model,
        layer,
        tokenizer,
        calibration_texts,
        decode_steps=args.decode_steps,
        max_length=args.max_length,
        device=device,
    )
    test_states, test_tokens = collect_evaluation_states(
        model,
        layer,
        tokenizer,
        test_texts,
        decode_steps=args.decode_steps,
        max_length=args.max_length,
        device=device,
    )

    rows = []
    for block_size in args.block_sizes:
        blocks = math.ceil(width / block_size)
        scorer = LowBitBlockScorer(
            address,
            mlp,
            block_size=block_size,
            width=width,
            device=device,
        )
        for requested_fraction in args.fractions:
            k = max(1, min(blocks, math.floor(width * requested_fraction / block_size)))
            sparse_fraction = min(1.0, (k * block_size) / width)
            calibration = evaluate_candidates(
                model,
                mlp,
                calibration_states,
                scorer,
                k=k,
                block_size=block_size,
                width=width,
                device=device,
            )
            test = evaluate_candidates(
                model,
                mlp,
                test_states,
                scorer,
                k=k,
                block_size=block_size,
                width=width,
                device=device,
            )
            sparse_only_test = fallback_metrics(
                test,
                test.logit_margin,
                threshold=-math.inf,
                sparse_execution_fraction=sparse_fraction,
                metadata_fraction=metadata,
            )
            for mode in ("logit_margin", "probability_margin", "block_gap"):
                calibrated = calibrate_threshold(
                    calibration,
                    getattr(calibration, mode),
                    sparse_execution_fraction=sparse_fraction,
                    metadata_fraction=metadata,
                    minimum_calibration_top1=args.minimum_calibration_top1,
                    maximum_calibration_kl=args.maximum_calibration_kl,
                )
                if calibrated is None:
                    rows.append(
                        {
                            "block_size": block_size,
                            "selected_blocks": k,
                            "requested_sparse_fraction": requested_fraction,
                            "sparse_execution_fraction": sparse_fraction,
                            "confidence_mode": mode,
                            "calibrated": False,
                            "calibration": None,
                            "test": None,
                            "sparse_only_test": sparse_only_test,
                        }
                    )
                    continue
                threshold, calibration_metrics = calibrated
                test_metrics = fallback_metrics(
                    test,
                    getattr(test, mode),
                    threshold=threshold,
                    sparse_execution_fraction=sparse_fraction,
                    metadata_fraction=metadata,
                )
                free_running = None
                if args.free_running and test_metrics["passes_strict_gate"]:
                    free_running = evaluate_free_running(
                        model,
                        layer,
                        tokenizer,
                        test_texts,
                        test_tokens,
                        scorer,
                        k=k,
                        block_size=block_size,
                        width=width,
                        confidence_mode=mode,
                        threshold=threshold,
                        sparse_execution_fraction=sparse_fraction,
                        metadata_fraction=metadata,
                        max_length=args.max_length,
                        device=device,
                    )
                rows.append(
                    {
                        "block_size": block_size,
                        "selected_blocks": k,
                        "requested_sparse_fraction": requested_fraction,
                        "sparse_execution_fraction": sparse_fraction,
                        "confidence_mode": mode,
                        "calibrated": True,
                        "threshold": threshold,
                        "calibration": calibration_metrics,
                        "test": test_metrics,
                        "sparse_only_test": sparse_only_test,
                        "free_running": free_running,
                    }
                )

    payload = {
        "experiment": "hf_block_lowbit_fallback_v0",
        "model": args.model,
        "model_revision": getattr(model.config, "_commit_hash", None)
        or args.revision,
        "tokenizer_revision": tokenizer.init_kwargs.get("_commit_hash")
        or args.revision,
        "text_source": args.text_url or args.text_file or "built-in defaults",
        "deduplicated_texts": args.deduplicate_texts,
        "source_prompt_count_after_deduplication": len(texts),
        "source_prompt_sha256": source_prompt_sha256,
        "calibration_prompt_sha256": text_set_sha256(calibration_texts),
        "test_prompt_sha256": text_set_sha256(test_texts),
        "split_seed": args.split_seed,
        "seed": args.seed,
        "precision": str(dtype),
        "device": str(device),
        "layer": len(layers) - 1,
        "address_bits": args.address_bits,
        "packed_address_bytes": address_bytes,
        "dense_final_layer_mlp_fp16_bytes": dense_mlp_bytes,
        "resident_selector_fraction": metadata,
        "calibration_prompts": len(calibration_texts),
        "calibration_states": len(calibration_states),
        "test_prompts": len(test_texts),
        "test_states": len(test_states),
        "decode_steps": args.decode_steps,
        "quality_gate": {
            "minimum_top1_agreement": 0.99,
            "maximum_mean_kl_nats_exclusive": 0.01,
            "maximum_average_execution_fraction": 0.90,
            "metadata_plus_average_execution_must_be_below_dense": True,
        },
        "calibration_guardrail": {
            "minimum_top1_agreement": args.minimum_calibration_top1,
            "maximum_mean_kl_nats_exclusive": args.maximum_calibration_kl,
        },
        "physical_bytes_per_token": None,
        "latency_ms_per_token": None,
        "rows": rows,
        "claim_boundary": (
            "The block address uses only a packed low-bit approximation of the "
            "current final-layer gate/up projections. Confidence is computed from "
            "the sparse candidate logits before deciding whether to execute the "
            "omitted blocks as a dense fallback. Execution is the state-average "
            "exact-neuron fraction including fallbacks; address metadata is charged "
            "separately against FP16 final-layer MLP bytes. The diagnostic reconstructs "
            "the packed address into FP32 tensors and the fallback requires a candidate "
            "vocabulary projection before any omitted blocks are completed; neither "
            "operation has a fused kernel or latency accounting. PyTorch hooks still "
            "execute dense instrumentation, so physical bytes and latency are null."
        ),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        temporary.replace(output)


if __name__ == "__main__":
    main()
