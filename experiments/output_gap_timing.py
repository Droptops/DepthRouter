"""Optional quality-gated timing and free-running audit for output-gap completion.

The caller must freeze and pass all development and held-out gates before this
module is entered. Each timing block contains every eager/graph arm in a shuffled
order. Cache writes overwrite one fixed decode position; the history never grows
between repetitions. Graph capture and prefill are outside timed regions.
"""

from __future__ import annotations

import contextlib
import csv
import gc
import hashlib
import json
import random
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from depth_router.fused_sparse_evaluation import answer_success
from depth_router.fused_swiglu import CUDAGraphCallable


@contextlib.contextmanager
def _replace_mlp(layer: nn.Module, module: nn.Module) -> Iterator[None]:
    original = layer.mlp
    layer.mlp = module
    try:
        yield
    finally:
        layer.mlp = original


def balanced_generation_subset(
    examples: Sequence[dict[str, Any]], per_split: int = 24
) -> list[dict[str, Any]]:
    """Use the first IDs within every task, never a performance-ranked subset."""
    result = []
    splits = sorted({row["split"] for row in examples if row["split"].startswith("heldout")})
    if not splits:
        raise ValueError("free-running validation requires heldout splits")
    for split in splits:
        tasks = sorted({row["task"] for row in examples if row["split"] == split})
        if per_split < len(tasks) or per_split % len(tasks):
            raise ValueError("generation subset must contain equal positive counts per task")
        for task in tasks:
            available = sorted(
                (row for row in examples if row["split"] == split and row["task"] == task),
                key=lambda row: row["id"],
            )
            chosen = available[: per_split // len(tasks)]
            if len(chosen) != per_split // len(tasks):
                raise ValueError("not enough examples to form a balanced generation subset")
            result.extend(chosen)
    return result


def paired_speedup_ci(
    dense: Sequence[float], candidate: Sequence[float], *, seed: int, resamples: int
) -> dict[str, Any]:
    """Bootstrap paired timing blocks, using the ratio of mean block latencies."""
    reference = np.asarray(dense, dtype=np.float64)
    proposed = np.asarray(candidate, dtype=np.float64)
    if reference.shape != proposed.shape or reference.ndim != 1 or len(reference) < 2:
        raise ValueError("paired timing arrays must have matching lengths of at least two")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(proposed)):
        raise ValueError("timing samples must be finite")
    if np.any(reference <= 0) or np.any(proposed <= 0) or resamples < 100:
        raise ValueError("latencies must be positive and resamples must be at least 100")
    indices = np.random.default_rng(seed).integers(0, len(reference), (resamples, len(reference)))
    ratios = reference[indices].mean(axis=1) / proposed[indices].mean(axis=1)
    low, high = np.quantile(ratios, [0.025, 0.975])
    return {
        "ratio_of_means": float(reference.mean() / proposed.mean()),
        "paired_bootstrap_ci95": [float(low), float(high)],
        "ci_supports_improvement": bool(low > 1.0),
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def _log(log: Any, event: str, **fields: Any) -> None:
    if hasattr(log, "write"):
        log.write(event, **fields)
    elif callable(log):
        log(event, **fields)


def _encode(
    tokenizer: Any, prompts: Sequence[str], device: torch.device, length: int, *, fixed: bool
) -> dict[str, Tensor]:
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length" if fixed else True,
        truncation=True,
        max_length=length,
        add_special_tokens=False,
    )
    return {key: value.to(device) for key, value in encoded.items()}


class _CachedDecode(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        cache: Any,
        attention_mask: Tensor,
        position_ids: Tensor,
        cache_position: Tensor,
    ) -> None:
        super().__init__()
        self.model = model
        self.cache = cache
        self.attention_mask = attention_mask
        self.position_ids = position_ids
        self.cache_position = cache_position

    def forward(self, token: Tensor) -> Tensor:
        return self.model(
            input_ids=token,
            attention_mask=self.attention_mask,
            position_ids=self.position_ids,
            cache_position=self.cache_position,
            past_key_values=self.cache,
            use_cache=True,
            logits_to_keep=1,
        ).logits


@torch.inference_mode()
def _prepare_decode(
    model: nn.Module, tokenizer: Any, prompts: Sequence[str], context: int
) -> tuple[_CachedDecode, Tensor]:
    from transformers import StaticCache

    device = next(model.parameters()).device
    encoded = _encode(tokenizer, prompts, device, context, fixed=True)
    position_ids = encoded["attention_mask"].long().cumsum(-1) - 1
    position_ids.masked_fill_(encoded["attention_mask"] == 0, 0)
    cache = StaticCache(config=model.config, max_cache_len=context + 1)
    output = model(
        **encoded,
        position_ids=position_ids,
        cache_position=torch.arange(context, device=device),
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=1,
    )
    token = output.logits[:, -1].argmax(-1, keepdim=True)
    mask = torch.cat(
        [encoded["attention_mask"], torch.ones_like(encoded["attention_mask"][:, :1])], dim=-1
    )
    runner = _CachedDecode(
        model,
        cache,
        mask,
        encoded["attention_mask"].sum(-1, keepdim=True),
        torch.tensor([context], device=device),
    )
    return runner, token


@torch.inference_mode()
def _measure_blocks(
    runners: dict[str, tuple[Callable[[], Tensor], Callable[[], None]]],
    *,
    scope: str,
    batch: int,
    config: dict[str, Any],
    log: Any,
) -> list[dict[str, Any]]:
    warmup = int(config.get("warmup", 10))
    blocks = int(config.get("blocks", 20))
    repetitions = int(config.get("repetitions_per_block", 20))
    if warmup < 1 or blocks < 2 or repetitions < 1:
        raise ValueError("timing requires positive warmups/repetitions and at least two blocks")
    for run, setup in runners.values():
        setup()
        for _ in range(warmup):
            run()
    torch.cuda.synchronize()
    rng = random.Random(int(config.get("seed", 1732)) + batch)
    rows = []
    for block in range(blocks):
        names = list(runners)
        rng.shuffle(names)
        for order, name in enumerate(names):
            run, setup = runners[name]
            setup()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            finish = torch.cuda.Event(enable_timing=True)
            baseline_memory = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            wall_start = time.perf_counter()
            start.record()
            for _ in range(repetitions):
                run()
            finish.record()
            finish.synchronize()
            wall_ms = (time.perf_counter() - wall_start) * 1000 / repetitions
            rows.append(
                {
                    "scope": scope,
                    "batch_size": batch,
                    "block": block,
                    "order": order,
                    "method": name,
                    "repetitions": repetitions,
                    "wall_ms": wall_ms,
                    "cuda_ms": start.elapsed_time(finish) / repetitions,
                    "peak_incremental_allocated_bytes": max(
                        0, torch.cuda.max_memory_allocated() - baseline_memory
                    ),
                }
            )
        _log(log, "timing_block", scope=scope, batch=batch, block=block)
    return rows


def _summarize(
    rows: list[dict[str, Any]], config: dict[str, Any], graph_errors: list[dict[str, Any]]
) -> dict[str, Any]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["scope"], row["batch_size"]), []).append(row)
    summaries = []
    for (scope, batch), scoped in sorted(groups.items()):
        by_method = {
            method: sorted(
                (row for row in scoped if row["method"] == method), key=lambda row: row["block"]
            )
            for method in sorted({row["method"] for row in scoped})
        }
        dense_methods = [method for method in by_method if method.startswith("dense:")]
        reference = min(
            dense_methods, key=lambda method: np.mean([r["wall_ms"] for r in by_method[method]])
        )
        # Requiring improvement against BOTH dense executions prevents selecting a
        # fortuitously slow reference. The strongest observed dense is also shown.
        for method, samples in by_method.items():
            comparisons = {
                dense: paired_speedup_ci(
                    [r["wall_ms"] for r in by_method[dense]],
                    [r["wall_ms"] for r in samples],
                    seed=int(config.get("seed", 1732)),
                    resamples=int(config.get("bootstrap_resamples", 2000)),
                )
                for dense in dense_methods
            }
            graph_complete = "dense:graph" in by_method and not any(
                failure["scope"] == scope and failure["batch_size"] == batch
                for failure in graph_errors
            )
            summaries.append(
                {
                    "scope": scope,
                    "batch_size": batch,
                    "method": method,
                    "blocks": len(samples),
                    "mean_wall_ms": float(np.mean([r["wall_ms"] for r in samples])),
                    "median_wall_ms": float(np.median([r["wall_ms"] for r in samples])),
                    "mean_cuda_ms": float(np.mean([r["cuda_ms"] for r in samples])),
                    "strongest_dense_method": reference,
                    "versus_strongest_dense": comparisons[reference],
                    "versus_each_dense": comparisons,
                    "all_matching_graphs_available": graph_complete,
                    "timing_ci_supports_savings": bool(
                        not method.startswith("dense:")
                        and graph_complete
                        and all(item["ci_supports_improvement"] for item in comparisons.values())
                    ),
                }
            )
    return {"rows": summaries, "graph_errors": graph_errors}


@torch.inference_mode()
def _generation(
    model: nn.Module,
    tokenizer: Any,
    layer: nn.Module,
    arms: dict[str, nn.Module],
    subset: list[dict[str, Any]],
    config: dict[str, Any],
    log: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    device = next(model.parameters()).device
    rows = []
    max_new = int(config.get("generation_max_new_tokens", 16))
    if not 1 <= max_new <= 16:
        raise ValueError("the bounded generation audit permits at most 16 new tokens")
    for example in subset:
        encoded = _encode(
            tokenizer, [example["prompt"]], device,
            int(config.get("generation_max_prompt_tokens", 128)), fixed=False,
        )
        length = encoded["input_ids"].shape[1]
        completions = {}
        for name, module in arms.items():
            with _replace_mlp(layer, module):
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=max_new,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                )
            ids = generated[0, length:].tolist()
            text = tokenizer.decode(ids, skip_special_tokens=True).strip()
            completions[name] = (ids, text, answer_success(text, example))
        dense_ids, dense_text, dense_success = completions["dense"]
        for name, (ids, text, success) in completions.items():
            rows.append(
                {
                    "id": example["id"], "split": example["split"], "task": example["task"],
                    "arm": name, "dense_text": dense_text, "candidate_text": text,
                    "dense_token_ids": json.dumps(dense_ids), "token_ids": json.dumps(ids),
                    "exact_token_match": ids == dense_ids, "exact_text_match": text == dense_text,
                    "dense_task_success": dense_success, "task_success": success,
                }
            )
        _log(log, "generation_example", id=example["id"])
    summary = {}
    for name in arms:
        summary[name] = {}
        for split in sorted({row["split"] for row in subset}):
            scoped = [row for row in rows if row["arm"] == name and row["split"] == split]
            delta = float(np.mean([r["task_success"] - r["dense_task_success"] for r in scoped]))
            summary[name][split] = {
                "examples": len(scoped),
                "exact_token_agreement": float(np.mean([r["exact_token_match"] for r in scoped])),
                "task_success": float(np.mean([r["task_success"] for r in scoped])),
                "dense_task_success": float(np.mean([r["dense_task_success"] for r in scoped])),
                "task_success_delta": delta,
                "no_unacceptable_task_degradation": delta >= -0.01,
            }
    return summary, rows


@torch.inference_mode()
def run_timing(
    model: nn.Module,
    tokenizer: Any,
    layer: nn.Module,
    arms: dict[str, nn.Module],
    examples: list[dict[str, Any]],
    config: dict[str, Any],
    output_dir: Path,
    log: Any,
) -> dict[str, Any]:
    """Measure supplied, already-frozen arms only after the caller passes quality.

    ``config['timing']['quality_gates_passed']`` must explicitly be true. The
    caller retains responsibility for enforcing the separate objective gate.
    ``arms['dense']`` is the original BF16 full-width MLP; every other module
    includes its complete sparse computation, correction, and internal casts.
    """
    options = dict(config.get("timing", {}))
    if options.get("quality_gates_passed") is not True:
        raise ValueError("timing is forbidden until development and heldout quality gates pass")
    if "dense" not in arms or len(arms) < 2:
        raise ValueError("timing requires the original dense arm and at least one sparse arm")
    if not torch.cuda.is_available() or next(model.parameters()).device.type != "cuda":
        raise RuntimeError("measured GPU timing requires the CUDA model")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    subset = balanced_generation_subset(examples, int(options.get("generation_examples_per_split", 24)))
    encoded_subset = json.dumps(subset, sort_keys=True, separators=(",", ":")).encode()
    _write_json(output_dir / "generation_subset.json", subset)
    _write_json(output_dir / "timing_protocol.json", {
        "options": options, "arms": list(arms),
        "generation_subset_sha256": hashlib.sha256(encoded_subset).hexdigest(),
        "scope": "one decode token after fixed 64-token padded context; final MLP separately",
        "cache_policy": "StaticCache capacity context+1; overwrite the same position each call",
        "dense_reference": "original full-width BF16 MLP, eager and CUDA graph",
        "latency": "wall time includes calls and final synchronization, amortized per block",
        "physical_dram_traffic": "unknown; allocator bytes are not DRAM traffic",
    })
    original = layer.mlp
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    batches = [int(batch) for batch in options.get("batches", [1, 8, 32])]
    context = int(options.get("context_tokens", 64))
    if batches != [1, 8, 32] or context != 64:
        raise ValueError("preregistered timing shapes are batches 1,8,32 and context 64")
    try:
        for batch in batches:
            runners: dict[str, tuple[Callable[[], Tensor], Callable[[], None]]] = {}
            retained_objects = []
            captured_inputs = []
            prompts = [subset[index % len(subset)]["prompt"] for index in range(batch)]
            try:
                layer.mlp = arms["dense"]
                dense_decode, token = _prepare_decode(model, tokenizer, prompts, context)
                hook = arms["dense"].register_forward_pre_hook(
                    lambda _module, inputs, saved=captured_inputs: saved.append(
                        inputs[0].detach().clone()
                    )
                )
                try:
                    dense_decode(token)
                finally:
                    hook.remove()
                mlp_input = captured_inputs[-1]
                for scope in ("complete_final_mlp", "complete_model_cached_decode"):
                    runners.clear()
                    retained_objects.clear()
                    for name, module in arms.items():
                        layer.mlp = module
                        if scope == "complete_final_mlp":
                            operation = module
                            argument = mlp_input
                        else:
                            operation, _ = _prepare_decode(model, tokenizer, prompts, context)
                            # Every arm decodes the SAME token selected by dense.
                            argument = token
                        expected = operation(argument).clone()
                        # Verify overwrite behavior before capturing; a changing
                        # result here would invalidate repeated cached timings.
                        repeated = operation(argument)
                        if not torch.equal(expected, repeated):
                            raise RuntimeError(f"non-repeatable fixed-cache output for {name}")
                        def setup(module=module):
                            layer.mlp = module
                        runners[f"{name}:eager"] = (
                            lambda op=operation, x=argument: op(x), setup
                        )
                        retained_objects.append(operation)
                        try:
                            graph = CUDAGraphCallable(operation, argument)
                            observed = graph().clone()
                            torch.cuda.synchronize()
                            if not torch.equal(expected, observed):
                                raise RuntimeError("CUDA graph/eager outputs differ")
                            runners[f"{name}:graph"] = (graph, lambda: None)
                            retained_objects.append(graph)
                        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
                            failures.append({
                                "scope": scope, "batch_size": batch, "arm": name,
                                "error": f"{type(exc).__name__}: {exc}",
                            })
                    rows.extend(_measure_blocks(
                        runners, scope=scope, batch=batch, config=options, log=log
                    ))
                    _write_csv(output_dir / "timing_blocks.csv", rows)
            except torch.cuda.OutOfMemoryError as exc:
                failures.append({
                    "scope": "all", "batch_size": batch, "arm": "all",
                    "error": f"OutOfMemoryError: {exc}",
                })
                _log(log, "timing_batch_oom", batch=batch, error=str(exc))
            finally:
                runners.clear()
                retained_objects.clear()
                captured_inputs.clear()
                gc.collect()
                torch.cuda.empty_cache()
        generation_summary, generation_rows = _generation(
            model, tokenizer, layer, arms, subset, options, log
        )
    finally:
        layer.mlp = original
        tokenizer.padding_side = original_padding_side
    _write_csv(output_dir / "free_running_generation.csv", generation_rows)
    timing = _summarize(rows, options, failures)
    timing["free_running_generation"] = generation_summary
    timing["measured_physical_dram_bytes"] = None
    timing["measured_bandwidth_status"] = "unknown: hardware counters not collected by this module"
    timing["speed_claims"] = {
        name: bool(
            all(part["no_unacceptable_task_degradation"] for part in generation_summary[name].values())
            and all(
                any(
                    row["scope"] == "complete_model_cached_decode"
                    and row["batch_size"] == batch
                    and row["method"].startswith(name + ":")
                    and row["timing_ci_supports_savings"]
                    for row in timing["rows"]
                )
                for batch in batches
            )
        )
        for name in arms if name != "dense"
    }
    timing["scope_limit"] = (
        "Decode results measure one cached token at fixed context. Free-running generation is "
        "a separate quality audit. No DRAM traffic or whole-sequence speedup is inferred."
    )
    _write_json(output_dir / "timing_summary.json", timing)
    return timing
