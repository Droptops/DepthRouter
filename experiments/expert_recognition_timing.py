"""Conditional economics for a frozen input-only final-MLP routing policy.

This module does not declare quality success or infer traffic from arithmetic.
Callers must supply measured development and independent held-out quality, plus
the preregistered portfolio-vs-binary continuation evidence. Routing and its CPU
synchronization are inside timed calls; prepacking, prefill and graph capture are
outside. The eager routed full model is compared with *both* eager and graph
dense, not merely the slower convenient baseline.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import hashlib
import json
import math
import random
import shutil
import subprocess
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from depth_router.expert_partitions import candidate_combinations, partition_indices
from depth_router.fused_swiglu import CUDAGraphCallable, PrepackedSwiGLU

SPLITS = ("development", "heldout_matched", "heldout_ood")
DENSE_ROUTE = 28


def require_quality_gate(
    seed_metrics: Mapping[str | int, Mapping[str, Mapping[str, Any]]],
    runtime_metrics: Mapping[str, Mapping[str, Any]],
    *,
    policy_family: str,
    portfolio_advantage_ci_low: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Fail closed before allocation, with explicit measured values for all splits.

    greedy_agreement and cheap_fraction are fractions, not percentages. Portfolio
    advantage is cheap-frequency improvement over the matched binary selector;
    the caller supplies the paired context-bootstrap lower 95% endpoint.
    """
    if policy_family not in ("portfolio", "binary"):
        raise ValueError("policy_family must be portfolio or binary")
    seeds = {str(key): value for key, value in seed_metrics.items()}
    if set(seeds) != {"17", "29", "43"}:
        raise ValueError("timing requires all three frozen router seeds")

    def check_rows(split_metrics, name):
        if set(split_metrics) != set(SPLITS):
            raise ValueError(f"{name} requires all three development and heldout splits")
        evidence = {}
        for split in SPLITS:
            row = split_metrics[split]
            required = ("greedy_agreement", "mean_kl", "cheap_fraction")
            if any(key not in row for key in required):
                raise ValueError(f"missing measured quality fields for {name}/{split}")
            values = {key: float(row[key]) for key in required}
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError(f"nonfinite quality evidence for {name}/{split}")
            if not 0.99 <= values["greedy_agreement"] <= 1.0:
                raise ValueError(f"agreement gate failed for {name}/{split}")
            if not 0.0 <= values["mean_kl"] < 0.01:
                raise ValueError(f"KL gate failed for {name}/{split}")
            if not 0.0 < values["cheap_fraction"] <= 1.0:
                raise ValueError(f"nonzero cheap-use gate failed for {name}/{split}")
            if row.get("quality_gate", {}).get("point_pass") is not True:
                raise ValueError(f"explicit point quality gate missing for {name}/{split}")
            evidence[split] = values
        return evidence

    seed_evidence = {seed: check_rows(rows, f"seed{seed}") for seed, rows in seeds.items()}
    runtime_evidence = check_rows(runtime_metrics, "runtime_batch1")
    for split in SPLITS:
        if runtime_metrics[split].get("batch_size") != 1:
            raise ValueError("runtime quality must come from genuine batch1 execution")
        if runtime_metrics[split].get("source") != "fresh_batch1_fullmodel_dense_prefix_capture":
            raise ValueError("runtime quality requires genuine fullmodel batch1 dense prefixes")
    advantage = {}
    if policy_family == "portfolio":
        if portfolio_advantage_ci_low is None or set(portfolio_advantage_ci_low) != set(SPLITS[1:]):
            raise ValueError("portfolio requires matched and OOD paired advantage intervals")
        for split, lower_value in portfolio_advantage_ci_low.items():
            lower = float(lower_value)
            if not math.isfinite(lower) or not 0.0 < lower <= 1.0:
                raise ValueError(f"resolved portfolio advantage gate failed for {split}")
            advantage[split] = lower
    return {"passed": True, "policy_family": policy_family, "seed_evidence": seed_evidence,
            "runtime_evidence": runtime_evidence, "portfolio_advantage_ci95_lower": advantage}


class RoutedPackedMLP(nn.Module):
    """x-only policy and 28 prepacked banks, with original dense as route 28.

    ``decision(x_flat)`` must return one int64 route per input row. It contains
    all preprocessing, selector inference, thresholding and dense fallback.
    No dense activation, reference logits, oracle label or weight gather is
    accessible to the decision. Cohort regrouping gathers *inputs*, never
    weights. CPU dispatch and the required device-to-host synchronization are
    intentional, visible costs of this bounded implementation.
    """

    def __init__(
        self, dense: nn.Module, labels: Tensor, decision: nn.Module,
        *, fixed_candidate: int | None = None,
    ) -> None:
        super().__init__()
        if labels.numel() != dense.gate_proj.out_features:
            raise ValueError("partition and dense width differ")
        self.dense = dense
        self.decision = decision
        if fixed_candidate is not None and not 0 <= fixed_candidate < DENSE_ROUTE:
            raise ValueError("fixed_candidate must be a candidate index 0..27")
        chosen = range(DENSE_ROUTE) if fixed_candidate is None else [fixed_candidate]
        combinations = candidate_combinations()
        self.candidates = nn.ModuleDict({
            str(index): PrepackedSwiGLU(
                dense.gate_proj, dense.up_proj, dense.down_proj,
                partition_indices(labels, combinations[index]).sort().values,
            )
            for index in chosen
        })
        self.hidden = dense.gate_proj.in_features
        self._branch_graphs: dict[int, CUDAGraphCallable] = {}
        self._decision_graph: CUDAGraphCallable | None = None

    def routes(self, x: Tensor) -> Tensor:
        flat = x.reshape(-1, self.hidden)
        result = self.decision(flat) if self._decision_graph is None else self._decision_graph(flat)
        if result.dtype != torch.long or result.shape != (len(flat),):
            raise ValueError("decision must return one int64 route per input row")
        return result

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.hidden:
            raise ValueError("input width does not match the final MLP")
        flat = x.reshape(-1, self.hidden)
        # This transfers at least one integer to CPU and synchronizes CUDA.
        # It is deliberately inside forward, including every timed call.
        routes = self.routes(x).tolist()
        if any(route != DENSE_ROUTE and str(route) not in self.candidates for route in routes):
            raise ValueError("decision returned an out-of-range route")
        if len(routes) == 1:
            route = routes[0]
            if self._branch_graphs:
                return self._branch_graphs[route](x)
            module = self.dense if route == DENSE_ROUTE else self.candidates[str(route)]
            return module(x)
        if self._branch_graphs or self._decision_graph is not None:
            raise ValueError("partially captured dispatch is restricted to one token")
        output = torch.empty_like(flat)
        for route in sorted(set(routes)):
            indices = torch.tensor(
                [index for index, chosen in enumerate(routes) if chosen == route],
                device=x.device, dtype=torch.long,
            )
            module = self.dense if route == DENSE_ROUTE else self.candidates[str(route)]
            output.index_copy_(0, indices, module(flat.index_select(0, indices)))
        return output.reshape(x.shape)

    def weight_accounting(self) -> dict[str, Any]:
        candidate_bytes = [candidate.weight_bytes for candidate in self.candidates.values()]
        selector_bytes = sum(
            value.numel() * value.element_size()
            for value in (*self.decision.parameters(), *self.decision.buffers())
        )
        dense_bytes = sum(value.numel() * value.element_size() for value in self.dense.parameters())
        index_bytes = sum(
            candidate.indices.numel() * candidate.indices.element_size()
            for candidate in self.candidates.values()
        )
        return {
            "candidate_banks": len(candidate_bytes),
            "candidate_weight_bytes": candidate_bytes,
            "duplicated_candidate_weight_bytes": sum(candidate_bytes),
            "selector_parameter_and_buffer_bytes": selector_bytes,
            "candidate_index_metadata_bytes": index_bytes,
            "original_dense_weight_bytes": dense_bytes,
            "extra_resident_bytes_before_graphs": sum(candidate_bytes) + selector_bytes + index_bytes,
            "physical_dram_bytes": None,
            "note": "resident tensor sizes are not measured device-memory traffic",
        }

    @torch.inference_mode()
    def capture_branches(self, example: Tensor) -> None:
        """Optional partial graphs; host dispatch and synchronization remain."""
        if example.shape != (1, 1, self.hidden) or not example.is_cuda:
            raise ValueError("partial graph dispatch requires CUDA shape [1,1,hidden]")
        self._decision_graph = CUDAGraphCallable(self.decision, example.reshape(1, self.hidden))
        self._branch_graphs = {int(index): CUDAGraphCallable(module, example)
                               for index, module in self.candidates.items()}
        self._branch_graphs[DENSE_ROUTE] = CUDAGraphCallable(self.dense, example)


@contextlib.contextmanager
def replace_mlp(layer: nn.Module, module: nn.Module) -> Iterator[None]:
    original = layer.mlp
    layer.mlp = module
    try:
        yield
    finally:
        layer.mlp = original


def _json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def sequence_selection(records: Sequence[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Deterministic context round-robin, never routing/performance-ranked."""
    if limit < 1 or not records:
        raise ValueError("a nonempty replay sequence and positive limit are required")
    by_context: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_context.setdefault(record["id"], []).append(record)
    ordered = [sorted(values, key=lambda row: row["position"])
               for _, values in sorted(by_context.items())]
    selected = [values[position] for position in range(max(map(len, ordered)))
                for values in ordered if position < len(values)]
    return selected[:limit]


def paired_speedup(
    dense: Sequence[float], routed: Sequence[float], *, seed: int = 17, resamples: int = 2000,
) -> dict[str, Any]:
    reference, proposed = np.asarray(dense, dtype=float), np.asarray(routed, dtype=float)
    if (reference.ndim != 1 or reference.shape != proposed.shape or len(reference) < 2
            or resamples < 100 or not np.isfinite(reference).all()
            or not np.isfinite(proposed).all() or (reference <= 0).any() or (proposed <= 0).any()):
        raise ValueError("paired positive finite timing samples and >=100 resamples required")
    indices = np.random.default_rng(seed).integers(0, len(reference), (resamples, len(reference)))
    ratios = reference[indices].mean(1) / proposed[indices].mean(1)
    low, high = np.quantile(ratios, [0.025, 0.975])
    return {"ratio_of_means": float(reference.mean() / proposed.mean()),
            "paired_block_bootstrap_ci95": [float(low), float(high)],
            "resolved_savings": bool(low > 1)}


@torch.inference_mode()
def measure_sequence(
    runners: Mapping[str, tuple[Callable[[Tensor], Tensor], Sequence[Tensor]]],
    *,
    scope: str,
    blocks: int,
    warmup: int,
    repetitions: int,
    seed: int,
    budget: Any,
    setups: Mapping[str, Callable[[], None]] | None = None,
) -> list[dict[str, Any]]:
    """Time identical state sequences, including policy inference and dispatch."""
    if blocks < 2 or warmup < 1 or repetitions < 1 or not runners:
        raise ValueError("positive warmup/repetitions and >=2 paired timing blocks required")
    lengths = {len(arguments) for _, arguments in runners.values()}
    if len(lengths) != 1 or 0 in lengths:
        raise ValueError("all methods must replay the same nonzero number of states")
    setups = {} if setups is None else setups
    for name, (run, arguments) in runners.items():
        if name in setups:
            setups[name]()
        for _ in range(warmup):
            for argument in arguments:
                budget.check()
                run(argument)
    torch.cuda.synchronize()
    rows = []
    rng = random.Random(seed)
    for block in range(blocks):
        names = list(runners)
        rng.shuffle(names)
        for order, name in enumerate(names):
            budget.check()
            run, arguments = runners[name]
            if name in setups:
                setups[name]()
            torch.cuda.synchronize()
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            started, finished = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            wall_start = time.perf_counter()
            started.record()
            for _ in range(repetitions):
                for argument in arguments:
                    run(argument)
            finished.record()
            finished.synchronize()
            count = repetitions * len(arguments)
            rows.append({
                "scope": scope, "method": name, "block": block, "order": order,
                "sequence_states": len(arguments), "repetitions": repetitions,
                "wall_ms_per_state": (time.perf_counter() - wall_start) * 1000 / count,
                "cuda_ms_per_state": started.elapsed_time(finished) / count,
                "peak_incremental_allocated_bytes": max(0, torch.cuda.max_memory_allocated() - baseline),
            })
    return rows


class FixedCachedDecode(nn.Module):
    """Overwrite exactly one cache position on every call; no growing history."""

    def __init__(self, model: nn.Module, cache: Any, length: int, device: torch.device) -> None:
        super().__init__()
        self.model, self.cache = model, cache
        self.register_buffer("attention_mask", torch.ones(1, length, device=device, dtype=torch.long))
        self.register_buffer("position_ids", torch.tensor([[length - 1]], device=device))
        self.register_buffer("cache_position", torch.tensor([length - 1], device=device))

    def forward(self, token: Tensor) -> Tensor:
        return self.model(
            input_ids=token, attention_mask=self.attention_mask, position_ids=self.position_ids,
            cache_position=self.cache_position, past_key_values=self.cache,
            use_cache=True, logits_to_keep=1,
        ).logits


@torch.inference_mode()
def prepare_cached_decode(model: nn.Module, prefix_token_ids: Sequence[int]) -> tuple[FixedCachedDecode, Tensor]:
    """Dense prefill is outside timing and shared unchanged by dense/routed decode."""
    from transformers import StaticCache

    if len(prefix_token_ids) < 2:
        raise ValueError("cached decode requires at least one prior token and one current token")
    device = next(model.parameters()).device
    ids = torch.tensor([list(prefix_token_ids)], device=device, dtype=torch.long)
    cache = StaticCache(config=model.config, max_cache_len=ids.shape[1])
    model(input_ids=ids[:, :-1], attention_mask=torch.ones_like(ids[:, :-1]),
          position_ids=torch.arange(ids.shape[1] - 1, device=device)[None],
          cache_position=torch.arange(ids.shape[1] - 1, device=device),
          past_key_values=cache, use_cache=True, logits_to_keep=1)
    return FixedCachedDecode(model, cache, ids.shape[1], device), ids[:, -1:]


@torch.inference_mode()
def validate_fullmodel_replay(
    model: nn.Module, dense: nn.Module, routed: RoutedPackedMLP, replay_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Validate routing with actual residual/norm/head, including cohort shape."""
    layer = model.model.layers[-1]
    captured: dict[str, Tensor] = {}

    def pre(name: str):
        def hook(_module, args):
            captured[name] = args[0].detach().clone()
        return hook

    handles = [dense.register_forward_pre_hook(pre("x")),
               layer.post_attention_layernorm.register_forward_pre_hook(pre("base"))]
    try:
        with replace_mlp(layer, dense):
            teacher = model(**copy.deepcopy(replay_kwargs), use_cache=True, logits_to_keep=1)
    finally:
        for handle in handles:
            handle.remove()
    predicted = routed(captured["x"])
    cached = model.lm_head(model.model.norm(captured["base"] + predicted)).float()[:, -1]
    with replace_mlp(layer, routed):
        actual = model(**copy.deepcopy(replay_kwargs), use_cache=True,
                       logits_to_keep=1).logits.float()[:, -1]
    if not torch.equal(cached, actual):
        raise RuntimeError("runtime routed cached logits differ from complete model")
    logp = teacher.logits.float()[:, -1].log_softmax(-1)
    kl = (logp.exp() * (logp - actual.log_softmax(-1))).sum(-1)
    return {"exact_cached_vs_fullmodel_logits": True,
            "states": len(actual), "runtime_shape": list(captured["x"].shape),
            "agreement": float((logp.argmax(-1) == actual.argmax(-1)).float().mean()),
            "mean_kl": float(kl.mean()), "max_kl": float(kl.max())}


def attempt_physical_counters(
    command: Sequence[str], output_dir: Path, *, timeout_seconds: int = 120,
) -> dict[str, Any]:
    """Attempt a caller-supplied frozen policy region, preserving exact failure.

    The command must enable/disable CUDA profiling around the complete routing
    region. A successful CSV still needs semantic inspection before claiming
    application-total DRAM bytes, so this helper never sums arbitrary rows.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    executable = shutil.which("ncu")
    if executable is None:
        result = {"status": "unavailable", "physical_dram_bytes": None,
                  "reason": "shutil.which('ncu') returned None in the experiment runtime"}
        _json(output_dir / "counter_status.json", result)
        return result
    if not command:
        raise ValueError("hardware counter capture needs a real frozen-policy command")
    argv = [executable, "--profile-from-start", "off", "--csv",
            "--metrics", "dram__bytes_read.sum,dram__bytes_write.sum", *command]
    try:
        capture = subprocess.run(argv, capture_output=True, text=True, check=False,
                                 timeout=timeout_seconds)
        raw = (capture.stdout or "") + (capture.stderr or "")
        result = {"status": "captured_requires_region_validation" if capture.returncode == 0 else "failed",
                  "returncode": capture.returncode, "physical_dram_bytes": None,
                  "exact_failure": None if capture.returncode == 0 else raw,
                  "command": argv}
    except (OSError, subprocess.TimeoutExpired) as exc:
        raw = str(exc)
        result = {"status": "failed", "physical_dram_bytes": None,
                  "exact_failure": f"{type(exc).__name__}: {exc}", "command": argv}
    (output_dir / "ncu.log").write_text(raw, encoding="utf-8")
    _json(output_dir / "counter_status.json", result)
    return result


def _summaries(rows: list[dict[str, Any]], *, seed: int, resamples: int) -> list[dict[str, Any]]:
    summary = []
    for scope in sorted({row["scope"] for row in rows}):
        by_method = {}
        for name in sorted({row["method"] for row in rows if row["scope"] == scope}):
            samples = sorted((row for row in rows if row["scope"] == scope and row["method"] == name),
                             key=lambda row: row["block"])
            by_method[name] = [row["wall_ms_per_state"] for row in samples]
        dense_names = [name for name in by_method if name.startswith("dense:")]
        strongest = min(dense_names, key=lambda name: np.mean(by_method[name]))
        for name, samples in by_method.items():
            comparisons = {dense: paired_speedup(values, samples, seed=seed, resamples=resamples)
                           for dense, values in by_method.items() if dense in dense_names}
            summary.append({"scope": scope, "method": name, "mean_wall_ms": float(np.mean(samples)),
                            "strongest_dense": strongest, "versus_dense": comparisons,
                            "resolved_savings": not name.startswith("dense:")
                            and "dense:graph" in by_method
                            and all(value["resolved_savings"] for value in comparisons.values())})
    return summary


@torch.inference_mode()
def run_isolated_timing(
    dense: nn.Module,
    routed: RoutedPackedMLP,
    groups: Sequence[dict[str, Tensor]],
    records: Sequence[dict[str, Any]],
    output_dir: Path,
    *,
    seed_metrics: Mapping[str | int, Mapping[str, Mapping[str, Any]]],
    runtime_metrics: Mapping[str, Mapping[str, Any]],
    policy_family: str,
    budget: Any,
    config: Mapping[str, Any],
    portfolio_advantage_ci_low: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Quality-gated isolated final-MLP sequence timing; not full-model savings."""
    gate = require_quality_gate(seed_metrics, runtime_metrics, policy_family=policy_family,
                                portfolio_advantage_ci_low=portfolio_advantage_ci_low)
    if not torch.cuda.is_available() or dense.gate_proj.weight.device.type != "cuda":
        raise RuntimeError("measured GPU timing requires CUDA weights")
    selected = sequence_selection(records, int(config.get("states", 64)))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _json(output_dir / "timing_sequence.json", selected)
    state_hash = hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()
    device = dense.gate_proj.weight.device
    arguments = [groups[row["group"]]["x"][row["slot"]:row["slot"] + 1].to(device)
                 for row in selected]
    routes = [int(routed.routes(argument).item()) for argument in arguments]
    native_graph = CUDAGraphCallable(dense, arguments[0])
    for argument in arguments:
        if not torch.equal(dense(argument), native_graph(argument)):
            raise RuntimeError("native dense eager/graph numerical mismatch")
    methods = {"dense:eager": (dense, arguments), "dense:graph": (native_graph, arguments),
               "routed:eager": (routed, arguments)}
    rows = measure_sequence(methods, scope="isolated_final_mlp_batch1", blocks=int(config.get("blocks", 20)),
                            warmup=int(config.get("warmup", 2)), repetitions=int(config.get("repetitions", 2)),
                            seed=int(config.get("seed", 17)), budget=budget)
    _csv(output_dir / "timing_blocks.csv", rows)
    result = {"quality_gate": gate, "scope": "isolated final MLP, not end-to-end decoding",
              "sequence_sha256": state_hash, "route_ids": routes,
              "dense_frequency": routes.count(DENSE_ROUTE) / len(routes),
              "memory": routed.weight_accounting(),
              "rows": _summaries(rows, seed=int(config.get("seed", 17)),
                                 resamples=int(config.get("bootstrap_resamples", 2000))),
              "physical_dram_bytes": None,
              "runtime_overhead": "selector, threshold, host sync, dispatch and all MLP launches included"}
    _json(output_dir / "timing_summary.json", result)
    return result


def static_cache_subset_gate(
    numerical: Sequence[Mapping[str, Any]], *, expected_states: int,
) -> dict[str, Any]:
    """Additional small timing-subset guard, not a replacement held-out audit."""
    if expected_states < 1 or len(numerical) != expected_states:
        raise ValueError("StaticCache preflight must cover the complete frozen timing subset")
    if any(row.get("dense_matches_stored_token") is not True
           or row.get("dense_graph_exact") is not True
           or row.get("routed_reconstruction_exact") is not True for row in numerical):
        raise RuntimeError("StaticCache timing subset failed a dense/parity technical control")
    kl = np.asarray([row["kl"] for row in numerical], dtype=float)
    agreement = np.asarray([row["greedy_agreement"] for row in numerical])
    if not np.isfinite(kl).all() or (kl < 0).any() or not np.isin(agreement, [False, True]).all():
        raise ValueError("StaticCache preflight requires finite nonnegative KL and boolean agreement")
    mean_kl, greedy = float(kl.mean()), float(agreement.mean())
    return {"states": expected_states, "greedy_agreement": greedy, "mean_kl": mean_kl,
            "point_pass": bool(greedy >= 0.99 and mean_kl < 0.01),
            "scope": "additional frozen small timing-subset guard only",
            "replaces_full_heldout_runtime_gate": False}


@torch.inference_mode()
def _prepare_timing_prefix(model, dense, routed, record):
    """Allocate one cache, check exact numerics, and return its untimed runners."""
    layer = model.model.layers[-1]
    layer.mlp = dense
    operation, token = prepare_cached_decode(model, record["prefix_token_ids"])
    captured = {}

    def save(name):
        def hook(_module, arguments):
            captured[name] = arguments[0].detach().clone()
        return hook

    handles = [dense.register_forward_pre_hook(save("x")),
               layer.post_attention_layernorm.register_forward_pre_hook(save("base"))]
    try:
        reference = operation(token).clone()
    finally:
        for handle in handles:
            handle.remove()
    if int(reference[:, -1].argmax(-1).item()) != int(record["dense_token"]):
        raise RuntimeError("StaticCache dense argmax differs from stored batch1 DynamicCache token")
    if not torch.equal(reference, operation(token)):
        raise RuntimeError("fixed dense cache does not reproduce the same output")
    graph = CUDAGraphCallable(operation, token)
    if not torch.equal(reference, graph(token)):
        raise RuntimeError("complete-model dense eager/graph logits differ")
    route = int(routed.routes(captured["x"]).item())
    predicted = model.lm_head(model.model.norm(captured["base"] + routed(captured["x"])))
    layer.mlp = routed
    actual = operation(token).clone()
    if not torch.equal(actual, operation(token)) or not torch.equal(actual, predicted):
        raise RuntimeError("complete-model routed/cache reconstruction mismatch")
    logp, logq = reference.float().log_softmax(-1), actual.float().log_softmax(-1)
    numerical = {"id": record["id"], "position": record["position"],
                 "dense_graph_exact": True, "routed_reconstruction_exact": True,
                 "dense_matches_stored_token": True, "route": route,
                 "greedy_agreement": bool(reference.argmax(-1).eq(actual.argmax(-1)).all()),
                 "kl": float((logp.exp() * (logp - logq)).sum(-1).clamp_min(0).mean())}
    return operation, token, graph, numerical


@torch.inference_mode()
def run_cached_decode_timing(
    model: nn.Module,
    dense: nn.Module,
    routed: RoutedPackedMLP,
    records: Sequence[dict[str, Any]],
    output_dir: Path,
    *,
    seed_metrics: Mapping[str | int, Mapping[str, Mapping[str, Any]]],
    runtime_metrics: Mapping[str, Mapping[str, Any]],
    policy_family: str,
    budget: Any,
    config: Mapping[str, Any],
    portfolio_advantage_ci_low: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Measure a representative sequence of genuine fixed-cache batch1 decodes.

    Each dense prefix is prepared with the original model. Dense/routed calls
    overwrite the same final KV slot, which is independent of the final MLP.
    No prefill, cache allocation/reset, graph capture or model-module replacement
    occurs in timed regions. Every model layer, head, route inference, dispatch,
    CPU synchronization and final synchronization does occur in timed regions.
    All fixed timing prefixes pass a StaticCache preflight BEFORE any timed
    call. A failed dense graph or numerical check leaves savings unknown; a
    failed subset fidelity gate prevents timing. This guard does not replace
    the full development/held-out DynamicCache runtime quality evaluation.
    """
    gate = require_quality_gate(seed_metrics, runtime_metrics, policy_family=policy_family,
                                portfolio_advantage_ci_low=portfolio_advantage_ci_low)
    if not torch.cuda.is_available() or dense.gate_proj.weight.device.type != "cuda":
        raise RuntimeError("measured GPU timing requires CUDA weights")
    selected = sequence_selection(records, int(config.get("fullmodel_states", 8)))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _json(output_dir / "cached_decode_sequence.json", selected)
    layer, original = model.model.layers[-1], model.model.layers[-1].mlp
    rows, numerical, routes = [], [], []
    failure, subset_gate = None, None
    try:
        # Validate the entire frozen subset before timing any prefix. Release
        # each cache/graph immediately, then recreate it outside timed regions.
        for record in selected:
            budget.check()
            operation, token, graph, checked = _prepare_timing_prefix(model, dense, routed, record)
            numerical.append(checked)
            _json(output_dir / "cached_decode_numerical_controls.json", numerical)
            del graph, operation, token
        subset_gate = static_cache_subset_gate(numerical, expected_states=len(selected))
        _json(output_dir / "static_cache_subset_gate.json", subset_gate)
        if not subset_gate["point_pass"]:
            failure = {"type": "StaticCacheSubsetQualityFailure",
                       "exact_error": "frozen StaticCache timing subset fails 99% agreement or mean KL <0.01",
                       "completed_sequence_states": 0}
        for index, record in enumerate(selected):
            if failure is not None:
                break
            budget.check()
            operation, token, graph, checked = _prepare_timing_prefix(model, dense, routed, record)
            if checked != numerical[index]:
                raise RuntimeError("StaticCache timing prefix changed after the complete subset preflight")
            route = checked["route"]
            routes.append(route)
            setups = {"dense:eager": lambda: setattr(layer, "mlp", dense),
                      "dense:graph": lambda: setattr(layer, "mlp", dense),
                      "routed:eager": lambda: setattr(layer, "mlp", routed)}
            methods = {"dense:eager": (operation, [token]), "dense:graph": (graph, [token]),
                       "routed:eager": (operation, [token])}
            measured = measure_sequence(
                methods, scope="complete_model_cached_decode_batch1",
                blocks=int(config.get("blocks", 20)), warmup=int(config.get("warmup", 2)),
                repetitions=int(config.get("repetitions", 2)),
                seed=int(config.get("seed", 17)) + index, budget=budget, setups=setups,
            )
            rows.extend({**row, "id": record["id"], "position": record["position"],
                         "sequence_index": index, "route": route} for row in measured)
            _csv(output_dir / "cached_decode_blocks.csv", rows)
            del graph, operation, token
    except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
        failure = {"type": type(exc).__name__, "exact_error": str(exc),
                   "completed_sequence_states": len(routes)}
    finally:
        layer.mlp = original
    # Every block below averages the SAME fixed sequence. Pairing the sequence
    # means avoids treating repeated calls to one prompt as independent prompts.
    averaged = []
    if rows and failure is None:
        for method in sorted({row["method"] for row in rows}):
            for block in sorted({row["block"] for row in rows}):
                samples = [row for row in rows if row["method"] == method and row["block"] == block]
                averaged.append({"method": method, "block": block,
                                 "scope": "complete_model_cached_decode_batch1",
                                 "wall_ms_per_state": float(np.mean([row["wall_ms_per_state"] for row in samples]))})
    status = "complete" if failure is None else "unknown_due_to_technical_failure"
    if subset_gate is not None and not subset_gate["point_pass"]:
        status = "skipped_static_cache_subset_quality_failed"
    result = {
        "status": status,
        "quality_gate": gate, "failure": failure,
        "scope": "fixed-prefix one-token cached decode; not whole-sequence generation speed",
        "timing_cache_backend": "transformers.StaticCache",
        "full_runtime_quality_cache_backend": "transformers.DynamicCache",
        "static_cache_subset_gate": subset_gate,
        "all_prefixes_preflighted_before_timed_calls": failure is None or subset_gate is not None,
        "sequence_states": len(selected), "route_ids": routes,
        "dense_frequency": routes.count(DENSE_ROUTE) / len(routes) if routes else None,
        "rows": _summaries(averaged, seed=int(config.get("seed", 17)),
                            resamples=int(config.get("bootstrap_resamples", 2000))) if averaged else [],
        "memory": routed.weight_accounting(), "physical_dram_bytes": None,
        "routed_fullmodel_graph": "not available: causal CPU branch and synchronization remain in the measured eager policy",
        "graph_dense_required": True,
        "exclusions": ["dense prefill", "static-cache allocation", "graph capture", "module replacement"],
        "included": ["all decoder layers", "final norm", "full vocabulary head", "selector", "threshold",
                     "CPU synchronization", "dispatch", "all kernel launches", "final synchronization"],
    }
    _json(output_dir / "cached_decode_summary.json", result)
    return result


def profile_evidence(run_dir: Path, family: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read and validate persisted gates before the profiler may touch CUDA."""
    from expert_construction import sha_file, verify_seal
    from expert_recognition import verify_sources

    if family not in ("portfolio", "binary"):
        raise ValueError("profiler family must be portfolio or binary")
    run_dir = Path(run_dir)
    verify_sources(run_dir)
    registration = json.loads((run_dir / "preregistration.json").read_text(encoding="utf-8"))
    seal = json.loads((run_dir / "heldout_seal.json").read_text(encoding="utf-8"))
    verify_seal(seal, run_dir)
    decisions = json.loads((run_dir / "recognition_decision.json").read_text(encoding="utf-8"))
    if decisions[family].get("eligible_for_runtime_quality") is not True:
        raise ValueError("profiling forbidden for a policy that failed recognition gates")
    seeds = json.loads((run_dir / "recognition_metrics.json").read_text(encoding="utf-8"))[family]
    runtime = json.loads((run_dir / "runtime/runtime_quality.json").read_text(encoding="utf-8"))[family]["quality"]
    paired = json.loads((run_dir / "paired_portfolio_minus_binary.json").read_text(encoding="utf-8"))
    gate = require_quality_gate(
        seeds, runtime, policy_family=family,
        portfolio_advantage_ci_low={split: paired[split]["cheap_fraction"]["lower"]
                                    for split in SPLITS[1:]},
    )
    cache = run_dir / "runtime/cache-heldout_matched.pt"
    cache_metadata = json.loads((run_dir / "runtime/capture-heldout_matched.json").read_text(encoding="utf-8"))
    if sha_file(cache) != cache_metadata["cache_sha256"]:
        raise ValueError("runtime cache hash differs from genuine batch1 capture")
    return registration["config"], gate


@torch.inference_mode()
def profile_frozen_run(run_dir: Path, family: str) -> None:
    """Nsight target: the exact frozen policy over the same 64 timing states."""
    from expert_recognition import FrozenDecision
    from transformers import AutoModelForCausalLM

    from depth_router.expert_recognition import load_checkpoint

    run_dir = Path(run_dir)
    config, gate = profile_evidence(run_dir, family)
    if not torch.cuda.is_available():
        raise RuntimeError("physical GPU profiling requires CUDA")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    choices = json.loads((run_dir / "frozen_policy_choices.json").read_text(encoding="utf-8"))
    primary = str(config["primary_runtime_seed"])
    if primary != "17":
        raise ValueError("the profiler must use preregistered primary seed17")
    choice = choices[family][primary]
    selector = load_checkpoint(
        run_dir / f"selector-{family}-{choice['architecture']}-{primary}.pt", device="cuda")
    model = AutoModelForCausalLM.from_pretrained(
        config["model"], revision=config["revision"], torch_dtype=torch.bfloat16,
        attn_implementation=config["attention"], local_files_only=True,
    ).cuda().eval().requires_grad_(False)
    dense = model.model.layers[-1].mlp
    labels = torch.tensor(json.loads((run_dir / "frozen_partition.json").read_text(encoding="utf-8")))
    routed = RoutedPackedMLP(
        dense, labels, FrozenDecision(selector, choice),
        fixed_candidate=config["fixed_candidate_index"] if family == "binary" else None,
    ).eval()
    groups = torch.load(run_dir / "runtime/cache-heldout_matched.pt", map_location="cpu", weights_only=True)
    records = [json.loads(line) for line in
               (run_dir / "runtime/states-heldout_matched.jsonl").read_text(encoding="utf-8").splitlines()]
    selected = sequence_selection(records, int(config["runtime"]["states"]))
    timed = json.loads((run_dir / f"runtime/{family}/isolated/timing_sequence.json").read_text(encoding="utf-8"))
    if selected != timed:
        raise ValueError("profile target sequence differs from the frozen timing sequence")
    arguments = [groups[row["group"]]["x"][row["slot"]:row["slot"] + 1].cuda() for row in selected]
    if any(argument.shape != (1, 1, dense.gate_proj.in_features) for argument in arguments):
        raise ValueError("profile inputs are not genuine single-token batch1 states")
    for argument in arguments:
        routed(argument)
    torch.cuda.synchronize()
    destination = run_dir / "runtime" / family / "counters"
    destination.mkdir(parents=True, exist_ok=True)
    _json(destination / "profile_target.json", {
        "family": family, "quality_gate": gate, "states": len(arguments),
        "scope": "complete isolated final-MLP policy including selector, dispatch and dense fallback",
        "sequence_sha256": hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest(),
        "route_ids": [int(routed.routes(argument).item()) for argument in arguments],
        "physical_dram_bytes": None,
    })
    torch.cuda.cudart().cudaProfilerStart()
    try:
        for argument in arguments:
            routed(argument)
        torch.cuda.synchronize()
    finally:
        torch.cuda.cudart().cudaProfilerStop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-run", type=Path, required=True)
    parser.add_argument("--family", choices=("portfolio", "binary"), required=True)
    args = parser.parse_args()
    profile_frozen_run(args.profile_run, args.family)


if __name__ == "__main__":
    main()
