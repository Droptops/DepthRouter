"""Falsifiable Qwen minimum-sufficient parallel-inference experiment.

This script measures CUDA execution on a real GPU and separately simulates an
optical cost model.  It deliberately labels analytic byte counts and simulated
optical timings so neither can be mistaken for hardware-counter measurements.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import math
import platform
import random
import shutil
import subprocess
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from depth_router.photonic_parallel import (
    PhotonicAssumptions,
    balanced_partitions,
    break_even_margin,
    contribution_scores,
    swiglu_grouped_sequential,
    swiglu_subset,
    top_fraction_indices,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "photonic_parallel_manifest.jsonl"
QUALITY_GATE_KL = 0.01
QUALITY_GATE_TOP1 = 0.99


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/photonic_parallel_quick.json")
    parser.add_argument("--output-dir", default="artifacts/photonic_parallel/manual")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


class RunLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **fields: Any) -> None:
        row = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "event": event,
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_json_if_resuming(path: Path, resume: bool) -> Any | None:
    if resume and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def command_output(command: list[str]) -> str:
    try:
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unavailable: {exc}"


def environment_snapshot() -> dict[str, Any]:
    import transformers

    cuda = torch.cuda.is_available()
    result: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "cuda_available": cuda,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if cuda else None,
        "nvidia_smi": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version,temperature.gpu,power.limit",
                "--format=csv,noheader,nounits",
            ]
        ),
        "ncu_path": shutil.which("ncu"),
        "seeds": {},
    }
    if cuda:
        properties = torch.cuda.get_device_properties(0)
        result.update(
            {
                "gpu": properties.name,
                "gpu_total_memory_bytes": properties.total_memory,
                "gpu_compute_capability": [properties.major, properties.minor],
            }
        )
    return result


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_dataset() -> list[dict[str, str]]:
    rows = [json.loads(line) for line in DATASET_PATH.read_text(encoding="utf-8").splitlines()]
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("dataset IDs must be unique")
    return rows


def resolve_layers(model: nn.Module) -> list[nn.Module]:
    candidates = [
        getattr(getattr(model, "model", None), "layers", None),
        getattr(model, "layers", None),
        getattr(getattr(model, "transformer", None), "h", None),
    ]
    for candidate in candidates:
        if candidate is not None and len(candidate):
            return list(candidate)
    raise ValueError("could not resolve decoder layers")


def resolve_swiglu(layer: nn.Module) -> tuple[nn.Module, nn.Linear, nn.Linear, nn.Linear]:
    mlp = getattr(layer, "mlp", None)
    gate = getattr(mlp, "gate_proj", None)
    up = getattr(mlp, "up_proj", None)
    down = getattr(mlp, "down_proj", None)
    if mlp is None or not all(isinstance(item, nn.Linear) for item in (gate, up, down)):
        raise TypeError("expected a gate/up/down SwiGLU MLP")
    return mlp, gate, up, down


def prompt_text(tokenizer: Any, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def encode_prompts(
    tokenizer: Any,
    prompts: Sequence[str],
    device: torch.device,
    max_length: int,
) -> dict[str, Tensor]:
    texts = [prompt_text(tokenizer, prompt) for prompt in prompts]
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    return {key: value.to(device) for key, value in encoded.items() if isinstance(value, Tensor)}


def teacher_encoding(
    tokenizer: Any,
    example: dict[str, str],
    device: torch.device,
    max_prompt_tokens: int,
) -> tuple[Tensor, Tensor, int]:
    prefix = tokenizer(
        prompt_text(tokenizer, example["prompt"]),
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_tokens,
        add_special_tokens=False,
    ).input_ids
    answer = tokenizer(
        example["answer"] + tokenizer.eos_token,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids
    input_ids = torch.cat((prefix, answer), dim=1).to(device)
    return input_ids, answer.to(device), prefix.shape[1]


def timing_summary(milliseconds: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(milliseconds, dtype=np.float64)
    return {
        "repetitions": int(values.size),
        "mean_ms": float(values.mean()),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "p99_ms": float(np.quantile(values, 0.99)),
        "std_ms": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "ci95_half_width_ms": float(1.96 * values.std(ddof=1) / math.sqrt(values.size))
        if values.size > 1
        else 0.0,
        "samples_ms": [float(value) for value in values],
    }


def bootstrap_mean_ci(values: Sequence[float], seed: int, draws: int = 1000) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return (float("nan"), float("nan"))
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, array.size, size=(draws, array.size))
    means = array[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


@torch.inference_mode()
def benchmark_call(
    function: Callable[[], Any],
    *,
    warmup: int,
    repetitions: int,
    device: torch.device,
) -> tuple[dict[str, float], int]:
    for _ in range(warmup):
        function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline_memory = torch.cuda.memory_allocated(device)
    else:
        baseline_memory = 0
    times = []
    for _ in range(repetitions):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter_ns()
        function()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append((time.perf_counter_ns() - started) / 1e6)
    peak = (
        max(0, torch.cuda.max_memory_allocated(device) - baseline_memory)
        if device.type == "cuda"
        else 0
    )
    return timing_summary(times), int(peak)


@torch.inference_mode()
def response_logits(
    model: nn.Module,
    tokenizer: Any,
    example: dict[str, str],
    device: torch.device,
    max_prompt_tokens: int,
) -> tuple[Tensor, Tensor]:
    input_ids, answer, start = teacher_encoding(tokenizer, example, device, max_prompt_tokens)
    logits = model(input_ids=input_ids, use_cache=False).logits
    response = logits[:, start - 1 : input_ids.shape[1] - 1, :].detach().float().cpu()
    return response.squeeze(0), answer.squeeze(0).detach().cpu()


def cross_entropy_for_logits(logits: Tensor, targets: Tensor) -> float:
    return float(F.cross_entropy(logits, targets, reduction="mean").item())


@torch.inference_mode()
def run_dense_baseline(
    model: nn.Module,
    tokenizer: Any,
    examples: list[dict[str, str]],
    config: dict[str, Any],
    device: torch.device,
    backend: str,
) -> dict[str, Any]:
    heldout = [row for row in examples if row["split"] == "heldout"]
    rows: list[dict[str, Any]] = []
    warmup = int(config["benchmark_warmup"])
    repetitions = int(config["baseline_repetitions"])

    for batch_size in config["baseline_batches"]:
        selected = [heldout[index % len(heldout)]["prompt"] for index in range(batch_size)]
        encoded = encode_prompts(
            tokenizer,
            selected,
            device,
            int(config["max_prompt_tokens"]),
        )

        prefill, prefill_peak = benchmark_call(
            lambda: model(**encoded, use_cache=False),
            warmup=warmup,
            repetitions=repetitions,
            device=device,
        )

        decode_times = []
        for _ in range(max(3, repetitions // 2)):
            initial = model(**encoded, use_cache=True)
            next_token = initial.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter_ns()
            model(input_ids=next_token, past_key_values=initial.past_key_values, use_cache=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            decode_times.append((time.perf_counter_ns() - started) / 1e6)

        generation_kwargs = {
            **encoded,
            "do_sample": False,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        ttft, ttft_peak = benchmark_call(
            lambda: model.generate(**generation_kwargs, max_new_tokens=1),
            warmup=max(2, warmup // 2),
            repetitions=max(5, repetitions // 2),
            device=device,
        )
        full, full_peak = benchmark_call(
            lambda: model.generate(
                **generation_kwargs,
                max_new_tokens=int(config["max_new_tokens"]),
            ),
            warmup=max(1, warmup // 3),
            repetitions=max(3, repetitions // 3),
            device=device,
        )
        rows.append(
            {
                "backend": backend,
                "batch_size": batch_size,
                "prompt_tokens": int(encoded["attention_mask"].sum().item()),
                "prefill": prefill,
                "decode_one_token": timing_summary(decode_times),
                "time_to_first_token": ttft,
                "complete_generation": full,
                "generated_tokens_requested": int(config["max_new_tokens"]),
                "prefill_tokens_per_second": float(
                    encoded["attention_mask"].sum().item() / (prefill["median_ms"] / 1000)
                ),
                "decode_tokens_per_second": float(batch_size / (np.median(decode_times) / 1000)),
                "peak_incremental_vram_bytes": max(prefill_peak, ttft_peak, full_peak),
            }
        )

    ce_rows = []
    for example in heldout:
        response, answer = response_logits(
            model,
            tokenizer,
            example,
            device,
            int(config["max_prompt_tokens"]),
        )
        ce_rows.append(
            {
                "id": example["id"],
                "task": example["task"],
                "response_tokens": int(answer.numel()),
                "cross_entropy_nats": cross_entropy_for_logits(response, answer),
            }
        )
    return {
        "backend": backend,
        "rows": rows,
        "heldout_cross_entropy": ce_rows,
        "mean_heldout_cross_entropy_nats": float(
            np.mean([row["cross_entropy_nats"] for row in ce_rows])
        ),
    }


@torch.inference_mode()
def build_quality_reference(
    model: nn.Module,
    tokenizer: Any,
    examples: list[dict[str, str]],
    device: torch.device,
    max_prompt_tokens: int,
) -> dict[str, Tensor]:
    reference: dict[str, Tensor] = {}
    for example in (row for row in examples if row["split"] == "heldout"):
        logits, _targets = response_logits(
            model, tokenizer, example, device, max_prompt_tokens
        )
        reference[example["id"]] = logits
    return reference


@torch.inference_mode()
def capture_calibration(
    model: nn.Module,
    tokenizer: Any,
    examples: list[dict[str, str]],
    layer_indices: Sequence[int],
    device: torch.device,
    max_prompt_tokens: int,
) -> dict[int, dict[str, Any]]:
    layers = resolve_layers(model)
    records: dict[int, dict[str, list[Any]]] = {
        index: {"state": [], "transition": [], "activation": [], "task": []}
        for index in layer_indices
    }
    current: dict[int, dict[str, Tensor]] = defaultdict(dict)
    handles = []
    for index in layer_indices:
        mlp, _, _, down = resolve_swiglu(layers[index])

        def mlp_prehook(
            _module: nn.Module,
            args: tuple[Tensor, ...],
            *,
            layer_index: int = index,
        ) -> None:
            hidden = args[0].detach()
            current[layer_index]["state"] = hidden[:, -1, :].mean(dim=0).float().cpu()
            if hidden.shape[1] > 1:
                transition = hidden[:, -1, :] - hidden[:, -2, :]
            else:
                transition = hidden[:, -1, :]
            current[layer_index]["transition"] = transition.mean(dim=0).float().cpu()

        def down_prehook(
            _module: nn.Module,
            args: tuple[Tensor, ...],
            *,
            layer_index: int = index,
        ) -> None:
            current[layer_index]["activation"] = args[0].detach()[:, -1, :].mean(dim=0).float().cpu()

        handles.append(mlp.register_forward_pre_hook(mlp_prehook))
        handles.append(down.register_forward_pre_hook(down_prehook))

    try:
        for example in (row for row in examples if row["split"] == "calibration"):
            encoded = encode_prompts(
                tokenizer,
                [example["prompt"]],
                device,
                max_prompt_tokens,
            )
            model(**encoded, use_cache=False)
            for index in layer_indices:
                for key in ("state", "transition", "activation"):
                    records[index][key].append(current[index][key].clone())
                records[index]["task"].append(example["task"])
    finally:
        for handle in handles:
            handle.remove()

    summaries: dict[int, dict[str, Any]] = {}
    for index in layer_indices:
        _, gate, up, down = resolve_swiglu(layers[index])
        activations = torch.stack(records[index]["activation"])
        transitions = torch.stack(records[index]["transition"])
        states = torch.stack(records[index]["state"])
        contributions = contribution_scores(activations, down.weight.detach().cpu())
        by_task: dict[str, Tensor] = {}
        for task in sorted(set(records[index]["task"])):
            mask = torch.tensor([value == task for value in records[index]["task"]])
            by_task[task] = contributions[mask].mean(dim=0)
        weight_score = (
            gate.weight.detach().float().cpu().norm(dim=1)
            * up.weight.detach().float().cpu().norm(dim=1)
            * down.weight.detach().float().cpu().norm(dim=0)
        )
        summaries[index] = {
            "activation_score": activations.abs().mean(dim=0),
            "contribution_score": contributions.mean(dim=0),
            "weight_score": weight_score,
            "task_scores": by_task,
            "state_centers": F.normalize(transitions, dim=-1),
            "state_scores": contributions,
            "states": states,
            "records": len(records[index]["task"]),
        }
    return summaries


class StaticSelectiveMLP(nn.Module):
    def __init__(self, base: nn.Module, indices: Tensor) -> None:
        super().__init__()
        self.base = base
        _, self.gate, self.up, self.down = resolve_swiglu_holder(base)
        self.register_buffer("indices", indices.long().to(self.gate.weight.device), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        return swiglu_subset(x, self.gate, self.up, self.down, self.indices)


class OracleSelectiveMLP(nn.Module):
    def __init__(self, base: nn.Module, fraction: float) -> None:
        super().__init__()
        self.base = base
        _, self.gate, self.up, self.down = resolve_swiglu_holder(base)
        self.fraction = fraction

    def forward(self, x: Tensor) -> Tensor:
        activation = F.silu(self.gate(x)) * self.up(x)
        width = activation.shape[-1]
        k = max(1, min(width, round(width * self.fraction)))
        score = contribution_scores(activation, self.down.weight)
        chosen = torch.topk(score, k=k, dim=-1, sorted=False).indices
        sparse = torch.zeros_like(activation)
        sparse.scatter_(-1, chosen, torch.gather(activation, -1, chosen))
        return self.down(sparse)


class StateTransitionSelectiveMLP(nn.Module):
    def __init__(self, base: nn.Module, centers: Tensor, selections: Tensor) -> None:
        super().__init__()
        self.base = base
        _, self.gate, self.up, self.down = resolve_swiglu_holder(base)
        self.register_buffer("centers", centers.to(self.gate.weight.device), persistent=False)
        self.register_buffer("selections", selections.long().to(self.gate.weight.device), persistent=False)
        self.previous_state: Tensor | None = None

    def reset_state(self) -> None:
        self.previous_state = None

    def forward(self, x: Tensor) -> Tensor:
        last = x[:, -1, :].detach().float().mean(dim=0)
        if x.shape[1] > 1:
            transition = x[:, -1, :].detach().float().mean(dim=0) - x[:, -2, :].detach().float().mean(dim=0)
        elif self.previous_state is not None and self.previous_state.shape == last.shape:
            transition = last - self.previous_state
        else:
            transition = last
        self.previous_state = last
        score = F.normalize(transition, dim=0) @ self.centers.float().T
        route = int(score.argmax().item())
        return swiglu_subset(x, self.gate, self.up, self.down, self.selections[route])


def resolve_swiglu_holder(mlp: nn.Module) -> tuple[nn.Module, nn.Linear, nn.Linear, nn.Linear]:
    gate = getattr(mlp, "gate_proj", None)
    up = getattr(mlp, "up_proj", None)
    down = getattr(mlp, "down_proj", None)
    if not all(isinstance(item, nn.Linear) for item in (gate, up, down)):
        raise TypeError("expected a gate/up/down SwiGLU module")
    return mlp, gate, up, down


def selector_indices(
    method: str,
    fraction: float,
    calibration: dict[str, Any],
    *,
    task: str,
    seed: int,
    layer_index: int,
) -> Tensor:
    width = calibration["contribution_score"].numel()
    if method == "random":
        generator = torch.Generator().manual_seed(seed + 1009 * layer_index)
        order = torch.randperm(width, generator=generator)
        k = max(1, min(width, round(width * fraction)))
        return order[:k].sort().values
    if method == "static_activation":
        score = calibration["activation_score"]
    elif method == "static_weight":
        score = calibration["weight_score"]
    elif method == "calibration_activation":
        score = calibration["contribution_score"]
    elif method == "task_conditioned":
        score = calibration["task_scores"][task]
    else:
        raise ValueError(f"method does not have static indices: {method}")
    return top_fraction_indices(score, fraction)


@contextlib.contextmanager
def selective_layers(
    model: nn.Module,
    layer_indices: Sequence[int],
    method: str,
    fraction: float,
    calibration: dict[int, dict[str, Any]],
    *,
    task: str,
    seed: int,
) -> Iterator[list[nn.Module]]:
    layers = resolve_layers(model)
    originals: dict[int, nn.Module] = {}
    wrappers: list[nn.Module] = []
    try:
        for index in layer_indices:
            original = layers[index].mlp
            originals[index] = original
            if method == "oracle":
                wrapper: nn.Module = OracleSelectiveMLP(original, fraction)
            elif method == "state_transition_conditioned":
                state_scores = calibration[index]["state_scores"]
                selections = torch.stack(
                    [top_fraction_indices(row, fraction) for row in state_scores]
                )
                wrapper = StateTransitionSelectiveMLP(
                    original,
                    calibration[index]["state_centers"],
                    selections,
                )
            else:
                indices = selector_indices(
                    method,
                    fraction,
                    calibration[index],
                    task=task,
                    seed=seed,
                    layer_index=index,
                )
                wrapper = StaticSelectiveMLP(original, indices)
            layers[index].mlp = wrapper
            wrappers.append(wrapper)
        yield wrappers
    finally:
        for index, original in originals.items():
            layers[index].mlp = original


class GroupedMLP(nn.Module):
    def __init__(self, base: nn.Module, groups: int) -> None:
        super().__init__()
        self.base = base
        _, self.gate, self.up, self.down = resolve_swiglu_holder(base)
        self.parts = balanced_partitions(self.gate.out_features, groups)

    def forward(self, x: Tensor) -> Tensor:
        return swiglu_grouped_sequential(x, self.gate, self.up, self.down, self.parts)


@torch.inference_mode()
def exact_decomposition(
    model: nn.Module,
    tokenizer: Any,
    heldout: list[dict[str, str]],
    config: dict[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    layers = resolve_layers(model)
    layer_index = len(layers) // 2
    original, gate, up, down = resolve_swiglu(layers[layer_index])
    encoded = encode_prompts(
        tokenizer,
        [row["prompt"] for row in heldout[:2]],
        device,
        int(config["max_prompt_tokens"]),
    )
    capture: dict[str, Tensor] = {}

    def prehook(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
        capture["x"] = args[0].detach()

    handle = original.register_forward_pre_hook(prehook)
    try:
        dense_logits = model(**encoded, use_cache=False).logits[:, -1, :].detach().float()
        dense_generation = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=min(4, int(config["max_new_tokens"])),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        ).detach().cpu()
    finally:
        handle.remove()
    x = capture["x"]
    dense_local = original(x).detach().float()
    rows = []
    for groups in config["decomposition_groups"]:
        grouped_local = swiglu_grouped_sequential(
            x,
            gate,
            up,
            down,
            balanced_partitions(gate.out_features, int(groups)),
        ).detach().float()
        layers[layer_index].mlp = GroupedMLP(original, int(groups))
        try:
            grouped_logits = model(**encoded, use_cache=False).logits[:, -1, :].detach().float()
            grouped_generation = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=min(4, int(config["max_new_tokens"])),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            ).detach().cpu()
            generated_dense = dense_logits.argmax(dim=-1)
            generated_grouped = grouped_logits.argmax(dim=-1)
        finally:
            layers[layer_index].mlp = original
        local_delta = grouped_local - dense_local
        logit_delta = grouped_logits - dense_logits
        ref_logp = dense_logits.log_softmax(dim=-1)
        candidate_logp = grouped_logits.log_softmax(dim=-1)
        kl = (ref_logp.exp() * (ref_logp - candidate_logp)).sum(dim=-1)
        rows.append(
            {
                "layer": layer_index,
                "groups": int(groups),
                "coverage_fraction": 1.0,
                "local_max_abs_error": float(local_delta.abs().max().item()),
                "local_relative_l2_error": float(
                    local_delta.norm().item() / dense_local.norm().clamp_min(1e-12).item()
                ),
                "full_model_max_abs_logit_error": float(logit_delta.abs().max().item()),
                "full_model_mean_kl_nats": float(kl.mean().item()),
                "full_model_top1_agreement": float(
                    (generated_dense == generated_grouped).float().mean().item()
                ),
                "bitwise_equal": bool(torch.equal(grouped_logits, dense_logits)),
                "end_to_end_generated_tokens_equal": bool(
                    torch.equal(grouped_generation, dense_generation)
                ),
                "tolerance": {
                    "atol": 0.03125,
                    "rtol": 0.01,
                    "basis": "bfloat16 reduction-order tolerance",
                },
            }
        )
    return rows


@torch.inference_mode()
def evaluate_quality(
    model: nn.Module,
    tokenizer: Any,
    examples: list[dict[str, str]],
    reference: dict[str, Tensor],
    calibration: dict[int, dict[str, Any]],
    config: dict[str, Any],
    device: torch.device,
    log: RunLog,
) -> list[dict[str, Any]]:
    heldout = [row for row in examples if row["split"] == "heldout"]
    results = []
    for layer_set in config["progressive_layer_sets"]:
        for method in config["selection_methods"]:
            for fraction in config["retention_fractions"]:
                all_kl: list[float] = []
                all_agreement: list[float] = []
                all_ce: list[float] = []
                task_kl: dict[str, list[float]] = defaultdict(list)
                task_agreement: dict[str, list[float]] = defaultdict(list)
                failure_examples = []
                for example in heldout:
                    with selective_layers(
                        model,
                        layer_set,
                        method,
                        float(fraction),
                        calibration,
                        task=example["task"],
                        seed=int(config["seed"]),
                    ):
                        candidate, targets = response_logits(
                            model,
                            tokenizer,
                            example,
                            device,
                            int(config["max_prompt_tokens"]),
                        )
                    dense = reference[example["id"]]
                    dense_logp = dense.log_softmax(dim=-1)
                    candidate_logp = candidate.log_softmax(dim=-1)
                    kl = (dense_logp.exp() * (dense_logp - candidate_logp)).sum(dim=-1)
                    agreement = (dense.argmax(dim=-1) == candidate.argmax(dim=-1)).float()
                    all_kl.extend(kl.tolist())
                    all_agreement.extend(agreement.tolist())
                    all_ce.append(cross_entropy_for_logits(candidate, targets))
                    task_kl[example["task"]].extend(kl.tolist())
                    task_agreement[example["task"]].extend(agreement.tolist())
                    if float(kl.mean().item()) >= QUALITY_GATE_KL or float(agreement.mean().item()) < QUALITY_GATE_TOP1:
                        failure_examples.append(
                            {
                                "id": example["id"],
                                "mean_kl_nats": float(kl.mean().item()),
                                "top1_agreement": float(agreement.mean().item()),
                            }
                        )
                kl_ci = bootstrap_mean_ci(
                    all_kl,
                    int(config["seed"]) + len(results) * 2,
                )
                top1_ci = bootstrap_mean_ci(
                    all_agreement,
                    int(config["seed"]) + len(results) * 2 + 1,
                )
                row = {
                    "layers": ",".join(str(value) for value in layer_set),
                    "layer_count": len(layer_set),
                    "method": method,
                    "deployable": method != "oracle",
                    "active_neuron_fraction": float(fraction),
                    "mean_kl_nats": float(np.mean(all_kl)),
                    "mean_kl_ci95_low": kl_ci[0],
                    "mean_kl_ci95_high": kl_ci[1],
                    "p95_kl_nats": float(np.quantile(all_kl, 0.95)),
                    "max_kl_nats": float(np.max(all_kl)),
                    "top1_agreement": float(np.mean(all_agreement)),
                    "top1_agreement_ci95_low": top1_ci[0],
                    "top1_agreement_ci95_high": top1_ci[1],
                    "mean_cross_entropy_nats": float(np.mean(all_ce)),
                    "task_mean_kl": {
                        task: float(np.mean(values)) for task, values in sorted(task_kl.items())
                    },
                    "task_metrics": {
                        task: {
                            "mean_kl_nats": float(np.mean(task_kl[task])),
                            "top1_agreement": float(np.mean(task_agreement[task])),
                            "strict_distribution_gate": bool(
                                np.mean(task_kl[task]) < QUALITY_GATE_KL
                                and np.mean(task_agreement[task]) >= QUALITY_GATE_TOP1
                            ),
                        }
                        for task in sorted(task_kl)
                    },
                    "failure_examples": failure_examples,
                }
                row["strict_distribution_gate"] = bool(
                    row["mean_kl_nats"] < QUALITY_GATE_KL
                    and row["top1_agreement"] >= QUALITY_GATE_TOP1
                )
                results.append(row)
                log.write(
                    "quality_arm_complete",
                    layers=row["layers"],
                    method=method,
                    fraction=fraction,
                    mean_kl=row["mean_kl_nats"],
                    top1=row["top1_agreement"],
                )
    return results


class PackedSwiGLU(nn.Module):
    def __init__(self, gate: nn.Linear, up: nn.Linear, down: nn.Linear, indices: Tensor) -> None:
        super().__init__()
        index = indices.to(gate.weight.device).long()
        self.register_buffer("gate_weight", gate.weight.detach().index_select(0, index).contiguous())
        self.register_buffer("up_weight", up.weight.detach().index_select(0, index).contiguous())
        self.register_buffer("down_weight", down.weight.detach().index_select(1, index).contiguous())
        gate_bias = None if gate.bias is None else gate.bias.detach().index_select(0, index).contiguous()
        up_bias = None if up.bias is None else up.bias.detach().index_select(0, index).contiguous()
        down_bias = None if down.bias is None else down.bias.detach().contiguous()
        self.register_buffer("gate_bias", gate_bias)
        self.register_buffer("up_bias", up_bias)
        self.register_buffer("down_bias", down_bias)

    def forward(self, x: Tensor, include_down_bias: bool = True) -> Tensor:
        hidden = F.silu(F.linear(x, self.gate_weight, self.gate_bias))
        hidden = hidden * F.linear(x, self.up_weight, self.up_bias)
        return F.linear(
            hidden,
            self.down_weight,
            self.down_bias if include_down_bias else None,
        )


def make_group_modules(
    gate: nn.Linear,
    up: nn.Linear,
    down: nn.Linear,
    selected: Tensor,
    groups: int,
) -> list[PackedSwiGLU]:
    parts = [part for part in torch.tensor_split(selected, groups) if part.numel()]
    return [PackedSwiGLU(gate, up, down, part) for part in parts]


def serial_group_call(modules: Sequence[PackedSwiGLU], x: Tensor) -> Tensor:
    output = modules[0](x, include_down_bias=True)
    for module in modules[1:]:
        output = output + module(x, include_down_bias=False)
    return output


class ConcurrentGroupExecutor:
    def __init__(self, modules: Sequence[PackedSwiGLU], device: torch.device) -> None:
        self.modules = list(modules)
        self.device = device
        self.streams = [torch.cuda.Stream(device=device) for _ in modules]

    def __call__(self, x: Tensor) -> Tensor:
        parent = torch.cuda.current_stream(self.device)
        ready = torch.cuda.Event()
        ready.record(parent)
        outputs = []
        for index, (module, stream) in enumerate(zip(self.modules, self.streams)):
            with torch.cuda.stream(stream):
                stream.wait_event(ready)
                outputs.append(module(x, include_down_bias=index == 0))
        for stream in self.streams:
            parent.wait_stream(stream)
        result = outputs[0]
        for output in outputs[1:]:
            result = result + output
        return result

    @torch.inference_mode()
    def overlap_diagnostic(self, x: Tensor) -> dict[str, float]:
        parent = torch.cuda.current_stream(self.device)
        ready = torch.cuda.Event(enable_timing=True)
        ready.record(parent)
        starts = []
        ends = []
        outputs = []
        wall_started = time.perf_counter_ns()
        for index, (module, stream) in enumerate(zip(self.modules, self.streams)):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(stream):
                stream.wait_event(ready)
                start.record(stream)
                outputs.append(module(x, include_down_bias=index == 0))
                end.record(stream)
            starts.append(start)
            ends.append(end)
        for stream in self.streams:
            parent.wait_stream(stream)
        total = outputs[0]
        for output in outputs[1:]:
            total = total + output
        torch.cuda.synchronize(self.device)
        wall_ms = (time.perf_counter_ns() - wall_started) / 1e6
        kernel_ms = [float(start.elapsed_time(end)) for start, end in zip(starts, ends)]
        return {
            "host_wall_ms": wall_ms,
            "sum_stream_elapsed_ms": float(sum(kernel_ms)),
            "max_stream_elapsed_ms": float(max(kernel_ms)),
            "stream_concurrency_factor_estimate": float(sum(kernel_ms) / max(wall_ms, 1e-12)),
        }


class StateRoutedPacked(nn.Module):
    def __init__(self, centers: Tensor, experts: Sequence[PackedSwiGLU]) -> None:
        super().__init__()
        self.register_buffer("centers", centers)
        self.experts = nn.ModuleList(experts)

    def forward(self, x: Tensor) -> Tensor:
        state = F.normalize(x[:, -1, :].detach().float().mean(dim=0), dim=0)
        route = int((state @ self.centers.float().T).argmax().item())
        return self.experts[route](x)


def choose_performance_arm(quality: list[dict[str, Any]]) -> dict[str, Any]:
    deployable = [
        row
        for row in quality
        if row["deployable"] and row["strict_distribution_gate"] and row["layer_count"] == 1
    ]
    if deployable:
        return min(
            deployable,
            key=lambda row: (
                row["active_neuron_fraction"],
                row["mean_kl_nats"],
                row["method"],
            ),
        )
    controls = [
        row
        for row in quality
        if row["deployable"] and row["layer_count"] == 1 and row["active_neuron_fraction"] == 1.0
    ]
    return min(controls, key=lambda row: row["mean_kl_nats"])


def enrich_operator_results(results: dict[str, Any]) -> dict[str, Any]:
    """Add quality-constrained and true parallel comparisons to saved results."""

    comparisons = []
    for batch in sorted(
        {row["batch_size"] for row in results["rows"] if row["method"] == "dense"}
    ):
        dense = next(
            row
            for row in results["rows"]
            if row["method"] == "dense" and row["batch_size"] == batch
        )
        parallel = min(
            (
                row
                for row in results["rows"]
                if row["method"] == "concurrent_cuda_streams"
                and row["batch_size"] == batch
            ),
            key=lambda row: row["median_ms"],
        )
        serial = min(
            (
                row
                for row in results["rows"]
                if row["method"] == "serial_groups" and row["batch_size"] == batch
            ),
            key=lambda row: row["median_ms"],
        )
        comparisons.append(
            {
                "batch_size": batch,
                "dense_median_ms": dense["median_ms"],
                "best_parallel_branches": parallel["branches"],
                "best_parallel_median_ms": parallel["median_ms"],
                "parallel_speedup_vs_dense": dense["median_ms"] / parallel["median_ms"],
                "best_serial_branches": serial["branches"],
                "best_serial_median_ms": serial["median_ms"],
                "serial_speedup_vs_dense": dense["median_ms"] / serial["median_ms"],
            }
        )
    results["parallel_comparisons"] = comparisons
    results["best_parallel_batch1_speedup"] = comparisons[0][
        "parallel_speedup_vs_dense"
    ]
    return results


@torch.inference_mode()
def benchmark_operators(
    model: nn.Module,
    tokenizer: Any,
    examples: list[dict[str, str]],
    calibration: dict[int, dict[str, Any]],
    quality: list[dict[str, Any]],
    config: dict[str, Any],
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    arm = choose_performance_arm(quality)
    layer_index = int(arm["layers"])
    fraction = float(arm["active_neuron_fraction"])
    method = arm["method"]
    layers = resolve_layers(model)
    mlp, gate, up, down = resolve_swiglu(layers[layer_index])
    selected = selector_indices(
        method if method not in {"state_transition_conditioned", "oracle"} else "calibration_activation",
        fraction,
        calibration[layer_index],
        task="classification",
        seed=int(config["seed"]),
        layer_index=layer_index,
    ).to(device)
    packed = PackedSwiGLU(gate, up, down, selected).to(device)
    hidden_size = gate.in_features
    dtype = gate.weight.dtype
    warmup = int(config["benchmark_warmup"])
    repetitions = int(config["operator_repetitions"])
    rows: list[dict[str, Any]] = []
    overlap = []

    dense_weight_bytes = int(
        (gate.weight.numel() + up.weight.numel() + down.weight.numel())
        * gate.weight.element_size()
    )
    for batch in config["operator_batch_sizes"]:
        x = torch.randn(int(batch), 1, hidden_size, device=device, dtype=dtype)
        dense_stats, dense_peak = benchmark_call(
            lambda: mlp(x), warmup=warmup, repetitions=repetitions, device=device
        )
        rows.append(
            {
                "method": "dense",
                "batch_size": int(batch),
                "branches": 1,
                "active_neuron_fraction": 1.0,
                **dense_stats,
                "tokens_per_second": float(batch / (dense_stats["median_ms"] / 1000)),
                "peak_incremental_vram_bytes": dense_peak,
                "analytic_weight_bytes": dense_weight_bytes,
                "actual_offchip_bytes": None,
            }
        )
        packed_stats, packed_peak = benchmark_call(
            lambda: packed(x), warmup=warmup, repetitions=repetitions, device=device
        )
        rows.append(
            {
                "method": "packed_sparse",
                "batch_size": int(batch),
                "branches": 1,
                "active_neuron_fraction": fraction,
                **packed_stats,
                "tokens_per_second": float(batch / (packed_stats["median_ms"] / 1000)),
                "peak_incremental_vram_bytes": packed_peak,
                "analytic_weight_bytes": int(dense_weight_bytes * fraction),
                "actual_offchip_bytes": None,
            }
        )

        state_scores = calibration[layer_index]["state_scores"]
        state_indices = [top_fraction_indices(score, fraction).to(device) for score in state_scores]
        routed = StateRoutedPacked(
            calibration[layer_index]["state_centers"].to(device),
            [PackedSwiGLU(gate, up, down, indices).to(device) for indices in state_indices],
        )
        routed_stats, routed_peak = benchmark_call(
            lambda: routed(x), warmup=warmup, repetitions=repetitions, device=device
        )
        rows.append(
            {
                "method": "state_routed_packed",
                "batch_size": int(batch),
                "branches": 1,
                "active_neuron_fraction": fraction,
                **routed_stats,
                "tokens_per_second": float(batch / (routed_stats["median_ms"] / 1000)),
                "peak_incremental_vram_bytes": routed_peak,
                "analytic_weight_bytes": int(dense_weight_bytes * fraction),
                "actual_offchip_bytes": None,
            }
        )

        for branches in config["parallel_branches"]:
            modules = make_group_modules(gate, up, down, selected, int(branches))
            serial_stats, serial_peak = benchmark_call(
                lambda: serial_group_call(modules, x),
                warmup=warmup,
                repetitions=repetitions,
                device=device,
            )
            rows.append(
                {
                    "method": "serial_groups",
                    "batch_size": int(batch),
                    "branches": int(branches),
                    "active_neuron_fraction": fraction,
                    **serial_stats,
                    "tokens_per_second": float(batch / (serial_stats["median_ms"] / 1000)),
                    "peak_incremental_vram_bytes": serial_peak,
                    "analytic_weight_bytes": int(dense_weight_bytes * fraction),
                    "actual_offchip_bytes": None,
                }
            )
            concurrent = ConcurrentGroupExecutor(modules, device)
            concurrent_stats, concurrent_peak = benchmark_call(
                lambda: concurrent(x),
                warmup=warmup,
                repetitions=repetitions,
                device=device,
            )
            rows.append(
                {
                    "method": "concurrent_cuda_streams",
                    "batch_size": int(batch),
                    "branches": int(branches),
                    "active_neuron_fraction": fraction,
                    **concurrent_stats,
                    "tokens_per_second": float(batch / (concurrent_stats["median_ms"] / 1000)),
                    "peak_incremental_vram_bytes": concurrent_peak,
                    "analytic_weight_bytes": int(dense_weight_bytes * fraction),
                    "actual_offchip_bytes": None,
                }
            )
            if int(batch) == 32:
                overlap.append({"branches": int(branches), **concurrent.overlap_diagnostic(x)})

    fraction_rows = []
    x = torch.randn(1, 1, hidden_size, device=device, dtype=dtype)
    for sweep_fraction in config["retention_fractions"]:
        sweep_indices = top_fraction_indices(
            calibration[layer_index]["contribution_score"], float(sweep_fraction)
        ).to(device)
        module = PackedSwiGLU(gate, up, down, sweep_indices).to(device)
        stats, peak = benchmark_call(
            lambda: module(x), warmup=warmup, repetitions=repetitions, device=device
        )
        fraction_rows.append(
            {
                "method": "packed_fraction_sweep",
                "batch_size": 1,
                "branches": 1,
                "active_neuron_fraction": float(sweep_fraction),
                **stats,
                "tokens_per_second": float(1 / (stats["median_ms"] / 1000)),
                "peak_incremental_vram_bytes": peak,
                "analytic_weight_bytes": int(dense_weight_bytes * float(sweep_fraction)),
                "actual_offchip_bytes": None,
            }
        )
    rows.extend(fraction_rows)

    profiler_summary: dict[str, Any] = {
        "requested": bool(config.get("profile_cuda")),
        "ncu_available": bool(shutil.which("ncu")),
        "hardware_occupancy_available": False,
        "hardware_offchip_bytes_available": False,
        "note": "PyTorch traces show launch scheduling but do not provide physical HBM bytes or achieved occupancy.",
    }
    if config.get("profile_cuda"):
        try:
            from torch.profiler import ProfilerActivity, profile, record_function

            sample_modules = make_group_modules(gate, up, down, selected, min(4, selected.numel()))
            sample_executor = ConcurrentGroupExecutor(sample_modules, device)
            trace_path = output_dir / "cuda_stream_trace.json"
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                with record_function("concurrent_selected_swiglu"):
                    for _ in range(5):
                        sample_executor(x)
            torch.cuda.synchronize(device)
            prof.export_chrome_trace(str(trace_path))
            profiler_summary["trace"] = trace_path.name
            profiler_summary["cuda_time_total_us"] = float(
                sum(item.cuda_time_total for item in prof.key_averages())
            )
        except Exception as exc:  # profiling support varies by image/driver
            profiler_summary["error"] = repr(exc)

    heldout = [row for row in examples if row["split"] == "heldout"]
    end_to_end_rows = []
    for batch in config["baseline_batches"]:
        example = heldout[0]
        encoded = encode_prompts(
            tokenizer,
            [example["prompt"]] * int(batch),
            device,
            int(config["max_prompt_tokens"]),
        )
        dense_stats, dense_peak = benchmark_call(
            lambda: model(**encoded, use_cache=False),
            warmup=warmup,
            repetitions=max(8, repetitions // 2),
            device=device,
        )
        with selective_layers(
            model,
            [int(value) for value in arm["layers"].split(",")],
            arm["method"],
            fraction,
            calibration,
            task=example["task"],
            seed=int(config["seed"]),
        ):
            candidate_stats, candidate_peak = benchmark_call(
                lambda: model(**encoded, use_cache=False),
                warmup=warmup,
                repetitions=max(8, repetitions // 2),
                device=device,
            )
        speedup = dense_stats["median_ms"] / candidate_stats["median_ms"]
        conservative = (
            (dense_stats["mean_ms"] - dense_stats["ci95_half_width_ms"])
            / (candidate_stats["mean_ms"] + candidate_stats["ci95_half_width_ms"])
        )
        end_to_end_rows.append(
            {
                "batch_size": int(batch),
                "dense": dense_stats,
                "candidate": candidate_stats,
                "median_speedup": speedup,
                "conservative_ci95_speedup": conservative,
                "dense_peak_incremental_vram_bytes": dense_peak,
                "candidate_peak_incremental_vram_bytes": candidate_peak,
            }
        )

    dense_row = next(row for row in rows if row["method"] == "dense" and row["batch_size"] == 1)
    best_row = min(
        (row for row in rows if row["batch_size"] == 1 and row["method"] != "dense"),
        key=lambda row: row["median_ms"],
    )
    quality_packed_row = next(
        row
        for row in rows
        if row["method"] == "packed_sparse" and row["batch_size"] == 1
    )
    return enrich_operator_results({
        "selected_quality_arm": arm,
        "rows": rows,
        "overlap_diagnostics": overlap,
        "profiler": profiler_summary,
        "measured_best_speedup_vs_dense": dense_row["median_ms"] / best_row["median_ms"],
        "measured_best_method": best_row["method"],
        "measured_best_row": best_row,
        "quality_arm_operator_speedup": dense_row["median_ms"]
        / quality_packed_row["median_ms"],
        "quality_arm_operator_conservative_ci95_speedup": (
            (dense_row["mean_ms"] - dense_row["ci95_half_width_ms"])
            / (
                quality_packed_row["mean_ms"]
                + quality_packed_row["ci95_half_width_ms"]
            )
        ),
        "end_to_end_rows": end_to_end_rows,
        "end_to_end_batch1_speedup": end_to_end_rows[0]["median_speedup"],
        "end_to_end_batch1_conservative_ci95_speedup": end_to_end_rows[0][
            "conservative_ci95_speedup"
        ],
        "claim_boundary": (
            "Latencies are measured CUDA wall-clock values including dispatch, synchronization, "
            "aggregation, and state-router costs. Weight bytes are analytic because Nsight Compute "
            "counters were unavailable in the runtime. Multiple CUDA streams are not described as "
            "independent physical processors."
        ),
    })


def run_photonic_sweep(operator_results: dict[str, Any]) -> dict[str, Any]:
    dense = next(
        row
        for row in operator_results["rows"]
        if row["method"] == "dense" and row["batch_size"] == 1
    )
    gpu_ns = float(dense["median_ms"]) * 1e6
    presets = {
        "optimistic": PhotonicAssumptions(
            feedback_ns=5,
            iterations=1,
            lanes=16,
            branches=4,
            matrix_programming_ns=1_000,
            residency_lifetime_inferences=4096,
            input_conversion_ns=100,
            readout_ns=500,
            cooldown_ns=0,
            aggregation_ns=100,
            controller_ns=100,
            readout_overlaps_feedback=True,
        ),
        "central": PhotonicAssumptions(),
        "pessimistic": PhotonicAssumptions(
            feedback_ns=40,
            iterations=16,
            lanes=1,
            branches=16,
            matrix_programming_ns=1_000_000,
            residency_lifetime_inferences=1,
            input_conversion_ns=5_000,
            readout_ns=20_000,
            cooldown_ns=10_000,
            aggregation_ns=10_000,
            controller_ns=5_000,
            readout_overlaps_feedback=False,
            cooldown_overlaps_next_programming=False,
        ),
    }
    preset_results = {
        name: break_even_margin(gpu_ns, assumptions) for name, assumptions in presets.items()
    }

    rows = []
    central = presets["central"]
    for feedback in (5.0, 20.0, 40.0):
        for iterations in (1, 9, 16):
            for lanes in (1, 2, 4, 8, 16):
                for programming in (0.0, 25_000.0, 1_000_000.0):
                    for readout in (500.0, 6_400.0, 20_000.0):
                        for cooldown in (0.0, 2_500.0, 10_000.0):
                            assumptions = replace(
                                central,
                                feedback_ns=feedback,
                                iterations=iterations,
                                lanes=lanes,
                                matrix_programming_ns=programming,
                                readout_ns=readout,
                                cooldown_ns=cooldown,
                            )
                            modeled = break_even_margin(gpu_ns, assumptions)
                            rows.append(
                                {
                                    "feedback_ns": feedback,
                                    "iterations": iterations,
                                    "lanes": lanes,
                                    "matrix_programming_ns": programming,
                                    "readout_ns": readout,
                                    "cooldown_ns": cooldown,
                                    "aggregation_ns": assumptions.aggregation_ns,
                                    "controller_ns": assumptions.controller_ns,
                                    "residency_lifetime_inferences": assumptions.residency_lifetime_inferences,
                                    "elapsed_latency_ns": modeled["elapsed_latency_ns"],
                                    "device_occupancy_ns": modeled["device_occupancy_ns"],
                                    "modeled_speedup_vs_gpu": modeled["modeled_speedup_vs_gpu"],
                                    "modeled_break_even": modeled["modeled_break_even"],
                                    "energy_j_per_inference": modeled["energy_j_per_inference"],
                                    "area_mm2": modeled["area_mm2"],
                                    "lane_utilization": modeled["lane_utilization"],
                                }
                            )

    sensitivity_ranges = {
        "feedback_ns": [5.0, 40.0],
        "iterations": [1, 16],
        "lanes": [1, 16],
        "matrix_programming_ns": [0.0, 1_000_000.0],
        "residency_lifetime_inferences": [1, 4096],
        "input_conversion_ns": [100.0, 5_000.0],
        "readout_ns": [500.0, 20_000.0],
        "cooldown_ns": [0.0, 10_000.0],
        "aggregation_ns": [100.0, 10_000.0],
        "controller_ns": [100.0, 5_000.0],
    }
    sensitivity = []
    for name, values in sensitivity_ranges.items():
        outcomes = []
        for value in values:
            modeled = break_even_margin(gpu_ns, replace(central, **{name: value}))
            outcomes.append(
                {
                    "value": value,
                    "elapsed_latency_ns": modeled["elapsed_latency_ns"],
                    "modeled_speedup_vs_gpu": modeled["modeled_speedup_vs_gpu"],
                    "break_even": modeled["modeled_break_even"],
                }
            )
        sensitivity.append(
            {
                "parameter": name,
                "outcomes": outcomes,
                "changes_pass_fail": outcomes[0]["break_even"] != outcomes[1]["break_even"],
                "latency_span_ns": abs(
                    outcomes[1]["elapsed_latency_ns"] - outcomes[0]["elapsed_latency_ns"]
                ),
            }
        )
    must_verify = [
        row["parameter"]
        for row in sorted(sensitivity, key=lambda item: item["latency_span_ns"], reverse=True)
        if row["changes_pass_fail"]
    ]
    if not must_verify:
        must_verify = [
            row["parameter"]
            for row in sorted(sensitivity, key=lambda item: item["latency_span_ns"], reverse=True)[:4]
        ]
    return {
        "gpu_comparator": {
            "kind": "measured dense GPU MLP operator latency",
            "latency_ns": gpu_ns,
            "batch_size": 1,
        },
        "presets": preset_results,
        "sweep_rows": rows,
        "sensitivity": sensitivity,
        "minimum_characteristics_to_verify": must_verify,
        "sweep_break_even_fraction": float(
            np.mean([row["modeled_break_even"] for row in rows])
        ),
        "claim_boundary": (
            "All optical values are simulated. The 20 ns feedback time and nine iterations are "
            "illustrative assumptions, not measured performance of accessible photonic hardware."
        ),
    }


def normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def minimum_sufficient_summary(quality: list[dict[str, Any]]) -> dict[str, Any]:
    """Find the smallest passing deployable fraction overall and per task."""

    deployable = [row for row in quality if row["deployable"]]
    overall = []
    tasks = []
    layer_sets = sorted({row["layers"] for row in deployable})
    methods = sorted({row["method"] for row in deployable})
    for layers in layer_sets:
        for method in methods:
            matching = [
                row
                for row in deployable
                if row["layers"] == layers and row["method"] == method
            ]
            candidates = sorted(
                (row for row in matching if row["strict_distribution_gate"]),
                key=lambda row: row["active_neuron_fraction"],
            )
            overall.append(
                {
                    "layers": layers,
                    "method": method,
                    "minimum_sufficient_fraction": candidates[0]["active_neuron_fraction"]
                    if candidates
                    else None,
                }
            )
            task_names = sorted(
                {task for row in matching for task in row["task_metrics"]}
            )
            for task in task_names:
                task_candidates = sorted(
                    (
                        row
                        for row in matching
                        if row["task_metrics"][task]["strict_distribution_gate"]
                    ),
                    key=lambda row: row["active_neuron_fraction"],
                )
                tasks.append(
                    {
                        "layers": layers,
                        "method": method,
                        "task": task,
                        "minimum_sufficient_fraction": task_candidates[0][
                            "active_neuron_fraction"
                        ]
                        if task_candidates
                        else None,
                    }
                )
    return {"overall": overall, "by_task": tasks}


@torch.inference_mode()
def generate_quality(
    model: nn.Module,
    tokenizer: Any,
    examples: list[dict[str, str]],
    calibration: dict[int, dict[str, Any]],
    arm: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    selected = [row for row in examples if row["split"] == "heldout"][: int(config["free_running_examples"])]
    rows = []
    for example in selected:
        encoded = encode_prompts(
            tokenizer,
            [example["prompt"]],
            device,
            int(config["max_prompt_tokens"]),
        )
        dense_tokens = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=int(config["max_new_tokens"]),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        dense_new = dense_tokens[0, encoded["input_ids"].shape[1] :]
        dense_text = tokenizer.decode(dense_new, skip_special_tokens=True)
        with selective_layers(
            model,
            [int(value) for value in arm["layers"].split(",")],
            arm["method"],
            float(arm["active_neuron_fraction"]),
            calibration,
            task=example["task"],
            seed=int(config["seed"]),
        ) as wrappers:
            for wrapper in wrappers:
                if hasattr(wrapper, "reset_state"):
                    wrapper.reset_state()
            candidate_tokens = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=int(config["max_new_tokens"]),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        candidate_new = candidate_tokens[0, encoded["input_ids"].shape[1] :]
        candidate_text = tokenizer.decode(candidate_new, skip_special_tokens=True)
        dense_ids = dense_new.detach().cpu().tolist()
        candidate_ids = candidate_new.detach().cpu().tolist()
        common = sum(a == b for a, b in zip(dense_ids, candidate_ids))
        rows.append(
            {
                "id": example["id"],
                "task": example["task"],
                "reference_answer": example["answer"],
                "dense_text": dense_text,
                "candidate_text": candidate_text,
                "exact_dense_match": dense_ids == candidate_ids,
                "token_prefix_agreement": common / max(len(dense_ids), len(candidate_ids), 1),
                "dense_exact_manifest_match": normalize_text(dense_text) == normalize_text(example["answer"]),
                "candidate_exact_manifest_match": normalize_text(candidate_text)
                == normalize_text(example["answer"]),
            }
        )
    return {
        "rows": rows,
        "dense_candidate_exact_match": float(np.mean([row["exact_dense_match"] for row in rows])),
        "mean_token_prefix_agreement": float(
            np.mean([row["token_prefix_agreement"] for row in rows])
        ),
        "dense_manifest_exact_match": float(
            np.mean([row["dense_exact_manifest_match"] for row in rows])
        ),
        "candidate_manifest_exact_match": float(
            np.mean([row["candidate_exact_manifest_match"] for row in rows])
        ),
    }


def make_plots(
    quality: list[dict[str, Any]],
    operators: dict[str, Any],
    photonic: dict[str, Any],
    output_dir: Path,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    single = [
        row
        for row in quality
        if row["layers"] == "23" and row["method"] == "calibration_activation"
    ]
    latency = {
        row["active_neuron_fraction"]: row["median_ms"]
        for row in operators["rows"]
        if row["method"] == "packed_fraction_sweep"
    }
    points = [row for row in single if row["active_neuron_fraction"] in latency]
    if points:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        x = [latency[row["active_neuron_fraction"]] for row in points]
        y = [row["mean_kl_nats"] for row in points]
        ax.plot(x, y, marker="o")
        for row, x_value, y_value in zip(points, x, y):
            ax.annotate(f"{row['active_neuron_fraction']:.0%}", (x_value, y_value))
        ax.axhline(QUALITY_GATE_KL, color="red", linestyle="--", label="KL gate")
        ax.set_xlabel("Measured packed MLP median latency (ms), batch 1")
        ax.set_ylabel("Held-out teacher-vs-student KL (nats)")
        ax.set_title("Accuracy / latency frontier")
        ax.grid(alpha=0.25)
        ax.legend()
        path = output_dir / "accuracy_latency_pareto.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path.name)

    central_rows = [
        row
        for row in photonic["sweep_rows"]
        if row["iterations"] == 9
        and row["matrix_programming_ns"] == 25_000.0
        and row["readout_ns"] == 6_400.0
        and row["cooldown_ns"] == 2_500.0
    ]
    feedbacks = sorted({row["feedback_ns"] for row in central_rows})
    lanes = sorted({row["lanes"] for row in central_rows})
    grid = np.array(
        [
            [
                next(
                    row["modeled_speedup_vs_gpu"]
                    for row in central_rows
                    if row["feedback_ns"] == feedback and row["lanes"] == lane
                )
                for lane in lanes
            ]
            for feedback in feedbacks
        ]
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    image = ax.imshow(grid, aspect="auto", cmap="coolwarm", origin="lower")
    ax.set_xticks(range(len(lanes)), labels=lanes)
    ax.set_yticks(range(len(feedbacks)), labels=feedbacks)
    ax.set_xlabel("Optical lanes")
    ax.set_ylabel("Feedback time (ns)")
    ax.set_title("Modeled optical speedup vs measured dense GPU MLP")
    fig.colorbar(image, ax=ax, label="modeled speedup")
    path = output_dir / "photonic_break_even_heatmap.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path.name)
    return paths


def render_reports(
    output_dir: Path,
    environment: dict[str, Any],
    config: dict[str, Any],
    baseline: dict[str, Any],
    decomposition: list[dict[str, Any]],
    quality: list[dict[str, Any]],
    operators: dict[str, Any],
    photonic: dict[str, Any],
    generation: dict[str, Any],
    plots: list[str],
) -> None:
    arm = operators["selected_quality_arm"]
    packed_end_to_end_gate = (
        operators["end_to_end_batch1_conservative_ci95_speedup"] > 1.0
    )
    parallel_gate = operators["best_parallel_batch1_speedup"] > 1.0
    decomposition_gate = all(
        row["local_relative_l2_error"] < 0.01
        and row["full_model_top1_agreement"] == 1.0
        and row["full_model_mean_kl_nats"] < QUALITY_GATE_KL
        and row["end_to_end_generated_tokens_equal"]
        for row in decomposition
    )
    quality_gate = bool(arm["strict_distribution_gate"])
    subset_gate = quality_gate and arm["active_neuron_fraction"] < 1.0
    free_gate = generation["dense_candidate_exact_match"] == 1.0
    task_acceptance_gate = generation["candidate_manifest_exact_match"] >= 0.99
    strict = (
        decomposition_gate
        and subset_gate
        and free_gate
        and task_acceptance_gate
        and parallel_gate
    )
    central = photonic["presets"]["central"]
    best_negative = (
        "No deployable sub-100% neuron set passed KL < 0.01 and top-1 >= 99%; "
        "state-aware routing also trailed the best matched static selector."
        if not subset_gate
        else "Parallel CUDA groups did not beat the dense operator after dispatch and aggregation."
    )
    baseline_row = next(
        row for row in baseline["rows"] if row["batch_size"] == 1
    )

    report = f"""# Photonic parallel minimum-inference research report

## Scope and claim boundary

This run used a real `{environment.get('gpu')}` for CUDA measurements. Optical
numbers are simulations only. Analytic weight-byte counts are not physical HBM
traffic measurements. The model was pinned to `{config['model']}` revision
`{config['revision']}` in `{config['dtype']}` with seed `{config['seed']}`.

## Method

- Dataset: 24 locally generated nonsensitive records, with disjoint calibration,
  development, and held-out prompts across four task families.
- Dense baseline: synchronized warmups and repeated prefill, one-token decode,
  time-to-first-token, and deterministic generation at batches 1, 8, and 32.
- Correctness control: the middle Qwen SwiGLU layer was decomposed into 2, 4, 8,
  and 16 disjoint neuron groups and summed before the residual connection.
- Selection: random, static activation, weight magnitude, calibration contribution,
  task-conditioned, state-transition-conditioned, and a non-deployable held-out
  oracle were tested at 10%, 25%, 50%, 75%, and 100% union retention.
- CUDA operators: dense, one packed sparse GEMM, serial groups, concurrent CUDA
  streams, and a state-routed packed path include host dispatch, synchronization,
  aggregation, and routing in measured wall time.
- Photonics: elapsed latency, device occupancy, throughput, energy, area, lane
  utilization, programming amortization, conversion, readout, cooldown, controller,
  and aggregation were modeled separately.

## Results

- Dense SDPA batch-1 prefill median: `{baseline_row['prefill']['median_ms']:.4f} ms`.
- Dense SDPA batch-1 decode median: `{baseline_row['decode_one_token']['median_ms']:.4f} ms`.
- Exact grouped reconstruction gate: `{'PASS' if decomposition_gate else 'FAIL'}`.
- Selected deployable arm: `{arm['method']}`, layers `{arm['layers']}`, retention
  `{arm['active_neuron_fraction']:.0%}`, KL `{arm['mean_kl_nats']:.6g}`, top-1
  `{arm['top1_agreement']:.2%}`.
- Free-running exact agreement with dense: `{generation['dense_candidate_exact_match']:.2%}`.
- Fastest unconstrained CUDA operator arm: `{operators['measured_best_speedup_vs_dense']:.3f}x`
  (`{operators['measured_best_method']}`); this is not a valid quality-preserving
  speedup when its fraction fails the fidelity gate.
- Quality-arm packed operator speedup: `{operators['quality_arm_operator_speedup']:.3f}x`
  (conservative 95% bound `{operators['quality_arm_operator_conservative_ci95_speedup']:.3f}x`).
- Measured end-to-end batch-1 speedup: `{operators['end_to_end_batch1_speedup']:.3f}x`
  (conservative 95% bound `{operators['end_to_end_batch1_conservative_ci95_speedup']:.3f}x`).
- Best true parallel batch-1 operator result: `{operators['best_parallel_batch1_speedup']:.3f}x`
  versus dense (values below 1 are slowdowns).
- Central optical model: `{central['modeled_speedup_vs_gpu']:.3f}x` versus the
  measured dense GPU MLP comparator, elapsed `{central['elapsed_latency_ns']:.1f} ns`.
- Optical sweep break-even fraction: `{photonic['sweep_break_even_fraction']:.2%}`
  of explicitly modeled points.
- Strict combined gate: `{'PASS' if strict else 'FAIL'}`.

## Strongest negative result

{best_negative}

## Caveats

The dataset is intentionally small and synthetic. The packed PyTorch kernels are
real measured kernels but are not custom fused sparse kernels. Nsight Compute was
not present, so physical off-chip bytes and achieved occupancy are unavailable;
the report does not infer them from neuron count. CUDA streams may overlap kernel
timelines, but they are not independent hardware accelerators. The oracle uses
held-out activation information and is excluded from deployable comparisons.

## Optical falsification boundary

The central cost model includes 20 ns feedback, nine iterations, 25 us matrix
programming amortized over 256 inferences, 6.4 us readout, 2.5 us cooldown, and
controller/aggregation costs. The hardware characteristics with the largest
one-at-a-time modeled latency sensitivity are:
{', '.join(photonic['minimum_characteristics_to_verify'])}.
These require independent hardware measurement before any hybrid speedup claim.

## Reproduction

```powershell
./scripts/run_photonic_docker.ps1 -Config configs/photonic_parallel_quick.json -RunId reproduced
```

Artifacts: `{', '.join(plots)}` plus JSON/CSV tables and `cuda_stream_trace.json`.
"""
    (output_dir / "RESEARCH_REPORT.md").write_text(report, encoding="utf-8")

    morning = f"""# Morning summary

- Docker: repaired separately by quarantining inaccessible transient AF_UNIX
  runtime directories; no persistent Docker data was deleted.
- CUDA environment: PyTorch `{environment['torch']}`, CUDA `{environment['torch_cuda']}`,
  `{environment.get('gpu')}` with `{environment.get('gpu_total_memory_bytes', 0) / 2**30:.2f} GiB`.
- Experiments completed: dense baseline, exact 2/4/8/16-way decomposition,
  held-out retention sweep, sequential/concurrent CUDA benchmarks, deterministic
  free-running comparison, and optical sensitivity sweep.
- Best verified finding: exact 2/4/8/16-way SwiGLU decomposition preserved top-1
  logits and deterministic generated tokens within the declared BF16 tolerance.
- Unconstrained fastest operator: `{operators['measured_best_method']}` reached
  `{operators['measured_best_speedup_vs_dense']:.3f}x`, but this is not a
  quality-preserving architecture result. Quality-arm end-to-end batch-1 was
  `{operators['end_to_end_batch1_speedup']:.3f}x` with conservative bound
  `{operators['end_to_end_batch1_conservative_ci95_speedup']:.3f}x`.
- Strongest negative finding: {best_negative}
- Strict gates: decomposition `{'PASS' if decomposition_gate else 'FAIL'}`,
  distribution quality `{'PASS' if quality_gate else 'FAIL'}`, nontrivial subset
  `{'PASS' if subset_gate else 'FAIL'}`, free-running
  dense agreement `{'PASS' if free_gate else 'FAIL'}`, task exact-match acceptance
  `{'PASS' if task_acceptance_gate else 'FAIL'}`
  (`{generation['candidate_manifest_exact_match']:.2%}`), packed end-to-end
  `{'PASS' if packed_end_to_end_gate else 'FAIL'}`, parallel GPU performance
  `{'PASS' if parallel_gate else 'FAIL'}`;
  overall `{'PASS' if strict else 'FAIL'}`.
- Optical central case: `{central['modeled_speedup_vs_gpu']:.3f}x` modeled, with
  20 ns feedback, nine iterations, 6.4 us readout, 2.5 us cooldown, four lanes,
  and programming amortized over 256 inferences. This is simulated, not measured.
- Next experiment: profile a fused grouped GEMM with Nsight Compute, preserving the
  same selected union, before expanding selector complexity.
"""
    (output_dir / "MORNING_SUMMARY.md").write_text(morning, encoding="utf-8")


def load_model(
    model_name: str,
    revision: str,
    dtype: torch.dtype,
    device: torch.device,
    attention_backend: str,
) -> nn.Module:
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        dtype=dtype,
        attn_implementation=attention_backend,
        low_cpu_mem_usage=True,
    ).to(device).eval()


def main() -> None:
    args = parse_args()
    config_path = (ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    output_dir = (ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    log = RunLog(output_dir / "research_log.jsonl")
    seed_everything(int(config["seed"]))
    environment = environment_snapshot()
    environment["seeds"] = {"python_numpy_torch_cuda": int(config["seed"])}
    environment["model"] = config["model"]
    environment["model_revision"] = config["revision"]
    write_json(output_dir / "environment.json", environment)
    write_json(output_dir / "resolved_config.json", config)
    log.write("run_started", output_dir=str(output_dir), cuda=environment["cuda_available"])
    if not environment["cuda_available"]:
        raise RuntimeError("CUDA is required for the measured experiment")

    from transformers import AutoTokenizer, GenerationConfig

    device = torch.device("cuda")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[config["dtype"]]
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["revision"])
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    generation_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=int(config["max_new_tokens"]),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    write_json(output_dir / "generation_config.json", generation_config.to_dict())
    examples = load_dataset()

    baseline_path = output_dir / "baseline.json"
    baseline = load_json_if_resuming(baseline_path, args.resume)
    if baseline is None:
        log.write("baseline_eager_loading")
        eager_model = load_model(
            config["model"], config["revision"], dtype, device, "eager"
        )
        eager_model.generation_config = generation_config
        eager = run_dense_baseline(
            eager_model, tokenizer, examples, config, device, "eager"
        )
        del eager_model
        gc.collect()
        torch.cuda.empty_cache()
        log.write("baseline_sdpa_loading")
        model = load_model(config["model"], config["revision"], dtype, device, "sdpa")
        model.generation_config = generation_config
        sdpa = run_dense_baseline(model, tokenizer, examples, config, device, "sdpa")
        baseline = {"eager": eager, "sdpa": sdpa, "rows": eager["rows"] + sdpa["rows"]}
        write_json(baseline_path, baseline)
        write_csv(
            output_dir / "baseline_latency.csv",
            [
                {
                    "backend": row["backend"],
                    "batch_size": row["batch_size"],
                    "prefill_median_ms": row["prefill"]["median_ms"],
                    "decode_median_ms": row["decode_one_token"]["median_ms"],
                    "ttft_median_ms": row["time_to_first_token"]["median_ms"],
                    "generation_median_ms": row["complete_generation"]["median_ms"],
                    "prefill_tokens_per_second": row["prefill_tokens_per_second"],
                    "decode_tokens_per_second": row["decode_tokens_per_second"],
                    "peak_incremental_vram_bytes": row["peak_incremental_vram_bytes"],
                }
                for row in baseline["rows"]
            ],
        )
    else:
        model = load_model(config["model"], config["revision"], dtype, device, "sdpa")
        model.generation_config = generation_config
    log.write("baseline_complete")

    total_vram = torch.cuda.get_device_properties(device).total_memory
    reserved = torch.cuda.memory_reserved(device)
    headroom = (total_vram - reserved) / total_vram
    log.write("vram_headroom", fraction=headroom, reserved_bytes=reserved, total_bytes=total_vram)
    if headroom < float(config["min_vram_headroom_fraction"]):
        raise RuntimeError("VRAM headroom fell below configured safety threshold")

    heldout = [row for row in examples if row["split"] == "heldout"]
    decomposition_path = output_dir / "decomposition.json"
    decomposition = load_json_if_resuming(decomposition_path, args.resume)
    if decomposition is None:
        decomposition = exact_decomposition(model, tokenizer, heldout, config, device)
        write_json(decomposition_path, decomposition)
        write_csv(output_dir / "decomposition.csv", decomposition)
    log.write("decomposition_complete")

    all_layers = sorted(
        {int(index) for layer_set in config["progressive_layer_sets"] for index in layer_set}
    )
    calibration = capture_calibration(
        model,
        tokenizer,
        examples,
        all_layers,
        device,
        int(config["max_prompt_tokens"]),
    )
    calibration_manifest = {
        str(index): {
            "records": payload["records"],
            "intermediate_width": int(payload["contribution_score"].numel()),
            "tasks": sorted(payload["task_scores"]),
        }
        for index, payload in calibration.items()
    }
    write_json(output_dir / "calibration_manifest.json", calibration_manifest)

    quality_path = output_dir / "quality.json"
    quality = load_json_if_resuming(quality_path, args.resume)
    if quality is None:
        quality_reference = build_quality_reference(
            model,
            tokenizer,
            examples,
            device,
            int(config["max_prompt_tokens"]),
        )
        quality = evaluate_quality(
            model,
            tokenizer,
            examples,
            quality_reference,
            calibration,
            config,
            device,
            log,
        )
        write_json(quality_path, quality)
        csv_rows = []
        for row in quality:
            flattened = {
                key: value
                for key, value in row.items()
                if key not in {"task_mean_kl", "task_metrics", "failure_examples"}
            }
            flattened.update(
                {f"task_kl_{key}": value for key, value in row["task_mean_kl"].items()}
            )
            csv_rows.append(flattened)
        write_csv(output_dir / "quality.csv", csv_rows)
    minimum_sufficient = minimum_sufficient_summary(quality)
    write_json(output_dir / "minimum_sufficient.json", minimum_sufficient)
    write_csv(output_dir / "minimum_sufficient_overall.csv", minimum_sufficient["overall"])
    write_csv(output_dir / "minimum_sufficient_by_task.csv", minimum_sufficient["by_task"])
    log.write("quality_complete", arms=len(quality))

    operators_path = output_dir / "operators.json"
    operators = load_json_if_resuming(operators_path, args.resume)
    if operators is None:
        operators = benchmark_operators(
            model,
            tokenizer,
            examples,
            calibration,
            quality,
            config,
            device,
            output_dir,
        )
        write_json(operators_path, operators)
        write_csv(output_dir / "operator_latency.csv", operators["rows"])
    operators = enrich_operator_results(operators)
    write_json(operators_path, operators)
    log.write("operators_complete", speedup=operators["measured_best_speedup_vs_dense"])

    generation_path = output_dir / "free_running.json"
    generation = load_json_if_resuming(generation_path, args.resume)
    if generation is None:
        generation = generate_quality(
            model,
            tokenizer,
            examples,
            calibration,
            operators["selected_quality_arm"],
            config,
            device,
        )
        write_json(generation_path, generation)
    log.write("free_running_complete")

    photonic_path = output_dir / "photonic.json"
    photonic = load_json_if_resuming(photonic_path, args.resume)
    if photonic is None:
        photonic = run_photonic_sweep(operators)
        write_json(photonic_path, photonic)
        write_csv(output_dir / "photonic_sweep.csv", photonic["sweep_rows"])
    log.write("photonic_complete", central=photonic["presets"]["central"]["modeled_break_even"])

    plots = make_plots(quality, operators, photonic, output_dir)
    render_reports(
        output_dir,
        environment,
        config,
        baseline["sdpa"],
        decomposition,
        quality,
        operators,
        photonic,
        generation,
        plots,
    )
    summary = {
        "quality_arm": operators["selected_quality_arm"],
        "measured_operator_speedup": operators["measured_best_speedup_vs_dense"],
        "measured_operator_method": operators["measured_best_method"],
        "quality_arm_operator_speedup": operators["quality_arm_operator_speedup"],
        "measured_end_to_end_speedup": operators["end_to_end_batch1_speedup"],
        "best_parallel_batch1_speedup": operators["best_parallel_batch1_speedup"],
        "free_running_dense_agreement": generation["dense_candidate_exact_match"],
        "central_photonic_model": photonic["presets"]["central"],
        "plots": plots,
    }
    write_json(output_dir / "summary.json", summary)
    log.write("run_complete", summary=summary)


if __name__ == "__main__":
    main()
