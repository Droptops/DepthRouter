"""Collect dense Qwen MLP demand traces for enterprise workloads.

This is a *diagnostic dense-model oracle*; replay is not deployable sparse
execution and cannot measure physical HBM savings. A lagged hidden-state delta
is used as a causal proxy signal for the *next* step, not a current-step oracle.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import time
from pathlib import Path

import torch
from enterprise_tasks import examples


def atomic_write(path, text):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(destination)


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return torch.device(requested)


def resolve_dtype(requested, device):
    if requested == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[requested]
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("FP16 CPU execution is unsupported for this experiment")
    return dtype


def percentile(values, fraction):
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--revision")
    p.add_argument("--layer", type=int, default=23)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--max-length", type=int, default=192)
    p.add_argument("--examples-per-task", type=int, default=100)
    p.add_argument("--page-units", type=int, default=128)
    p.add_argument("--active-fraction", type=float, default=0.10)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--dtype", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-out", required=True)
    p.add_argument("--test-out", required=True)
    p.add_argument("--metadata-out", required=True)
    a = p.parse_args()
    if (a.steps < 2 or a.examples_per_task < 8 or a.page_units < 1
            or not 0 < a.active_fraction <= 1):
        p.error("invalid steps/page-units/active-fraction")
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = resolve_device(a.device)
    dtype = resolve_dtype(a.dtype, device)
    torch.manual_seed(a.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(a.seed)
        torch.backends.cuda.matmul.allow_tf32 = False

    tokenizer = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    model_kwargs = {"revision": a.revision}
    if int(transformers.__version__.split(".")[0]) >= 5:
        model_kwargs["dtype"] = dtype
    else:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(a.model, **model_kwargs).to(device).eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    layers = model.model.layers
    if a.layer < 0 or a.layer >= len(layers):
        p.error("layer index out of range")
    mlp = layers[a.layer].mlp
    weight_norm = mlp.down_proj.weight.detach().float().norm(dim=0)
    width = weight_norm.numel()
    train, test = [], []
    forward_latencies_ms = []
    for item in examples(a.examples_per_task):
        ids = tokenizer(item["text"], return_tensors="pt", truncation=False)["input_ids"].to(
            device
        )
        if ids.shape[1] > a.max_length:
            raise ValueError(f"prompt truncated: {item['sequence']}")
        prev_hidden = None
        lagged_signal = 0.0
        for step in range(a.steps):
            observed = {}

            def hidden_hook(_module, args, observations=observed):
                observations["hidden"] = args[0][:, -1, :].detach().float().clone()

            def activation_hook(_module, args, observations=observed):
                observations["activation"] = args[0][:, -1, :].detach().float().clone()

            h1 = mlp.register_forward_pre_hook(hidden_hook)
            h2 = mlp.down_proj.register_forward_pre_hook(activation_hook)
            try:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                logits = model(input_ids=ids).logits[:, -1, :]
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                forward_latencies_ms.append((time.perf_counter() - started) * 1000)
            finally:
                h1.remove()
                h2.remove()
            score = observed["activation"].abs().flatten() * weight_norm
            k = max(1, round(width * a.active_fraction))
            neurons = torch.topk(score, k=k, sorted=False).indices.tolist()
            pages = sorted({index // a.page_units for index in neurons})
            row = {"task": item["task"], "sequence": item["sequence"],
                   "step": step, "pages": pages,
                   "transition_signal": lagged_signal}
            # Deterministic prompt-disjoint split: 70 train / 30 test per task.
            (train if int(item["sequence"].split("-")[-1]) % 10 < 7 else test).append(row)
            hidden = observed["hidden"]
            if prev_hidden is not None:
                lagged_signal = float(torch.linalg.vector_norm(hidden - prev_hidden) /
                                      torch.linalg.vector_norm(prev_hidden).clamp_min(1e-8))
            prev_hidden = hidden
            ids = torch.cat((ids, logits.argmax(dim=-1, keepdim=True)), dim=1)
    for path, rows in ((a.train_out, train), (a.test_out, test)):
        atomic_write(path, "".join(json.dumps(row) + "\n" for row in rows))
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    steady_latencies_ms = forward_latencies_ms[1:] or forward_latencies_ms
    meta = {"model": a.model, "layer": a.layer, "steps": a.steps,
            "examples_per_task": a.examples_per_task, "page_units": a.page_units,
            "active_fraction": a.active_fraction, "width": width,
            "train_rows": len(train), "test_rows": len(test),
            "requested_revision": a.revision,
            "model_revision": getattr(model.config, "_commit_hash", None) or a.revision,
            "tokenizer_revision": tokenizer.init_kwargs.get("_commit_hash") or a.revision,
            "git_commit": git_commit, "seed": a.seed,
            "device": str(device), "dtype": str(dtype),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda,
            "transformers_version": transformers.__version__,
            "peak_gpu_memory_bytes": (torch.cuda.max_memory_allocated(device)
                                      if device.type == "cuda" else None),
            "dense_forward_latency_ms_cold_start": forward_latencies_ms[0],
            "dense_forward_latency_ms_mean": statistics.fmean(steady_latencies_ms),
            "dense_forward_latency_ms_p50": percentile(steady_latencies_ms, 0.50),
            "dense_forward_latency_ms_p95": percentile(steady_latencies_ms, 0.95),
            "split": "prompt-disjoint index_mod_10: train=0..6, test=7..9",
            "signal": "one-step-lagged hidden delta, dense-model diagnostic",
            "physical_bytes_per_token": None, "mean_kl": None,
            "top1_agreement": None, "free_running_accuracy": None}
    atomic_write(a.metadata_out, json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
