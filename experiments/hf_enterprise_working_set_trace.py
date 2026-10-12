"""Collect dense Qwen MLP demand traces for enterprise workloads.

This is a *diagnostic dense-model oracle*; replay is not deployable sparse
execution and cannot measure physical HBM savings. A lagged hidden-state delta
is used as a causal proxy signal for the *next* step, not a current-step oracle.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from enterprise_tasks import examples


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--layer", type=int, default=23)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--max-length", type=int, default=192)
    p.add_argument("--examples-per-task", type=int, default=100)
    p.add_argument("--page-units", type=int, default=128)
    p.add_argument("--active-fraction", type=float, default=0.10)
    p.add_argument("--train-out", required=True)
    p.add_argument("--test-out", required=True)
    p.add_argument("--metadata-out", required=True)
    a = p.parse_args()
    if a.steps < 2 or a.page_units < 1 or not 0 < a.active_fraction <= 1:
        p.error("invalid steps/page-units/active-fraction")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).eval()
    layers = model.model.layers
    if a.layer < 0 or a.layer >= len(layers):
        p.error("layer index out of range")
    mlp = layers[a.layer].mlp
    weight_norm = mlp.down_proj.weight.detach().float().norm(dim=0)
    width = weight_norm.numel()
    train, test = [], []
    for item in examples(a.examples_per_task):
        ids = tokenizer(item["text"], return_tensors="pt", truncation=False)["input_ids"]
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
                logits = model(input_ids=ids).logits[:, -1, :]
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
        Path(path).write_text("".join(json.dumps(row) + "\n" for row in rows))
    meta = {"model": a.model, "layer": a.layer, "steps": a.steps,
            "examples_per_task": a.examples_per_task, "page_units": a.page_units,
            "active_fraction": a.active_fraction, "width": width,
            "train_rows": len(train), "test_rows": len(test),
            "signal": "one-step-lagged hidden delta, dense-model diagnostic",
            "physical_bytes_per_token": None, "mean_kl": None,
            "top1_agreement": None, "free_running_accuracy": None}
    Path(a.metadata_out).write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
