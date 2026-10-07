from __future__ import annotations

import argparse
import json
import random

import torch
from torch import nn

from depth_router import DepthRouterConfig, DepthRouterModel, StackedTransformerBaseline
from depth_router.cost import parameter_bytes, parameter_count
from depth_router.toy import PointerChaseSpec, pointer_chase_batch


def build_model(name: str, config: DepthRouterConfig) -> nn.Module:
    if name == "stacked":
        return StackedTransformerBaseline(config)
    if name == "shared":
        return DepthRouterModel(config, use_pass_adapters=False)
    if name == "shared-adapter":
        return DepthRouterModel(config, use_pass_adapters=True)
    raise ValueError(f"unknown model: {name}")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    spec: PointerChaseSpec,
    *,
    batches: int,
    batch_size: int,
    adaptive: bool,
    halt_threshold: float,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    correct = 0
    examples = 0
    logical_passes = 0.0

    for _ in range(batches):
        x, y, _ = pointer_chase_batch(batch_size, spec, device=device)
        if isinstance(model, DepthRouterModel):
            logits, stats = model(
                x,
                adaptive=adaptive,
                halt_threshold=halt_threshold,
                return_stats=True,
            )
        else:
            logits, stats = model(x, return_stats=True)

        correct += int((logits.argmax(dim=-1) == y).sum().item())
        examples += batch_size
        logical_passes += float(stats.sample_passes.sum().item())

    return {
        "accuracy": correct / examples,
        "mean_logical_passes": logical_passes / examples,
    }


def train_one(args: argparse.Namespace, name: str, device: torch.device) -> dict:
    spec = PointerChaseSpec(nodes=args.nodes, max_hops=args.max_hops)
    config = DepthRouterConfig(
        vocab_size=spec.vocab_size,
        max_seq_len=spec.seq_len,
        num_classes=spec.nodes,
        d_model=args.d_model,
        nhead=args.nhead,
        dim_feedforward=args.ffn,
        dropout=0.0,
        max_steps=args.max_steps,
        adapter_rank=args.adapter_rank,
    )
    model = build_model(name, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for _ in range(args.train_steps):
        x, y, _ = pointer_chase_batch(args.batch_size, spec, device=device)
        optimizer.zero_grad(set_to_none=True)

        if isinstance(model, DepthRouterModel):
            logits = model(x, adaptive=False)
        else:
            logits = model(x)

        loss = nn.functional.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

    fixed = evaluate(
        model,
        spec,
        batches=args.eval_batches,
        batch_size=args.batch_size,
        adaptive=False,
        halt_threshold=args.halt_threshold,
        device=device,
    )
    result = {
        "model": name,
        "parameters": parameter_count(model),
        "parameter_bytes": parameter_bytes(model),
        "fixed": fixed,
    }

    if isinstance(model, DepthRouterModel):
        result["adaptive"] = evaluate(
            model,
            spec,
            batches=args.eval_batches,
            batch_size=args.batch_size,
            adaptive=True,
            halt_threshold=args.halt_threshold,
            device=device,
        )

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["stacked", "shared", "shared-adapter"],
        choices=["stacked", "shared", "shared-adapter"],
    )
    parser.add_argument("--nodes", type=int, default=8)
    parser.add_argument("--max-hops", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--ffn", type=int, default=256)
    parser.add_argument("--adapter-rank", type=int, default=8)
    parser.add_argument("--train-steps", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--halt-threshold", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    results = [train_one(args, name, device) for name in args.models]
    payload = {
        "experiment": "pointer_chase_v0",
        "device": str(device),
        "config": {
            key: value
            for key, value in vars(args).items()
            if key not in {"models", "device"}
        },
        "results": results,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
