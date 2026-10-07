from __future__ import annotations

import argparse
import json
import random

import torch
from torch import Tensor, nn

from depth_router import DepthRouterConfig, DepthRouterModel
from depth_router.causal_memory import posterior_entropy
from depth_router.conformal_memory import (
    anytime_bonferroni_thresholds,
    aps_calibration_threshold,
    aps_prediction_mask,
    empirical_coverage,
    mean_set_size,
    nested_anytime_masks,
)
from depth_router.toy import PointerChaseSpec, pointer_chase_batch


def forward_with_spin_logits(
    model: DepthRouterModel,
    input_ids: Tensor,
) -> tuple[Tensor, list[Tensor]]:
    spin_logits: list[Tensor] = []

    def capture(_module: nn.Module, _args: tuple[Tensor, ...], output: Tensor) -> None:
        spin_logits.append(model.classify(output))

    handle = model.shared_block.register_forward_hook(capture)
    try:
        final_logits = model(input_ids)
    finally:
        handle.remove()

    return final_logits, spin_logits


def train_model(
    *,
    mode: str,
    config: DepthRouterConfig,
    spec: PointerChaseSpec,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> DepthRouterModel:
    torch.manual_seed(seed)
    random.seed(seed)

    model = DepthRouterModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    for _ in range(steps):
        input_ids, targets, _ = pointer_chase_batch(batch_size, spec)
        optimizer.zero_grad(set_to_none=True)
        final_logits, spin_logits = forward_with_spin_logits(model, input_ids)

        if mode == "final":
            loss = nn.functional.cross_entropy(final_logits, targets)
        elif mode == "monotone":
            per_spin = torch.stack(
                [
                    nn.functional.cross_entropy(
                        logits,
                        targets,
                        reduction="none",
                    )
                    for logits in spin_logits
                ]
            )
            final_loss = per_spin[-1].mean()
            early_loss = per_spin[:-1].mean()
            monotone_regret = torch.relu(per_spin[1:] - per_spin[:-1]).mean()
            loss = final_loss + 0.20 * early_loss + 0.50 * monotone_regret
        else:
            raise ValueError(f"unknown training mode: {mode}")

        loss.backward()
        optimizer.step()

    return model


@torch.no_grad()
def collect_traces(
    model: DepthRouterModel,
    spec: PointerChaseSpec,
    *,
    examples: int,
    batch_size: int,
) -> tuple[list[Tensor], Tensor]:
    per_spin: list[list[Tensor]] = [[] for _ in range(model.config.max_steps)]
    targets: list[Tensor] = []

    model.eval()
    remaining = examples
    while remaining > 0:
        current = min(batch_size, remaining)
        input_ids, target, _ = pointer_chase_batch(current, spec)
        _, spin_logits = forward_with_spin_logits(model, input_ids)

        for idx, logits in enumerate(spin_logits):
            per_spin[idx].append(torch.softmax(logits, dim=-1).cpu())
        targets.append(target.cpu())
        remaining -= current

    return [torch.cat(rows) for rows in per_spin], torch.cat(targets)


def top_mass_mask(probabilities: Tensor, mass: float) -> Tensor:
    order = torch.argsort(probabilities, dim=-1, descending=True)
    sorted_p = torch.gather(probabilities, 1, order)
    cumulative = sorted_p.cumsum(dim=-1)
    counts = (cumulative < mass).sum(dim=-1) + 1
    counts = counts.clamp_max(probabilities.shape[1])

    mask = torch.zeros_like(probabilities, dtype=torch.bool)
    ranks = torch.arange(probabilities.shape[1])[None, :]
    selected = ranks < counts[:, None]
    mask.scatter_(1, order, selected)
    return mask


def cross_entropy_nats(probabilities: Tensor, targets: Tensor) -> float:
    p_true = probabilities[
        torch.arange(probabilities.shape[0]),
        targets,
    ].clamp_min(1e-12)
    return float((-p_true.log()).mean().item())


def evaluate_mode(
    model: DepthRouterModel,
    spec: PointerChaseSpec,
    *,
    calibration_examples: int,
    test_examples: int,
    batch_size: int,
    alpha: float,
    page_bytes: int,
) -> dict:
    calibration, calibration_targets = collect_traces(
        model,
        spec,
        examples=calibration_examples,
        batch_size=batch_size,
    )
    test, test_targets = collect_traces(
        model,
        spec,
        examples=test_examples,
        batch_size=batch_size,
    )

    marginal_thresholds = [
        aps_calibration_threshold(p, calibration_targets, alpha=alpha)
        for p in calibration
    ]
    anytime_thresholds = anytime_bonferroni_thresholds(
        calibration,
        calibration_targets,
        alpha=alpha,
    )
    anytime_masks = nested_anytime_masks(test, anytime_thresholds)

    rows = []
    for spin, probabilities in enumerate(test):
        raw_mask = top_mass_mask(probabilities, 1.0 - alpha)
        marginal_mask = aps_prediction_mask(
            probabilities,
            marginal_thresholds[spin],
        )
        anytime_mask = anytime_masks[spin]

        ce = cross_entropy_nats(probabilities, test_targets)
        accuracy = float(
            (probabilities.argmax(dim=-1) == test_targets).float().mean().item()
        )

        rows.append(
            {
                "spin": spin + 1,
                "cross_entropy_nats": ce,
                "accuracy": accuracy,
                "posterior_entropy_nats": float(
                    posterior_entropy(probabilities).mean().item()
                ),
                "raw_top_mass_mean_pages": mean_set_size(raw_mask),
                "raw_top_mass_coverage": empirical_coverage(
                    raw_mask,
                    test_targets,
                ),
                "marginal_conformal_threshold": marginal_thresholds[spin],
                "marginal_conformal_mean_pages": mean_set_size(marginal_mask),
                "marginal_conformal_coverage": empirical_coverage(
                    marginal_mask,
                    test_targets,
                ),
                "anytime_threshold": anytime_thresholds[spin],
                "anytime_nested_mean_pages": mean_set_size(anytime_mask),
                "anytime_nested_coverage": empirical_coverage(
                    anytime_mask,
                    test_targets,
                ),
                "anytime_nested_mean_bytes": mean_set_size(anytime_mask)
                * page_bytes,
            }
        )

    final_anytime = anytime_masks[-1]
    ever_miss = ~final_anytime[
        torch.arange(test_targets.shape[0]),
        test_targets,
    ]

    return {
        "rows": rows,
        "simultaneous_anytime_miss_rate": float(ever_miss.float().mean().item()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--calibration-examples", type=int, default=3000)
    parser.add_argument("--test-examples", type=int, default=3000)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--ffn", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--page-bytes", type=int, default=1 << 20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["final", "monotone"],
        choices=["final", "monotone"],
    )
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = PointerChaseSpec(nodes=8, max_hops=args.max_steps)
    config = DepthRouterConfig(
        vocab_size=spec.vocab_size,
        max_seq_len=spec.seq_len,
        num_classes=spec.nodes,
        d_model=args.d_model,
        nhead=4,
        dim_feedforward=args.ffn,
        dropout=0.0,
        max_steps=args.max_steps,
        adapter_rank=4,
    )

    results = {}
    for offset, mode in enumerate(args.modes):
        model = train_model(
            mode=mode,
            config=config,
            spec=spec,
            steps=args.train_steps,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed + offset,
        )
        results[mode] = evaluate_mode(
            model,
            spec,
            calibration_examples=args.calibration_examples,
            test_examples=args.test_examples,
            batch_size=args.batch_size,
            alpha=args.alpha,
            page_bytes=args.page_bytes,
        )

    payload = {
        "experiment": "anytime_conformal_cache_v0",
        "claim_boundary": (
            "Toy recurrent-model experiment. Page identity is the pointer-chase "
            "latent/output class, not a real weight page."
        ),
        "config": {
            key: value
            for key, value in vars(args).items()
            if key != "summary_only"
        },
        "results": results,
    }
    if args.summary_only:
        payload["results"] = {
            mode: {
                "simultaneous_anytime_miss_rate": result[
                    "simultaneous_anytime_miss_rate"
                ],
                "rows": result["rows"],
            }
            for mode, result in results.items()
        }

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
