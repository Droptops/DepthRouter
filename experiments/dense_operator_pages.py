from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from depth_router import DepthRouterConfig, DepthRouterModel
from depth_router.conformal_memory import (
    aps_prediction_mask,
    required_set_calibration_threshold,
    required_set_coverage,
)
from depth_router.operator_pages import (
    OperatorPage,
    activation_address_scores,
    contiguous_mlp_pages,
    linear2_page_contributions,
    page_output_weight_norms,
)
from depth_router.toy import PointerChaseSpec, pointer_chase_batch


@dataclass
class TraceBatch:
    features_by_spin: list[Tensor]
    activations_by_spin: list[Tensor]
    required_pages: Tensor
    base_without_linear2: Tensor
    page_contributions: Tensor
    targets: Tensor
    full_logits: Tensor


def capture_forward(
    model: DepthRouterModel,
    input_ids: Tensor,
) -> tuple[Tensor, list[Tensor], Tensor, list[Tensor]]:
    block_inputs: list[Tensor] = []
    block_outputs: list[Tensor] = []
    ff_activations: list[Tensor] = []
    linear2_outputs: list[Tensor] = []

    def block_pre(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
        block_inputs.append(args[0])

    def block_post(
        _module: nn.Module,
        _args: tuple[Tensor, ...],
        output: Tensor,
    ) -> None:
        block_outputs.append(output)

    def linear2_pre(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
        ff_activations.append(args[0])

    def linear2_post(
        _module: nn.Module,
        _args: tuple[Tensor, ...],
        output: Tensor,
    ) -> None:
        linear2_outputs.append(output)

    handles = [
        model.shared_block.register_forward_pre_hook(block_pre),
        model.shared_block.register_forward_hook(block_post),
        model.shared_block.linear2.register_forward_pre_hook(linear2_pre),
        model.shared_block.linear2.register_forward_hook(linear2_post),
    ]
    try:
        logits = model(input_ids)
    finally:
        for handle in handles:
            handle.remove()

    if len(block_inputs) != model.config.max_steps:
        raise RuntimeError("did not capture every recurrent spin")

    features = [hidden[:, -1] for hidden in block_inputs]
    activations = [activation[:, -1] for activation in ff_activations]
    base = block_outputs[-1][:, -1] - linear2_outputs[-1][:, -1]
    return logits, features, base, activations


def classify_hidden(model: DepthRouterModel, hidden: Tensor) -> Tensor:
    return model.head(model.final_norm(hidden))


def kl_from_full(full_logits: Tensor, candidate_logits: Tensor) -> Tensor:
    full_logp = torch.log_softmax(full_logits.float(), dim=-1)
    candidate_logp = torch.log_softmax(candidate_logits.float(), dim=-1)
    p = full_logp.exp()
    return (p * (full_logp - candidate_logp)).sum(dim=-1)


@torch.no_grad()
def greedy_required_pages(
    model: DepthRouterModel,
    *,
    base_without_linear2: Tensor,
    page_contributions: Tensor,
    full_logits: Tensor,
    tolerance_nats: float,
) -> Tensor:
    """Greedy oracle: minimum-found page set reaching a logit-KL tolerance."""

    if tolerance_nats < 0:
        raise ValueError("tolerance_nats must be non-negative")

    examples, num_pages, _ = page_contributions.shape
    selected = torch.zeros(
        examples,
        num_pages,
        dtype=torch.bool,
        device=page_contributions.device,
    )

    bias = model.shared_block.linear2.bias
    current = base_without_linear2.clone()
    if bias is not None:
        current = current + bias

    rows = torch.arange(examples, device=current.device)

    for _ in range(num_pages + 1):
        current_logits = classify_hidden(model, current)
        current_kl = kl_from_full(full_logits, current_logits)
        active = current_kl > tolerance_nats
        if not bool(active.any()):
            break

        candidates = current[:, None, :] + page_contributions
        candidate_logits = classify_hidden(
            model,
            candidates.reshape(-1, candidates.shape[-1]),
        ).reshape(examples, num_pages, -1)

        full_logp = torch.log_softmax(full_logits.float(), dim=-1)
        candidate_logp = torch.log_softmax(candidate_logits.float(), dim=-1)
        p = full_logp.exp()[:, None, :]
        kl = (p * (full_logp[:, None, :] - candidate_logp)).sum(dim=-1)
        kl = kl.masked_fill(selected, float("inf"))

        best = kl.argmin(dim=-1)
        selected[rows[active], best[active]] = True
        current[active] = (
            current[active]
            + page_contributions[rows[active], best[active]]
        )
    else:
        raise RuntimeError("oracle failed to converge after selecting every page")

    return selected


def train_recurrent_model(
    *,
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

        spin_logits: list[Tensor] = []

        def capture(
            _module: nn.Module,
            _args: tuple[Tensor, ...],
            output: Tensor,
            _sink: list[Tensor] = spin_logits,
        ) -> None:
            _sink.append(classify_hidden(model, output[:, -1]))

        handle = model.shared_block.register_forward_hook(capture)
        try:
            final_logits = model(input_ids)
        finally:
            handle.remove()

        losses = torch.stack(
            [
                nn.functional.cross_entropy(
                    logits,
                    targets,
                    reduction="none",
                )
                for logits in spin_logits
            ]
        )
        monotone_regret = torch.relu(losses[1:] - losses[:-1]).mean()
        loss = (
            nn.functional.cross_entropy(final_logits, targets)
            + 0.20 * losses[:-1].mean()
            + 0.50 * monotone_regret
        )
        loss.backward()
        optimizer.step()

    return model


@torch.no_grad()
def collect_dataset(
    model: DepthRouterModel,
    spec: PointerChaseSpec,
    pages: list[OperatorPage],
    *,
    examples: int,
    batch_size: int,
    tolerance_nats: float,
) -> TraceBatch:
    features: list[list[Tensor]] = [
        [] for _ in range(model.config.max_steps)
    ]
    activations: list[list[Tensor]] = [
        [] for _ in range(model.config.max_steps)
    ]
    required: list[Tensor] = []
    bases: list[Tensor] = []
    contributions: list[Tensor] = []
    targets_all: list[Tensor] = []
    logits_all: list[Tensor] = []

    model.eval()
    remaining = examples
    while remaining > 0:
        current = min(batch_size, remaining)
        input_ids, targets, _ = pointer_chase_batch(current, spec)
        logits, spin_features, base, spin_activations = capture_forward(
            model,
            input_ids,
        )
        page_contrib = linear2_page_contributions(
            spin_activations[-1],
            model.shared_block.linear2,
            pages,
        )

        required_mask = greedy_required_pages(
            model,
            base_without_linear2=base,
            page_contributions=page_contrib,
            full_logits=logits,
            tolerance_nats=tolerance_nats,
        )

        for idx, values in enumerate(spin_features):
            features[idx].append(values.cpu())
        for idx, values in enumerate(spin_activations):
            activations[idx].append(values.cpu())
        required.append(required_mask.cpu())
        bases.append(base.cpu())
        contributions.append(page_contrib.cpu())
        targets_all.append(targets.cpu())
        logits_all.append(logits.cpu())
        remaining -= current

    return TraceBatch(
        features_by_spin=[torch.cat(rows) for rows in features],
        activations_by_spin=[torch.cat(rows) for rows in activations],
        required_pages=torch.cat(required),
        base_without_linear2=torch.cat(bases),
        page_contributions=torch.cat(contributions),
        targets=torch.cat(targets_all),
        full_logits=torch.cat(logits_all),
    )


def train_page_router(
    features: Tensor,
    labels: Tensor,
    *,
    steps: int,
    lr: float,
    seed: int,
) -> nn.Linear:
    torch.manual_seed(seed)
    router = nn.Linear(features.shape[-1], labels.shape[-1])
    optimizer = torch.optim.AdamW(router.parameters(), lr=lr)

    positives = labels.float().sum(dim=0)
    negatives = labels.shape[0] - positives
    pos_weight = (negatives / positives.clamp_min(1.0)).clamp(1.0, 20.0)

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = router(features)
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits,
            labels.float(),
            pos_weight=pos_weight,
        )
        loss.backward()
        optimizer.step()

    return router


def router_scores(router: nn.Linear, features: Tensor) -> Tensor:
    scores = torch.sigmoid(router(features))
    return scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def sparse_quality(
    model: DepthRouterModel,
    trace: TraceBatch,
    page_mask: Tensor,
) -> dict[str, float]:
    selected = (
        trace.page_contributions
        * page_mask[..., None].to(trace.page_contributions.dtype)
    ).sum(dim=1)

    hidden = trace.base_without_linear2 + selected
    bias = model.shared_block.linear2.bias
    if bias is not None:
        hidden = hidden + bias.cpu()

    logits = classify_hidden(model.cpu(), hidden)
    ce = nn.functional.cross_entropy(logits, trace.targets)
    full_ce = nn.functional.cross_entropy(
        trace.full_logits,
        trace.targets,
    )
    kl = kl_from_full(trace.full_logits, logits)
    accuracy = (logits.argmax(dim=-1) == trace.targets).float().mean()

    return {
        "cross_entropy_nats": float(ce.item()),
        "dense_cross_entropy_nats": float(full_ce.item()),
        "mean_kl_from_dense_nats": float(kl.mean().item()),
        "accuracy": float(accuracy.item()),
    }


def evaluate(
    model: DepthRouterModel,
    train: TraceBatch,
    calibration: TraceBatch,
    test: TraceBatch,
    pages: list[OperatorPage],
    *,
    alpha: float,
    router_steps: int,
    router_lr: float,
    seed: int,
) -> dict:
    page_bytes = torch.tensor([page.weight_bytes for page in pages])
    dense_bytes = int(page_bytes.sum().item())
    output_norms = page_output_weight_norms(
        model.shared_block.linear2,
        pages,
    ).cpu()
    rows = []

    def evaluate_scores(
        *,
        method: str,
        spin: int,
        calibration_scores: Tensor,
        test_scores: Tensor,
    ) -> None:
        threshold = required_set_calibration_threshold(
            calibration_scores,
            calibration.required_pages,
            alpha=alpha,
        )
        mask = aps_prediction_mask(test_scores, threshold)
        coverage = required_set_coverage(mask, test.required_pages)
        resident_bytes = (
            mask.to(torch.long) * page_bytes[None, :]
        ).sum(dim=-1)
        quality = sparse_quality(model, test, mask)
        rows.append(
            {
                "method": method,
                "spin": spin + 1,
                "conformal_threshold": threshold,
                "required_set_coverage": coverage,
                "mean_predicted_pages": float(
                    mask.sum(dim=-1).float().mean().item()
                ),
                "mean_resident_bytes": float(
                    resident_bytes.float().mean().item()
                ),
                "dense_cold_bytes": dense_bytes,
                "resident_fraction": float(
                    resident_bytes.float().mean().item() / dense_bytes
                ),
                **quality,
            }
        )

    for spin in range(model.config.max_steps):
        router = train_page_router(
            train.features_by_spin[spin],
            train.required_pages,
            steps=router_steps,
            lr=router_lr,
            seed=seed + spin,
        )
        evaluate_scores(
            method="learned_router",
            spin=spin,
            calibration_scores=router_scores(
                router,
                calibration.features_by_spin[spin],
            ),
            test_scores=router_scores(
                router,
                test.features_by_spin[spin],
            ),
        )

        evaluate_scores(
            method="intrinsic_address",
            spin=spin,
            calibration_scores=activation_address_scores(
                calibration.activations_by_spin[spin],
                pages,
                output_norms,
            ),
            test_scores=activation_address_scores(
                test.activations_by_spin[spin],
                pages,
                output_norms,
            ),
        )

    oracle_pages = train.required_pages.sum(dim=-1).float()
    return {
        "num_operator_pages": len(pages),
        "dense_cold_bytes": dense_bytes,
        "train_oracle_mean_required_pages": float(oracle_pages.mean().item()),
        "train_zero_fault_rate": float(
            (oracle_pages == 0).float().mean().item()
        ),
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-steps", type=int, default=700)
    parser.add_argument("--model-batch-size", type=int, default=128)
    parser.add_argument("--router-steps", type=int, default=250)
    parser.add_argument("--router-lr", type=float, default=2e-2)
    parser.add_argument("--train-examples", type=int, default=2000)
    parser.add_argument("--calibration-examples", type=int, default=1000)
    parser.add_argument("--test-examples", type=int, default=1000)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--ffn", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--units-per-page", type=int, default=8)
    parser.add_argument("--model-lr", type=float, default=1e-3)
    parser.add_argument("--oracle-kl", type=float, default=0.001)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=17)
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
    model = train_recurrent_model(
        config=config,
        spec=spec,
        steps=args.train_steps,
        batch_size=args.model_batch_size,
        lr=args.model_lr,
        seed=args.seed,
    )
    pages = contiguous_mlp_pages(
        model.shared_block.linear1,
        model.shared_block.linear2,
        units_per_page=args.units_per_page,
    )

    train = collect_dataset(
        model,
        spec,
        pages,
        examples=args.train_examples,
        batch_size=args.model_batch_size,
        tolerance_nats=args.oracle_kl,
    )
    calibration = collect_dataset(
        model,
        spec,
        pages,
        examples=args.calibration_examples,
        batch_size=args.model_batch_size,
        tolerance_nats=args.oracle_kl,
    )
    test = collect_dataset(
        model,
        spec,
        pages,
        examples=args.test_examples,
        batch_size=args.model_batch_size,
        tolerance_nats=args.oracle_kl,
    )

    result = evaluate(
        model,
        train,
        calibration,
        test,
        pages,
        alpha=args.alpha,
        router_steps=args.router_steps,
        router_lr=args.router_lr,
        seed=args.seed + 100,
    )

    if result["train_zero_fault_rate"] > 0.95:
        raise RuntimeError(
            "operator-page oracle is degenerate: >95% of training examples "
            "need no cold page; tighten --oracle-kl before interpreting results"
        )

    payload = {
        "experiment": "dense_operator_pages_v0",
        "claim_boundary": (
            "Toy dense transformer. Operator pages are exact slices of the "
            "shared FFN; the oracle set is a greedy set preserving final logits "
            "within the configured KL tolerance."
        ),
        "config": {
            key: value
            for key, value in vars(args).items()
            if key != "summary_only"
        },
        "result": result,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
