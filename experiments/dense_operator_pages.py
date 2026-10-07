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
    distortion_coverage,
    distortion_prefix_calibration_threshold,
    required_set_coverage,
)
from depth_router.operator_pages import (
    OperatorPage,
    activation_address_scores,
    contiguous_mlp_pages,
    input_sketch_address_scores,
    input_sketch_metadata_bytes,
    linear2_page_contributions,
    make_input_page_sketches,
    page_output_weight_norms,
)
from depth_router.toy import PointerChaseSpec, pointer_chase_batch


@dataclass
class TraceBatch:
    features_by_spin: list[Tensor]
    mlp_inputs_by_spin: list[Tensor]
    activations_by_spin: list[Tensor]
    required_pages: Tensor
    base_without_linear2: Tensor
    page_contributions: Tensor
    targets: Tensor
    full_logits: Tensor


def capture_forward(
    model: DepthRouterModel,
    input_ids: Tensor,
) -> tuple[Tensor, list[Tensor], list[Tensor], Tensor, list[Tensor]]:
    block_inputs: list[Tensor] = []
    block_outputs: list[Tensor] = []
    mlp_inputs: list[Tensor] = []
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

    def linear1_pre(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
        mlp_inputs.append(args[0])

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
        model.shared_block.linear1.register_forward_pre_hook(linear1_pre),
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
    inputs = [hidden[:, -1] for hidden in mlp_inputs]
    activations = [activation[:, -1] for activation in ff_activations]
    base = block_outputs[-1][:, -1] - linear2_outputs[-1][:, -1]
    return logits, features, inputs, base, activations


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


@torch.no_grad()
def exact_minimum_page_counts(
    model: DepthRouterModel,
    trace: TraceBatch,
    *,
    tolerance_nats: float,
    max_pages: int = 12,
) -> Tensor:
    """Exact minimum page count by enumerating all subsets for small P.

    This is a diagnostic oracle only. It validates whether apparent operator
    sparsity is real rather than an artifact of the greedy oracle.
    """

    num_pages = trace.page_contributions.shape[1]
    if num_pages > max_pages:
        raise ValueError(
            f"exact subset search limited to {max_pages} pages; got {num_pages}"
        )
    if tolerance_nats < 0:
        raise ValueError("tolerance_nats must be non-negative")

    subset_ids = torch.arange(1 << num_pages, dtype=torch.long)
    bit_ids = torch.arange(num_pages, dtype=torch.long)
    masks = ((subset_ids[:, None] >> bit_ids[None, :]) & 1).bool()
    counts = masks.sum(dim=-1)

    selected = torch.einsum(
        "sp,npd->nsd",
        masks.to(trace.page_contributions.dtype),
        trace.page_contributions,
    )
    hidden = trace.base_without_linear2[:, None, :] + selected
    bias = model.shared_block.linear2.bias
    if bias is not None:
        hidden = hidden + bias.detach().cpu()

    logits = classify_hidden(
        model.cpu(),
        hidden.reshape(-1, hidden.shape[-1]),
    ).reshape(hidden.shape[0], hidden.shape[1], -1)
    kl = kl_from_full(trace.full_logits[:, None, :], logits)
    valid = kl <= tolerance_nats

    large = torch.full_like(
        counts,
        fill_value=num_pages + 1,
    )[None, :].expand(valid.shape[0], -1)
    candidate_counts = torch.where(
        valid,
        counts[None, :].expand(valid.shape[0], -1),
        large,
    )
    best = candidate_counts.min(dim=-1).values
    if bool((best > num_pages).any()):
        raise RuntimeError("dense full-page subset failed the distortion target")
    return best


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
    mlp_inputs: list[list[Tensor]] = [
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
        logits, spin_features, spin_inputs, base, spin_activations = capture_forward(
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
        for idx, values in enumerate(spin_inputs):
            mlp_inputs[idx].append(values.cpu())
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
        mlp_inputs_by_spin=[torch.cat(rows) for rows in mlp_inputs],
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
def prefix_distortions_for_scores(
    model: DepthRouterModel,
    trace: TraceBatch,
    scores: Tensor,
) -> Tensor:
    """KL distortion for zero pages and every top-score prefix."""

    if scores.shape != trace.required_pages.shape:
        raise ValueError("scores must have shape [examples, pages]")

    order = torch.argsort(scores, dim=-1, descending=True)
    gather_index = order[..., None].expand(
        -1,
        -1,
        trace.page_contributions.shape[-1],
    )
    sorted_contributions = torch.gather(
        trace.page_contributions,
        1,
        gather_index,
    )
    prefix = sorted_contributions.cumsum(dim=1)
    zero = torch.zeros_like(prefix[:, :1])
    prefix = torch.cat([zero, prefix], dim=1)

    hidden = trace.base_without_linear2[:, None, :] + prefix
    bias = model.shared_block.linear2.bias
    if bias is not None:
        hidden = hidden + bias.detach().cpu()

    logits = classify_hidden(
        model.cpu(),
        hidden.reshape(-1, hidden.shape[-1]),
    ).reshape(hidden.shape[0], hidden.shape[1], -1)
    return kl_from_full(trace.full_logits[:, None, :], logits)


@torch.no_grad()
def sparse_distortions(
    model: DepthRouterModel,
    trace: TraceBatch,
    page_mask: Tensor,
) -> Tensor:
    selected = (
        trace.page_contributions
        * page_mask[..., None].to(trace.page_contributions.dtype)
    ).sum(dim=1)

    hidden = trace.base_without_linear2 + selected
    bias = model.shared_block.linear2.bias
    if bias is not None:
        hidden = hidden + bias.detach().cpu()

    logits = classify_hidden(model.cpu(), hidden)
    return kl_from_full(trace.full_logits, logits)


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
    distortion_tolerance: float,
    router_steps: int,
    router_lr: float,
    sketch_dims: list[int],
    seed: int,
) -> dict:
    page_bytes = torch.tensor([page.weight_bytes for page in pages])
    dense_bytes = int(page_bytes.sum().item())
    output_norms = page_output_weight_norms(
        model.shared_block.linear2,
        pages,
    ).cpu()
    output_norm_metadata_bytes = (
        output_norms.numel() * output_norms.element_size()
    )
    rows = []

    def evaluate_scores(
        *,
        method: str,
        spin: int,
        calibration_scores: Tensor,
        test_scores: Tensor,
        metadata_bytes: int,
        deployable_without_cold_payload: bool,
    ) -> None:
        calibration_prefix = prefix_distortions_for_scores(
            model,
            calibration,
            calibration_scores,
        )
        threshold = distortion_prefix_calibration_threshold(
            calibration_scores,
            calibration_prefix,
            tolerance=distortion_tolerance,
            alpha=alpha,
        )
        mask = aps_prediction_mask(test_scores, threshold)
        realized_distortion = sparse_distortions(model, test, mask)
        distortion_ok = distortion_coverage(
            realized_distortion,
            tolerance=distortion_tolerance,
        )
        oracle_set_coverage = required_set_coverage(
            mask,
            test.required_pages,
        )
        resident_bytes = (
            mask.to(torch.long) * page_bytes[None, :]
        ).sum(dim=-1)
        quality = sparse_quality(model, test, mask)
        rows.append(
            {
                "method": method,
                "spin": spin + 1,
                "calibrated_mass_threshold": threshold,
                "distortion_tolerance_nats": distortion_tolerance,
                "distortion_coverage": distortion_ok,
                "oracle_set_coverage_diagnostic": oracle_set_coverage,
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
                "metadata_bytes": int(metadata_bytes),
                "metadata_fraction_of_cold": float(
                    metadata_bytes / max(dense_bytes, 1)
                ),
                "deployable_without_cold_payload": deployable_without_cold_payload,
                "realized_distortion_violation_rate": 1.0 - distortion_ok,
                **quality,
            }
        )

    min_page_width = min(page.width for page in pages)
    for sketch_dim in sketch_dims:
        if sketch_dim < 1 or sketch_dim > min_page_width:
            raise ValueError(
                "every sketch dimension must be between 1 and the minimum page width"
            )

    input_sketches = {
        sketch_dim: make_input_page_sketches(
            model.shared_block.linear1,
            pages,
            sketch_dim=sketch_dim,
            seed=seed + 1000 + sketch_dim,
        )
        for sketch_dim in sketch_dims
    }

    for spin in range(model.config.max_steps):
        router = train_page_router(
            train.features_by_spin[spin],
            train.required_pages,
            steps=router_steps,
            lr=router_lr,
            seed=seed + spin,
        )
        router_metadata_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in router.parameters()
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
            metadata_bytes=router_metadata_bytes,
            deployable_without_cold_payload=True,
        )

        # Diagnostic upper-ish baseline: useful for understanding whether the
        # post-linear1 activation contains the address, but not deployable if W1
        # is itself cold because producing this activation already read W1.
        evaluate_scores(
            method="post_linear1_energy_diagnostic",
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
            metadata_bytes=output_norm_metadata_bytes,
            deployable_without_cold_payload=False,
        )

        for sketch_dim, sketches in input_sketches.items():
            metadata_bytes = (
                input_sketch_metadata_bytes(sketches)
                + output_norm_metadata_bytes
            )
            evaluate_scores(
                method=f"input_sketch_r{sketch_dim}",
                spin=spin,
                calibration_scores=input_sketch_address_scores(
                    calibration.mlp_inputs_by_spin[spin],
                    pages,
                    sketches,
                    output_norms,
                ),
                test_scores=input_sketch_address_scores(
                    test.mlp_inputs_by_spin[spin],
                    pages,
                    sketches,
                    output_norms,
                ),
                metadata_bytes=metadata_bytes,
                deployable_without_cold_payload=True,
            )

    oracle_pages = train.required_pages.sum(dim=-1).float()
    exact_counts = exact_minimum_page_counts(
        model,
        test,
        tolerance_nats=distortion_tolerance,
    ).float()
    return {
        "num_operator_pages": len(pages),
        "dense_cold_bytes": dense_bytes,
        "distortion_tolerance_nats": distortion_tolerance,
        "train_greedy_mean_required_pages": float(oracle_pages.mean().item()),
        "train_zero_fault_rate": float(
            (oracle_pages == 0).float().mean().item()
        ),
        "test_exact_mean_required_pages": float(exact_counts.mean().item()),
        "test_exact_p95_required_pages": float(
            torch.quantile(exact_counts, 0.95).item()
        ),
        "test_exact_zero_fault_rate": float(
            (exact_counts == 0).float().mean().item()
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
    parser.add_argument(
        "--sketch-dims",
        nargs="+",
        type=int,
        default=[1, 2, 4],
    )
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
        distortion_tolerance=args.oracle_kl,
        router_steps=args.router_steps,
        router_lr=args.router_lr,
        sketch_dims=args.sketch_dims,
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
            "shared FFN. Residency is calibrated directly against KL distortion "
            "from dense logits; exact oracle page identity is diagnostic only."
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
