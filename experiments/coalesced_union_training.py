from __future__ import annotations

import argparse
import json
import random

import torch
from torch import Tensor, nn

from depth_router.union_cost import (
    expected_independent_bytes,
    expected_union_bytes,
    normalized_independent_cost,
    normalized_union_cost,
)


class FamilyRouter(nn.Module):
    """Controlled router that isolates the physical batch-cost objective."""

    def __init__(self, families: int) -> None:
        super().__init__()
        self.families = families
        self.pages = families + 1
        self.logits = nn.Parameter(torch.full((families, self.pages), -4.0))

        # Page 0 is valid for every family. Page f+1 is a private but exactly
        # behavior-equivalent implementation for family f. Start by preferring
        # private pages so coalescing must be learned rather than assumed.
        with torch.no_grad():
            self.logits[:, 0] = 0.5
            for family in range(families):
                self.logits[family, family + 1] = 2.5

    def forward(self, family_ids: Tensor) -> Tensor:
        return torch.softmax(self.logits[family_ids], dim=-1)


def valid_mass(probabilities: Tensor, family_ids: Tensor) -> Tensor:
    rows = torch.arange(family_ids.shape[0], device=family_ids.device)
    shared = probabilities[:, 0]
    private = probabilities[rows, family_ids + 1]
    return shared + private


def train(
    *,
    objective: str,
    families: int,
    repeats_per_family: int,
    steps: int,
    lr: float,
    cost_weight: float,
    page_bytes: int,
    seed: int,
) -> FamilyRouter:
    torch.manual_seed(seed)
    random.seed(seed)
    router = FamilyRouter(families)
    optimizer = torch.optim.Adam(router.parameters(), lr=lr)
    byte_tensor = torch.full((families + 1,), float(page_bytes))
    base_batch = torch.arange(families).repeat_interleave(repeats_per_family)

    for _ in range(steps):
        family_ids = base_batch[torch.randperm(base_batch.numel())]
        probabilities = router(family_ids)

        task_loss = -valid_mass(
            probabilities,
            family_ids,
        ).clamp_min(1e-12).log().mean()

        if objective == "independent":
            physical_cost = normalized_independent_cost(
                probabilities,
                byte_tensor,
            )
        elif objective == "union":
            physical_cost = normalized_union_cost(
                probabilities,
                byte_tensor,
            )
        else:
            raise ValueError(f"unknown objective: {objective}")

        loss = task_loss + cost_weight * physical_cost
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return router


@torch.no_grad()
def evaluate(
    router: FamilyRouter,
    *,
    repeats_per_family: int,
    page_bytes: int,
) -> dict[str, float]:
    family_ids = torch.arange(router.families).repeat_interleave(
        repeats_per_family
    )
    probabilities = router(family_ids)
    hard = probabilities.argmax(dim=-1)

    valid = (hard == 0) | (hard == family_ids + 1)
    shared_fraction = (hard == 0).float().mean()
    unique = torch.unique(hard)

    hard_probabilities = torch.nn.functional.one_hot(
        hard,
        num_classes=router.pages,
    ).float()
    byte_tensor = torch.full((router.pages,), float(page_bytes))

    return {
        "task_valid_fraction": float(valid.float().mean().item()),
        "mean_valid_probability": float(
            valid_mass(probabilities, family_ids).mean().item()
        ),
        "shared_page_fraction": float(shared_fraction.item()),
        "mean_pages_per_token": 1.0,
        "unique_pages_per_batch": float(unique.numel()),
        "hard_coalesced_bytes": float(
            expected_union_bytes(
                hard_probabilities,
                byte_tensor,
            ).item()
        ),
        "hard_independent_bytes": float(
            expected_independent_bytes(
                hard_probabilities,
                byte_tensor,
            ).item()
        ),
        "soft_expected_union_bytes": float(
            expected_union_bytes(
                probabilities,
                byte_tensor,
            ).item()
        ),
        "soft_expected_independent_bytes": float(
            expected_independent_bytes(
                probabilities,
                byte_tensor,
            ).item()
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", type=int, default=8)
    parser.add_argument("--repeats-per-family", type=int, default=8)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--cost-weight", type=float, default=1.0)
    parser.add_argument("--page-bytes", type=int, default=1 << 20)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--assert-gate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.families < 2:
        raise ValueError("families must be >= 2")
    if args.repeats_per_family < 1:
        raise ValueError("repeats-per-family must be >= 1")

    results = {}
    for objective in ("independent", "union"):
        router = train(
            objective=objective,
            families=args.families,
            repeats_per_family=args.repeats_per_family,
            steps=args.steps,
            lr=args.lr,
            cost_weight=args.cost_weight,
            page_bytes=args.page_bytes,
            seed=args.seed,
        )
        results[objective] = evaluate(
            router,
            repeats_per_family=args.repeats_per_family,
            page_bytes=args.page_bytes,
        )

    independent = results["independent"]
    union = results["union"]
    payload = {
        "experiment": "coalesced_union_training_v0",
        "hypothesis": (
            "Expected-union-byte training coordinates behaviorally equivalent "
            "routes onto shared physical pages, while a token-separable byte "
            "objective cannot distinguish those routes."
        ),
        "config": vars(args),
        "results": results,
        "comparison": {
            "unique_page_reduction_fraction": (
                independent["unique_pages_per_batch"]
                - union["unique_pages_per_batch"]
            )
            / max(independent["unique_pages_per_batch"], 1.0),
            "coalesced_byte_reduction_fraction": (
                independent["hard_coalesced_bytes"]
                - union["hard_coalesced_bytes"]
            )
            / max(independent["hard_coalesced_bytes"], 1.0),
            "task_valid_delta": (
                union["task_valid_fraction"]
                - independent["task_valid_fraction"]
            ),
        },
        "claim_boundary": (
            "Controlled identifiability experiment with deliberately redundant "
            "behavioral routes. It tests the optimization geometry, not a "
            "language-model speedup."
        ),
    }
    if args.assert_gate:
        comparison = payload["comparison"]
        if results["independent"]["task_valid_fraction"] < 0.999:
            raise RuntimeError("independent baseline failed the controlled task")
        if results["union"]["task_valid_fraction"] < 0.999:
            raise RuntimeError("union-byte objective failed the controlled task")
        if comparison["unique_page_reduction_fraction"] < 0.50:
            raise RuntimeError(
                "union-byte objective did not reduce unique pages by at least 50%"
            )
        if comparison["coalesced_byte_reduction_fraction"] < 0.50:
            raise RuntimeError(
                "union-byte objective did not reduce coalesced bytes by at least 50%"
            )

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
