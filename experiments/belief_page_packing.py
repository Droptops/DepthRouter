from __future__ import annotations

import argparse
import json
import math
import statistics

import torch

from depth_router.causal_memory import (
    PageLayout,
    coalesced_transfer_bytes,
    greedy_affinity_layout,
    posterior_affinity,
    posterior_entropy,
    predictive_working_set_bytes,
    random_layout,
    requested_pages,
    sequential_layout,
    working_set_pages,
)


def make_family_mapping(
    *,
    families: int,
    states_per_family: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    total = families * states_per_family
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(total, generator=generator)
    family_of_state = torch.empty(total, dtype=torch.long)
    for family in range(families):
        members = permutation[
            family * states_per_family : (family + 1) * states_per_family
        ]
        family_of_state[members] = family
    return family_of_state, permutation


def sample_posteriors(
    *,
    samples: int,
    family_of_state: torch.Tensor,
    family_mass: float,
    true_state_mass_within_family: float,
    seed: int,
) -> torch.Tensor:
    if not 0 < family_mass < 1:
        raise ValueError("family_mass must be in (0, 1)")
    if not 0 < true_state_mass_within_family < 1:
        raise ValueError("true_state_mass_within_family must be in (0, 1)")

    generator = torch.Generator().manual_seed(seed)
    num_states = int(family_of_state.numel())
    families = int(family_of_state.max().item()) + 1
    states_by_family = [
        torch.nonzero(family_of_state == family, as_tuple=False).flatten()
        for family in range(families)
    ]

    true_states = torch.randint(
        num_states,
        (samples,),
        generator=generator,
    )
    posterior = torch.empty(samples, num_states)

    for row, true_state in enumerate(true_states.tolist()):
        family = int(family_of_state[true_state])
        family_states = states_by_family[family]
        outside = torch.nonzero(
            family_of_state != family,
            as_tuple=False,
        ).flatten()

        q = torch.zeros(num_states)
        sibling_mass = family_mass * (1.0 - true_state_mass_within_family)
        true_mass = family_mass * true_state_mass_within_family

        q[true_state] = true_mass
        siblings = family_states[family_states != true_state]
        if siblings.numel():
            q[siblings] = sibling_mass / siblings.numel()
        q[outside] = (1.0 - family_mass) / outside.numel()

        # Multiplicative jitter keeps the held-out posterior family structure
        # without making every row identical.
        jitter = 0.85 + 0.30 * torch.rand(num_states, generator=generator)
        q *= jitter
        q /= q.sum()
        posterior[row] = q

    return posterior


def oracle_family_layout(
    family_of_state: torch.Tensor,
    *,
    states_per_page: int,
    page_bytes: int,
) -> PageLayout:
    # The synthetic generator uses exactly states_per_page states per family.
    assignment = family_of_state.clone()
    return PageLayout(assignment, states_per_page, page_bytes)


def lru_faults(page_requests: list[int], capacity: int) -> int:
    if capacity < 1:
        raise ValueError("cache capacity must be >= 1")

    cache: list[int] = []
    faults = 0
    for page in page_requests:
        if page in cache:
            cache.remove(page)
            cache.append(page)
            continue

        faults += 1
        if len(cache) >= capacity:
            cache.pop(0)
        cache.append(page)

    return faults


def scheduling_faults(
    posterior: torch.Tensor,
    layout: PageLayout,
    *,
    mass_target: float,
    batch_size: int,
    cache_pages: int,
) -> tuple[int, int]:
    """Compare token-major demand with fault-grouped execution.

    The cache is reset at each scheduling window so the comparison measures
    within-window locality rather than relying on cross-window warm state.
    """

    requests = requested_pages(
        posterior,
        layout,
        mass_target=mass_target,
    )
    token_major_faults = 0
    grouped_faults = 0

    for start in range(0, len(requests), batch_size):
        window = requests[start : start + batch_size]
        token_major = [page for token_pages in window for page in token_pages]
        token_major_faults += lru_faults(token_major, cache_pages)

        # A fault-grouped scheduler runs every token needing a page while that
        # page is resident, so each unique page is loaded at most once here.
        grouped_faults += len(set(token_major))

    return token_major_faults, grouped_faults


def summarize(
    posterior: torch.Tensor,
    layout: PageLayout,
    *,
    mass_target: float,
    batch_size: int,
) -> dict[str, float]:
    pages = working_set_pages(posterior, layout, mass_target=mass_target).float()
    nbytes = predictive_working_set_bytes(
        posterior,
        layout,
        mass_target=mass_target,
    ).float()

    ratios = []
    transfer_bytes = []
    for start in range(0, posterior.shape[0], batch_size):
        batch = posterior[start : start + batch_size]
        requested, unique, moved = coalesced_transfer_bytes(
            batch,
            layout,
            mass_target=mass_target,
        )
        ratios.append(unique / max(requested, 1))
        transfer_bytes.append(moved)

    token_major_faults, grouped_faults = scheduling_faults(
        posterior,
        layout,
        mass_target=mass_target,
        batch_size=batch_size,
        cache_pages=4,
    )

    return {
        "entropy_nats": float(posterior_entropy(posterior).mean()),
        "mean_pages": float(pages.mean()),
        "p95_pages": float(torch.quantile(pages, 0.95)),
        "mean_working_set_bytes": float(nbytes.mean()),
        "coalesced_unique_over_requested": statistics.mean(ratios),
        "mean_batch_transfer_bytes": statistics.mean(transfer_bytes),
        "token_major_page_faults": float(token_major_faults),
        "fault_grouped_page_faults": float(grouped_faults),
        "fault_grouping_ratio": grouped_faults / max(token_major_faults, 1),
    }


def run_seed(args: argparse.Namespace, seed: int) -> dict:
    family_of_state, _ = make_family_mapping(
        families=args.families,
        states_per_family=args.states_per_page,
        seed=seed,
    )
    num_states = int(family_of_state.numel())

    train = sample_posteriors(
        samples=args.train_samples,
        family_of_state=family_of_state,
        family_mass=args.family_mass,
        true_state_mass_within_family=args.true_mass,
        seed=seed + 1000,
    )
    test = sample_posteriors(
        samples=args.test_samples,
        family_of_state=family_of_state,
        family_mass=args.family_mass,
        true_state_mass_within_family=args.true_mass,
        seed=seed + 2000,
    )

    layouts = {
        "sequential": sequential_layout(
            num_states,
            states_per_page=args.states_per_page,
            page_bytes=args.page_bytes,
        ),
        "random": random_layout(
            num_states,
            states_per_page=args.states_per_page,
            page_bytes=args.page_bytes,
            seed=seed + 3000,
        ),
        "posterior_affinity": greedy_affinity_layout(
            posterior_affinity(train),
            states_per_page=args.states_per_page,
            page_bytes=args.page_bytes,
        ),
        "oracle_family": oracle_family_layout(
            family_of_state,
            states_per_page=args.states_per_page,
            page_bytes=args.page_bytes,
        ),
    }

    metrics = {
        name: summarize(
            test,
            layout,
            mass_target=args.mass_target,
            batch_size=args.batch_size,
        )
        for name, layout in layouts.items()
    }
    return {"seed": seed, "metrics": metrics}


def aggregate(results: list[dict]) -> dict:
    names = list(results[0]["metrics"])
    out: dict[str, dict[str, float]] = {}
    for name in names:
        metric_names = list(results[0]["metrics"][name])
        out[name] = {}
        for metric in metric_names:
            values = [r["metrics"][name][metric] for r in results]
            out[name][metric] = statistics.mean(values)
            out[name][f"{metric}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", type=int, default=16)
    parser.add_argument("--states-per-page", type=int, default=8)
    parser.add_argument("--page-bytes", type=int, default=1 << 20)
    parser.add_argument("--train-samples", type=int, default=4000)
    parser.add_argument("--test-samples", type=int, default=4000)
    parser.add_argument("--family-mass", type=float, default=0.94)
    parser.add_argument("--true-mass", type=float, default=0.45)
    parser.add_argument("--mass-target", type=float, default=0.90)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.states_per_page < 2:
        raise ValueError("states-per-page must be >= 2")
    if args.seeds < 1:
        raise ValueError("seeds must be >= 1")

    results = [run_seed(args, seed) for seed in range(args.seeds)]
    payload = {
        "experiment": "belief_page_packing_v0",
        "hypothesis": (
            "Packing states that are jointly plausible into one physical page "
            "reduces the bytes required to cover a fixed posterior mass."
        ),
        "config": {
            key: value
            for key, value in vars(args).items()
            if key != "summary_only"
        },
        "aggregate": aggregate(results),
    }
    if not args.summary_only:
        payload["runs"] = results
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
