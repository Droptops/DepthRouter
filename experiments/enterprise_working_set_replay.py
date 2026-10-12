"""Enterprise working-set replay. No physical-HBM claims.

Input JSONL: {"task": str, "sequence": str, "step": int,
              "pages": [int, ...], "transition_signal": float}
Signal must be computed from causally available features BEFORE the demand.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


class CapacityError(ValueError):
    """Raised when a token's simultaneous demand cannot fit the cache budget."""


def load(path):
    rows = [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    for row in rows:
        if not isinstance(row["pages"], list) or not all(
            isinstance(page, int) and page >= 0 for page in row["pages"]
        ):
            raise ValueError("pages must be a list of non-negative integers")
        if float(row["transition_signal"]) < 0:
            raise ValueError("transition_signal must be non-negative")
    return rows


def unique_pages(pages):
    return list(dict.fromkeys(pages))


def insert_pages(cache, pages, capacity):
    """Insert priority-ordered pages and return cache, transfers, and evictions."""
    pages = unique_pages(pages)
    if len(pages) > capacity:
        raise CapacityError(f"insertion requires {len(pages)} pages but capacity is {capacity}")
    updated = unique_pages(pages + cache)[:capacity]
    transfers = sum(page not in cache for page in updated)
    evictions = sum(page not in updated for page in cache)
    if any(page not in updated for page in pages):
        raise AssertionError("requested insertion was truncated")
    return updated, transfers, evictions


def empty_policy_stats():
    return {
        "hits": 0,
        "misses": 0,
        "requests": 0,
        "tokens": 0,
        "transitions": 0,
        "evictions": 0,
        "prefetch_requests": 0,
        "prefetch_transfers": 0,
        "warmup_transfers": 0,
        "demand_transfers": 0,
        "total_transfers": 0,
        "cache_resets": 0,
        "max_resident_pages": 0,
    }


def record_residency(row, cache, capacity):
    if len(cache) > capacity:
        raise AssertionError("cache capacity exceeded")
    row["max_resident_pages"] = max(row["max_resident_pages"], len(cache))


def serve_and_retain(row, cache, requested, capacity):
    misses = sum(page not in cache for page in requested)
    row["hits"] += len(requested) - misses
    row["misses"] += misses
    cache, transfers, evictions = insert_pages(cache, requested, capacity)
    if transfers != misses:
        raise AssertionError("demand transfer accounting diverged from misses")
    row["demand_transfers"] += transfers
    row["evictions"] += evictions
    record_residency(row, cache, capacity)
    return cache


def prefetch_and_retain(row, cache, pages, capacity):
    pages = unique_pages(pages)
    row["prefetch_requests"] += len(pages)
    cache, transfers, evictions = insert_pages(cache, pages, capacity)
    row["prefetch_transfers"] += transfers
    row["evictions"] += evictions
    record_residency(row, cache, capacity)
    return cache


def evaluate(train, test, capacity, threshold):
    if capacity < 1:
        raise ValueError("capacity must be positive")
    for row in test:
        demand = unique_pages(row["pages"])
        if len(demand) > capacity:
            identity = f"{row['task']}/{row['sequence']} step {row['step']}"
            raise CapacityError(
                f"{identity} requires {len(demand)} pages but capacity is {capacity}"
            )

    freq = Counter(p for r in train for p in set(r["pages"]))
    popular = [p for p, _ in sorted(freq.items(), key=lambda x: (-x[1], x[0]))]
    policies = ("static", "previous", "transition", "oracle")
    stats = {policy: empty_policy_stats() for policy in policies}
    streams = defaultdict(list)
    for r in test:
        streams[(r["task"], r["sequence"])].append(r)
    for stream in streams.values():
        stream.sort(key=lambda r: r["step"])
        caches = {"static": popular[:capacity], "previous": [],
                  "transition": [], "oracle": []}
        for policy in policies:
            stats[policy]["cache_resets"] += 1
            record_residency(stats[policy], caches[policy], capacity)
        stats["static"]["warmup_transfers"] += len(caches["static"])

        for i, row in enumerate(stream):
            requested = unique_pages(row["pages"])
            for policy in policies:
                stats[policy]["requests"] += len(requested)
                stats[policy]["tokens"] += 1

            static_misses = sum(page not in caches["static"] for page in requested)
            stats["static"]["hits"] += len(requested) - static_misses
            stats["static"]["misses"] += static_misses
            stats["static"]["demand_transfers"] += static_misses

            caches["previous"] = serve_and_retain(
                stats["previous"], caches["previous"], requested, capacity
            )

            switched = float(row["transition_signal"]) >= threshold
            stats["transition"]["transitions"] += int(switched)
            if switched:
                # The signal is available before this demand. Popular pages come
                # only from training, and every insertion is charged as prefetch.
                caches["transition"] = prefetch_and_retain(
                    stats["transition"], caches["transition"], popular[:capacity], capacity
                )
            caches["transition"] = serve_and_retain(
                stats["transition"], caches["transition"], requested, capacity
            )

            caches["oracle"] = serve_and_retain(
                stats["oracle"], caches["oracle"], requested, capacity
            )
            if i + 1 < len(stream):
                # NONCAUSAL DIAGNOSTIC ORACLE: look-ahead is allowed only here,
                # after current demand, and all future-page insertions are charged.
                caches["oracle"] = prefetch_and_retain(
                    stats["oracle"], caches["oracle"], stream[i + 1]["pages"], capacity
                )

    for row in stats.values():
        row["total_transfers"] = (
            row["warmup_transfers"]
            + row["demand_transfers"]
            + row["prefetch_transfers"]
        )
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--capacity", type=int, required=True)
    p.add_argument("--transition-threshold", type=float, default=1.0)
    p.add_argument("--page-bytes", type=int, required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if args.capacity < 1 or args.page_bytes < 1:
        p.error("capacity and page-bytes must be positive")
    train, test = load(args.train), load(args.test)
    train_ids = {(r["task"], r["sequence"]) for r in train}
    test_ids = {(r["task"], r["sequence"]) for r in test}
    if train_ids & test_ids:
        p.error("train/test sequence leakage")
    if not train or not test:
        p.error("empty split")
    try:
        stats = evaluate(train, test, args.capacity, args.transition_threshold)
    except CapacityError as error:
        result = {
            "status": "infeasible_capacity",
            "physical_bytes_per_token": None,
            "measured_latency_ms": None,
            "capacity": args.capacity,
            "page_bytes": args.page_bytes,
            "cache_scope": "reset_each_sequence",
            "limitation": str(error),
        }
        Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2))
        return 2

    result = {"status": "trace_replay_only", "physical_bytes_per_token": None,
              "measured_latency_ms": None, "mean_kl": None,
              "top1_agreement": None, "free_running_accuracy": None,
              "detector_overhead_bytes": None,
              "warning": "Demand traces and oracle may use dense-model knowledge; no sparse quality or physical traffic was measured.",
              "capacity": args.capacity, "page_bytes": args.page_bytes,
              "cache_scope": "reset_each_sequence",
              "warmup_included_in_total_transfers": True,
              "oracle_label": "NONCAUSAL DIAGNOSTIC ORACLE",
              "policies": {}}
    for name, row in stats.items():
        tokens = max(1, row["tokens"])
        result["policies"][name] = {
            **row,
            "hit_rate": row["hits"] / max(1, row["requests"]),
            "modeled_demand_bytes_per_token": (
                row["demand_transfers"] * args.page_bytes / tokens
            ),
            "modeled_prefetch_bytes_per_token": (
                row["prefetch_transfers"] * args.page_bytes / tokens
            ),
            "modeled_warmup_bytes_total": row["warmup_transfers"] * args.page_bytes,
            "modeled_total_bytes_per_token": row["total_transfers"] * args.page_bytes / tokens,
            # Retained for consumers of the original field; corrected to include
            # warmup and prefetch traffic instead of demand misses alone.
            "modeled_cold_bytes_per_token": row["total_transfers"] * args.page_bytes / tokens,
        }
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
