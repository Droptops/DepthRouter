"""Enterprise working-set replay. No physical-HBM claims.

Input JSONL: {"task": str, "sequence": str, "step": int,
              "pages": [int, ...], "transition_signal": float}
Signal must be computed from causally available features BEFORE the demand.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def load(path):
    rows = [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    for row in rows:
        assert isinstance(row["pages"], list) and all(isinstance(p, int) for p in row["pages"])
        assert 0 <= float(row["transition_signal"])
    return rows


def evaluate(train, test, capacity, threshold):
    freq = Counter(p for r in train for p in set(r["pages"]))
    popular = [p for p, _ in sorted(freq.items(), key=lambda x: (-x[1], x[0]))]
    policies = ("static", "previous", "transition", "oracle")
    stats = {p: {"misses": 0, "requests": 0, "tokens": 0, "transitions": 0}
             for p in policies}
    streams = defaultdict(list)
    for r in test:
        streams[(r["task"], r["sequence"])].append(r)
    for key, stream in streams.items():
        stream.sort(key=lambda r: r["step"])
        caches = {"static": popular[:capacity], "previous": [],
                  "transition": [], "oracle": []}
        for i, row in enumerate(stream):
            requested = list(dict.fromkeys(row["pages"]))
            for policy in policies:
                cache = caches[policy]
                misses = [p for p in requested if p not in cache]
                stats[policy]["misses"] += len(misses)
                stats[policy]["requests"] += len(requested)
                stats[policy]["tokens"] += 1
                if policy == "static":
                    continue
                if policy == "previous":
                    candidates = requested + cache
                elif policy == "transition":
                    switched = float(row["transition_signal"]) >= threshold
                    stats[policy]["transitions"] += int(switched)
                    # Causal: signal is observed before demand; selection uses only
                    # current demanded pages AFTER faults, not future demand.
                    candidates = (requested + popular + cache) if switched else (requested + cache)
                else:
                    # Explicit look-ahead diagnostic, not a deployable policy.
                    future = stream[i + 1]["pages"] if i + 1 < len(stream) else []
                    candidates = requested + future + cache
                caches[policy] = list(dict.fromkeys(candidates))[:capacity]
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
    stats = evaluate(train, test, args.capacity, args.transition_threshold)
    result = {"status": "trace_replay_only", "physical_bytes_per_token": None,
              "measured_latency_ms": None, "mean_kl": None,
              "top1_agreement": None, "free_running_accuracy": None,
              "detector_overhead_bytes": None,
              "warning": "Demand traces and oracle may use dense-model knowledge; no sparse quality or physical traffic was measured.",
              "capacity": args.capacity, "page_bytes": args.page_bytes,
              "policies": {}}
    for name, row in stats.items():
        result["policies"][name] = {**row,
            "modeled_cold_bytes_per_token": row["misses"] * args.page_bytes / max(1, row["tokens"])}
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
