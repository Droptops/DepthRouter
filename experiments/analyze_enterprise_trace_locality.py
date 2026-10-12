"""Analyze held-out page-set locality with a prompt-disjoint control.

This consumes logical page-demand traces. It does not measure physical traffic
and does not turn the dense-model diagnostic signal into a deployable policy.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def atomic_write_json(path, payload):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


def page_metrics(left, right):
    left_pages, right_pages = set(left), set(right)
    intersection = len(left_pages & right_pages)
    union = len(left_pages | right_pages)
    return {
        "jaccard": intersection / union if union else 1.0,
        "retention": intersection / len(left_pages) if left_pages else 1.0,
        "new_page_fraction": len(right_pages - left_pages) / len(right_pages)
        if right_pages else 0.0,
        "new_pages": len(right_pages - left_pages),
    }


def mean(rows, key):
    return statistics.fmean(row[key] for row in rows) if rows else None


def pearson(left, right):
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean, right_mean = statistics.fmean(left), statistics.fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_energy = sum((x - left_mean) ** 2 for x in left)
    right_energy = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_energy * right_energy)
    return numerator / denominator if denominator else None


def analyze(rows):
    if not rows:
        raise ValueError("trace is empty")
    required = {"task", "sequence", "step", "pages", "transition_signal"}
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"row {index} is missing {sorted(missing)}")

    sequences = defaultdict(list)
    matched_steps = defaultdict(list)
    for row in rows:
        sequences[(row["task"], row["sequence"])].append(row)
        matched_steps[(row["task"], row["step"])].append(row)

    adjacent = []
    predictor_signals = []
    predictor_targets = []
    for sequence_rows in sequences.values():
        ordered = sorted(sequence_rows, key=lambda row: row["step"])
        for previous, current in itertools.pairwise(ordered):
            metrics = page_metrics(previous["pages"], current["pages"])
            adjacent.append(metrics)
            predictor_signals.append(float(current["transition_signal"]))
            predictor_targets.append(metrics["new_page_fraction"])

    # Prompt-disjoint control: compare different sequences from the same task at
    # the same decode step. Exhaustive pairs make this deterministic.
    controls = []
    for step_rows in matched_steps.values():
        for left_index, left in enumerate(step_rows):
            for right in step_rows[left_index + 1:]:
                if left["sequence"] != right["sequence"]:
                    controls.append(page_metrics(left["pages"], right["pages"]))

    adjacent_jaccard = mean(adjacent, "jaccard")
    control_jaccard = mean(controls, "jaccard")
    return {
        "rows": len(rows),
        "sequences": len(sequences),
        "adjacent_pairs": len(adjacent),
        "same_task_same_step_control_pairs": len(controls),
        "adjacent": {
            "mean_jaccard": adjacent_jaccard,
            "mean_retention": mean(adjacent, "retention"),
            "mean_new_page_fraction": mean(adjacent, "new_page_fraction"),
            "mean_new_pages": mean(adjacent, "new_pages"),
        },
        "between_sequence_control": {
            "mean_jaccard": control_jaccard,
            "mean_retention": mean(controls, "retention"),
            "mean_new_page_fraction": mean(controls, "new_page_fraction"),
            "mean_new_pages": mean(controls, "new_pages"),
        },
        "adjacent_jaccard_lift": (
            adjacent_jaccard - control_jaccard
            if adjacent_jaccard is not None and control_jaccard is not None else None
        ),
        "lagged_signal_vs_current_new_page_fraction_pearson": pearson(
            predictor_signals, predictor_targets
        ),
        "claim_boundary": (
            "Logical held-out page-set analysis only; no physical bytes, causal "
            "selector quality, or selective-kernel latency is inferred."
        ),
        "physical_bytes_per_token": None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = analyze(read_jsonl(args.trace))
    result["trace"] = str(Path(args.trace))
    atomic_write_json(args.output, result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
