"""Test whether the low-rank address bottleneck explains poor oracle recall.

Each arm uses identical training and held-out prompts, steps, and seed.
Rank changes predictor capacity and metadata; it does not add privileged inputs.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranks", nargs="+", type=int, default=[8, 16, 64])
    parser.add_argument("--train-prompts", type=int, default=20)
    parser.add_argument("--test-prompts", type=int, default=20)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--json-out", default="address_rank_capacity.json")
    args = parser.parse_args()
    if not args.ranks or min(args.ranks) <= 0:
        parser.error("ranks must be positive")
    runs = []
    for rank in args.ranks:
        path = Path(f"address_rank_{rank}.json")
        subprocess.run([
            sys.executable,
            str(Path(__file__).with_name("hf_temporal_tangent_exposure_diagnostic.py")),
            "--rank", str(rank),
            "--max-train-prompts", str(args.train_prompts),
            "--max-test-prompts", str(args.test_prompts),
            "--steps", str(args.steps),
            "--json-out", str(path),
        ], check=True)
        measured = json.loads(path.read_text(encoding="utf-8"))
        runs.append({
            "rank": rank,
            "train_states": measured["train_states"],
            "test_states": measured["test_states"],
            "address_head_params": measured["address_head_params"],
            "student_history": measured["diagnostic_all_steps"]["student_history"],
            "oracle_history": measured["diagnostic_all_steps"]["oracle_history"],
            "oracle_selection": measured["diagnostic_all_steps"]["oracle_selection"],
            "cache_rows": measured["cache_rows"],
        })
    payload = {
        "experiment": "address_rank_capacity_v1",
        "question": "Is the learned predictor bottlenecked by rank rather than training set size?",
        "controls": "fixed train/test prompts, seed, optimizer steps, active fraction, model and layer",
        "pass_gate": {"mean_kl_nats_lt": 0.01, "top1_agreement_gte": 0.99,
                      "metadata_plus_cold_fraction_lte": 0.25},
        "caveat": "Estimated ideal neuron-granular cold transfers; no hardware profiling or sparse rollout.",
        "runs": runs,
    }
    rendered = json.dumps(payload, indent=2)
    Path(args.json_out).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
