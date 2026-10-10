"""Controlled address-head training-data scaling experiment.

Uses the existing held-out diagnostic as the measurement implementation.
No synthetic metrics and no hardware-performance claims.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-prompts", type=int, nargs="+", default=[4, 12, 20])
    parser.add_argument("--test-prompts", type=int, default=20)
    parser.add_argument("--train-steps", type=int, default=300)
    parser.add_argument("--test-decode-steps", type=int, default=10)
    parser.add_argument("--json-out", default="address_learning_curve.json")
    args = parser.parse_args()
    if not args.train_prompts or min(args.train_prompts) < 1:
        parser.error("training prompt counts must be positive")
    runs = []
    for count in args.train_prompts:
        path = Path(f"address_learning_curve_train_{count}.json")
        command = [
            sys.executable, str(Path(__file__).with_name("hf_temporal_tangent_exposure_diagnostic.py")),
            "--max-train-prompts", str(count),
            "--max-test-prompts", str(args.test_prompts),
            "--test-decode-steps", str(args.test_decode_steps),
            "--steps", str(args.train_steps),
            "--json-out", str(path),
        ]
        subprocess.run(command, check=True)
        result = json.loads(path.read_text(encoding="utf-8"))
        metrics = result["diagnostic_all_steps"]
        runs.append({
            "train_prompts": count,
            "train_states": result["train_states"],
            "test_states": result["test_states"],
            "student_history": metrics["student_history"],
            "oracle_history": metrics["oracle_history"],
            "oracle_selection": metrics["oracle_selection"],
            "cache_rows": result["cache_rows"],
        })
    output = {
        "experiment": "address_learning_curve_v1",
        "controls": "same test prompts, seed, model, layer, fraction and architecture across runs",
        "limitations": "dense-generated prefixes; no free-running sparse decode or physical DRAM profiling",
        "runs": runs,
    }
    Path(args.json_out).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
