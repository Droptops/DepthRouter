"""Paired, split-controlled falsification of prompt-structure co-demand.

All demand sets are dense-oracle-derived. This does NOT measure HBM traffic,
residency, metadata cost, or deployable selective inference.
"""
from __future__ import annotations

import argparse
import json
import random
import runpy
import sys
from pathlib import Path

SUFFIX = (
    "\n\nFinal instruction for all tasks: Select the better option, "
    "then give exactly one concise reason."
)
SOURCE = Path(__file__).with_name("hf_prompt_structure_codemand.py")


def average(values: list[float]) -> float:
    return sum(values) / len(values)


def run_arm(
    *,
    arm: str,
    split_seed: int,
    output: Path,
    max_length: int,
    permutations: int,
) -> dict:
    namespace = runpy.run_path(str(SOURCE), run_name="depthrouter_paired_control")
    if split_seed:
        random.Random(split_seed).shuffle(namespace["SEMANTIC_CASES"])
    if arm == "shared_suffix":
        namespace["STRUCTURES"].update(
            {name: template + SUFFIX for name, template in namespace["STRUCTURES"].items()}
        )

    # Preflight full, untruncated prompts. Otherwise the shared suffix might
    # be clipped from some structures, invalidating the intended control.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    lengths = [
        len(tokenizer.encode(template.format(**semantic)))
        for semantic in namespace["SEMANTIC_CASES"]
        for template in namespace["STRUCTURES"].values()
    ]
    if max(lengths) > max_length:
        raise ValueError(
            f"{arm}/split={split_seed}: max prompt length {max(lengths)} "
            f"exceeds max_length={max_length}; refusing truncated control"
        )

    old_argv = sys.argv
    try:
        sys.argv = [
            str(SOURCE),
            "--model", "Qwen/Qwen2.5-0.5B-Instruct",
            "--device", "cpu",
            "--dtype", "float32",
            "--max-length", str(max_length),
            "--top-logits", "128",
            "--fraction", "0.30",
            "--block-size", "32",
            "--train-semantics", "6",
            "--prefetch-budgets", "0.10", "0.20", "0.30",
            "--permutations", str(permutations),
            "--seed", "1103",
            "--json-out", str(output),
        ]
        namespace["main"]()
    finally:
        sys.argv = old_argv

    data = json.loads(output.read_text(encoding="utf-8"))
    data["arm"] = arm
    data["split_seed"] = split_seed
    data["max_prompt_tokens"] = max(lengths)
    data["max_length"] = max_length
    data["truncated_prompts"] = 0
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-seeds", nargs="+", type=int, default=[0, 17, 29])
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--output-dir", default="paired_codemand")
    parser.add_argument("--json-out", default="paired_codemand_summary.json")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = []
    for split_seed in args.split_seeds:
        results = {}
        for arm in ("original", "shared_suffix"):
            result = run_arm(
                arm=arm,
                split_seed=split_seed,
                output=out_dir / f"{arm}_split_{split_seed}.json",
                max_length=args.max_length,
                permutations=args.permutations,
            )
            results[arm] = result
        budgets = []
        for original, suffix in zip(
            results["original"]["heldout_structure_prefetch"],
            results["shared_suffix"]["heldout_structure_prefetch"],
            strict=True,
        ):
            assert original["budget_fraction"] == suffix["budget_fraction"]
            budgets.append(
                {
                    "budget_fraction": original["budget_fraction"],
                    "original_lift": original["structure_recall_lift_over_global"],
                    "shared_suffix_lift": suffix["structure_recall_lift_over_global"],
                    "suffix_minus_original_lift": (
                        suffix["structure_recall_lift_over_global"]
                        - original["structure_recall_lift_over_global"]
                    ),
                    "original_permutation_p": original["structure_label_permutation_p_value"],
                    "shared_suffix_permutation_p": suffix[
                        "structure_label_permutation_p_value"
                    ],
                    "original_residual_fault_fraction": original[
                        "structure_residual_fault_fraction_of_full_width"
                    ],
                    "shared_suffix_residual_fault_fraction": suffix[
                        "structure_residual_fault_fraction_of_full_width"
                    ],
                }
            )
        pairs.append(
            {
                "split_seed": split_seed,
                "max_prompt_tokens": max(
                    results["original"]["max_prompt_tokens"],
                    results["shared_suffix"]["max_prompt_tokens"],
                ),
                "original_quality": results["original"]["oracle_quality"],
                "shared_suffix_quality": results["shared_suffix"]["oracle_quality"],
                "budgets": budgets,
            }
        )

    budget_summaries = []
    for i in range(len(pairs[0]["budgets"])):
        entries = [pair["budgets"][i] for pair in pairs]
        budget_summaries.append(
            {
                "budget_fraction": entries[0]["budget_fraction"],
                "mean_original_lift": average([x["original_lift"] for x in entries]),
                "mean_shared_suffix_lift": average([x["shared_suffix_lift"] for x in entries]),
                "mean_paired_delta": average([x["suffix_minus_original_lift"] for x in entries]),
                "shared_suffix_positive_splits": sum(
                    x["shared_suffix_lift"] > 0 for x in entries
                ),
                "shared_suffix_significant_splits": sum(
                    x["shared_suffix_permutation_p"] < 0.05 for x in entries
                ),
            }
        )
    result = {
        "experiment": "prompt_structure_paired_common_suffix_v1",
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "split_seeds": args.split_seeds,
        "pairs": pairs,
        "budget_summaries": budget_summaries,
        "metadata_fraction": None,
        "measured_hbm_bytes": None,
        "measured_residency": None,
        "claim_boundary": (
            "Oracle final-layer neuron demand, paired across prompt endings and "
            "held-out semantic splits. Cold-fault fractions are analytical set "
            "estimates only. No physical HBM, real residency, or end-to-end decode "
            "measurement; p-values are exploratory across multiple budgets/splits."
        ),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    Path(args.json_out).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
