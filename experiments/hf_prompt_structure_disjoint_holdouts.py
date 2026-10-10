"""Two disjoint semantic holdouts for the PR #12 neutral-suffix hypothesis.

This is cross-validation on ten fixed tasks, not independent external replication.
Dense oracle working sets only: no physical HBM or residency measurement.
"""
from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

SOURCE = Path(__file__).with_name("hf_prompt_structure_codemand.py")
PAIRED = Path(__file__).with_name("hf_prompt_structure_paired_control.py")
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def run_arm(
    arm: str,
    train_ids: list[str],
    test_ids: list[str],
    suffix: str,
    tokenizer: object,
    max_length: int,
    permutations: int,
    output: Path,
) -> dict:
    namespace = runpy.run_path(str(SOURCE), run_name="disjoint_holdout_import")
    by_id = {case["id"]: case for case in namespace["SEMANTIC_CASES"]}
    if set(train_ids) & set(test_ids):
        raise ValueError("Train/test semantic overlap")
    if set(train_ids + test_ids) != set(by_id):
        raise ValueError("Train/test must partition all semantic cases")
    namespace["SEMANTIC_CASES"][:] = [by_id[i] for i in train_ids + test_ids]
    if arm == "shared_suffix":
        namespace["STRUCTURES"].update(
            {k: v + suffix for k, v in namespace["STRUCTURES"].items()}
        )
    lengths = [
        len(tokenizer.encode(template.format(**case)))
        for case in namespace["SEMANTIC_CASES"]
        for template in namespace["STRUCTURES"].values()
    ]
    if max(lengths) > max_length:
        raise ValueError(
            f"{arm}: prompt length {max(lengths)} exceeds {max_length}; "
            "truncation would invalidate this control"
        )
    argv = sys.argv
    try:
        sys.argv = [
            str(SOURCE), "--model", MODEL, "--device", "cpu",
            "--dtype", "float32", "--max-length", str(max_length),
            "--top-logits", "128", "--fraction", "0.30",
            "--block-size", "32", "--train-semantics", str(len(train_ids)),
            "--prefetch-budgets", "0.10", "0.20", "0.30",
            "--permutations", str(permutations), "--seed", "1103",
            "--json-out", str(output),
        ]
        namespace["main"]()
    finally:
        sys.argv = argv
    result = json.loads(output.read_text(encoding="utf-8"))
    result["arm"] = arm
    result["train_semantic_ids"] = train_ids
    result["test_semantic_ids"] = test_ids
    result["max_prompt_tokens"] = max(lengths)
    result["truncated_prompts"] = 0
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--output-dir", default="disjoint_holdout_outputs")
    parser.add_argument("--json-out", default="disjoint_holdout_summary.json")
    args = parser.parse_args()
    from transformers import AutoTokenizer

    suffix = runpy.run_path(str(PAIRED), run_name="paired_constant_import")["SUFFIX"]
    source = runpy.run_path(str(SOURCE), run_name="semantic_cases_import")
    ids = [case["id"] for case in source["SEMANTIC_CASES"]]
    if len(ids) != 10 or len(set(ids)) != 10:
        raise ValueError("Expected exactly ten unique semantic cases")
    folds = [(ids[:5], ids[5:]), (ids[5:], ids[:5])]
    assert set(folds[0][1]).isdisjoint(folds[1][1])
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_results = []
    for fold, (train_ids, test_ids) in enumerate(folds):
        arms = {
            arm: run_arm(
                arm, train_ids, test_ids, suffix, tokenizer,
                args.max_length, args.permutations,
                out_dir / f"fold_{fold}_{arm}.json",
            )
            for arm in ("original", "shared_suffix")
        }
        budgets = []
        for original, suffix_arm in zip(
            arms["original"]["heldout_structure_prefetch"],
            arms["shared_suffix"]["heldout_structure_prefetch"],
            strict=True,
        ):
            if original["budget_fraction"] != suffix_arm["budget_fraction"]:
                raise ValueError("Budget mismatch")
            budgets.append({
                "budget_fraction": original["budget_fraction"],
                "original_lift": original["structure_recall_lift_over_global"],
                "shared_suffix_lift": suffix_arm["structure_recall_lift_over_global"],
                "paired_delta": (
                    suffix_arm["structure_recall_lift_over_global"]
                    - original["structure_recall_lift_over_global"]
                ),
                "original_p": original["structure_label_permutation_p_value"],
                "shared_suffix_p": suffix_arm["structure_label_permutation_p_value"],
            })
        fold_results.append({
            "fold": fold,
            "train_semantic_ids": train_ids,
            "test_semantic_ids": test_ids,
            "max_prompt_tokens": max(
                arm["max_prompt_tokens"] for arm in arms.values()
            ),
            "oracle_quality": {
                arm: result["oracle_quality"] for arm, result in arms.items()
            },
            "budgets": budgets,
        })
    primary = [fold["budgets"][-1] for fold in fold_results]
    result = {
        "experiment": "prompt_structure_disjoint_holdouts_v1",
        "model": MODEL,
        "suffix": suffix,
        "folds": fold_results,
        "primary_budget_fraction": 0.30,
        "primary_mean_shared_suffix_lift": sum(
            b["shared_suffix_lift"] for b in primary
        ) / len(primary),
        "stable_positive_lift_both_folds": all(
            b["shared_suffix_lift"] > 0 for b in primary
        ),
        "shared_suffix_p_below_0_05_both_folds": all(
            b["shared_suffix_p"] < 0.05 for b in primary
        ),
        "metadata_bytes": None,
        "measured_hbm_bytes": None,
        "measured_residency": None,
        "claim_boundary": (
            "Two complementary held-out semantic folds on the same ten tasks. "
            "Not independent external replication. Oracle-derived neuron sets "
            "and analytical prefetch recall only; no sparse hardware execution."
        ),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    Path(args.json_out).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
