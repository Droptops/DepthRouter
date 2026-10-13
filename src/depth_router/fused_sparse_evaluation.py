"""Deterministic development and held-out data for fused sparse SwiGLU work."""

from __future__ import annotations

from collections.abc import Sequence


SPLIT_SIZES = {
    "calibration": 256,
    "development": 256,
    "heldout_matched": 512,
    "heldout_ood": 512,
}


def _matched_example(split: str, task: str, index: int) -> tuple[str, str, str, str]:
    salt = {"calibration": 19, "development": 43, "heldout_matched": 71}[split]
    if task == "classification":
        value = 2003 + index * salt
        divisor = {"calibration": 5, "development": 7, "heldout_matched": 11}[split]
        answer = "YES" if value % divisor == 0 else "NO"
        return (
            f"Determine if {value} is divisible by {divisor}. Answer exactly YES or NO.",
            answer,
            f"divisible_{divisor}",
            "exact",
        )
    if task == "extraction":
        code = f"{(index * salt + 701) % 100000:05d}"
        wrappers = {
            "calibration": f"Packet src=alpha payload={{token:{code}}} end. Return only token.",
            "development": f"Ledger row {index}: owner|nova; secret[{code}]; state|valid. Give secret.",
            "heldout_matched": f"Telemetry <zone=red><key={code}><ok=true>. Output the key only.",
        }
        return wrappers[split], code, f"extract_{split}", "exact"
    if task == "numerical":
        left = 31 + (index * salt) % 137
        right = 4 + (index * 9) % 29
        if split == "calibration":
            answer = left + 2 * right
            prompt = f"Calculate {left} + 2 * {right}. Return only the integer."
        elif split == "development":
            answer = 3 * left - right
            prompt = f"Calculate 3 * {left} - {right}. Return only the integer."
        else:
            answer = 2 * (left + right)
            prompt = f"Calculate twice the sum of {left} and {right}. Return only the integer."
        return prompt, str(answer), f"arithmetic_{split}", "exact"
    topics = ("cache", "kernel", "bandwidth", "latency", "routing", "precision", "memory", "batch")
    topic = topics[index % len(topics)]
    prompts = {
        "calibration": f"Case {index}: in at most 14 words, state one benefit of {topic}. Include the word {topic}.",
        "development": f"Case {index}: write one sentence under 16 words warning about {topic}. Use the term {topic}.",
        "heldout_matched": f"Case {index}: give a falsifiable claim about {topic} in at most 18 words and mention {topic}.",
    }
    return prompts[split], topic, f"constrained_{split}", "contains"


def _ood_example(task: str, index: int) -> tuple[str, str, str, str]:
    if task == "boolean_logic":
        a = index % 2 == 0
        b = index % 3 == 0
        answer = "TRUE" if (a and not b) or (b and not a) else "FALSE"
        return (
            f"Logic case {index}: let A be {str(a).upper()} and B be {str(b).upper()}. Evaluate A XOR B. Reply TRUE or FALSE.",
            answer,
            "boolean_xor",
            "exact",
        )
    if task == "string_transform":
        letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
        raw = "".join(letters[(index * 5 + offset * 7) % len(letters)] for offset in range(5))
        return (
            f"String case {index}: reverse the five-character string {raw}. Return only the reversed string.",
            raw[::-1],
            "reverse_string",
            "exact",
        )
    if task == "comparison":
        left = (index * 97 + 503) % 1009
        right = (index * 53 + 211) % 1009
        answer = "LEFT" if left > right else "RIGHT" if right > left else "EQUAL"
        return (
            f"Which value is larger, left={left} or right={right}? Reply LEFT, RIGHT, or EQUAL.",
            answer,
            "integer_comparison",
            "exact",
        )
    base = 17 + index * 13
    modulus = 7 + index % 5
    return (
        f"Compute the remainder when {base} squared is divided by {modulus}. Return only the integer.",
        str((base * base) % modulus),
        "modular_square",
        "exact",
    )


def make_fused_sparse_manifest() -> list[dict[str, str]]:
    """Return a new, fixed, template-isolated 1,536-example evaluation set."""

    rows: list[dict[str, str]] = []
    matched_tasks = ("classification", "extraction", "numerical", "generation")
    ood_tasks = ("boolean_logic", "string_transform", "comparison", "modular_arithmetic")
    for split in ("calibration", "development", "heldout_matched"):
        per_task = SPLIT_SIZES[split] // len(matched_tasks)
        for task in matched_tasks:
            for index in range(per_task):
                prompt, answer, family, judge = _matched_example(split, task, index)
                rows.append(
                    {
                        "id": f"fs-{split[:3]}-{task[:3]}-{index:04d}",
                        "split": split,
                        "task": task,
                        "template_family": family,
                        "prompt": prompt,
                        "answer": answer,
                        "judge": judge,
                    }
                )
    per_task = SPLIT_SIZES["heldout_ood"] // len(ood_tasks)
    for task in ood_tasks:
        for index in range(per_task):
            prompt, answer, family, judge = _ood_example(task, index)
            rows.append(
                {
                    "id": f"fs-ood-{task[:3]}-{index:04d}",
                    "split": "heldout_ood",
                    "task": task,
                    "template_family": family,
                    "prompt": prompt,
                    "answer": answer,
                    "judge": judge,
                }
            )
    return rows


def validate_fused_sparse_manifest(
    rows: Sequence[dict[str, str]],
    *,
    historical_prompts: Sequence[str] = (),
) -> None:
    ids = [row["id"] for row in rows]
    prompts = [row["prompt"] for row in rows]
    if len(ids) != len(set(ids)) or len(prompts) != len(set(prompts)):
        raise ValueError("manifest ids and prompts must be unique")
    for split, expected in SPLIT_SIZES.items():
        split_rows = [row for row in rows if row["split"] == split]
        if len(split_rows) != expected:
            raise ValueError(f"{split} must contain {expected} rows")
        counts: dict[str, int] = {}
        for row in split_rows:
            counts[row["task"]] = counts.get(row["task"], 0) + 1
        if len(set(counts.values())) != 1:
            raise ValueError(f"{split} must be task balanced")
    families = {
        split: {row["template_family"] for row in rows if row["split"] == split}
        for split in SPLIT_SIZES
    }
    names = list(SPLIT_SIZES)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if families[left] & families[right]:
                raise ValueError("template families must not cross splits")
    overlap = set(prompts) & set(historical_prompts)
    if overlap:
        raise ValueError("new evaluation prompts overlap the historical 512-example set")


def answer_success(text: str, row: dict[str, str]) -> bool:
    normalized = " ".join(text.strip().split())
    expected = row["answer"].strip()
    if row["judge"] == "exact":
        first = normalized.split()[0].strip(".,:;!?`'\"") if normalized else ""
        return first.casefold() == expected.casefold()
    return expected.casefold() in normalized.casefold() and len(normalized.split()) <= 18
