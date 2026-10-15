"""Fresh, deterministic prompts for the controlled output-gap experiment.

Only calibration and development may inform fitting or policy selection.  The two
held-out splits are sealed until that policy is frozen.  This generator never
loads model results. Numeric/content variants are deliberately repeated within a
single template family; this is a controlled synthetic distribution, not 1,536
independent natural-language templates.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

SPLIT_SIZES = {
    "calibration": 256,
    "development": 256,
    "heldout_matched": 512,
    "heldout_ood": 512,
}
MATCHED_TASKS = ("classification", "extraction", "numerical", "generation")
OOD_TASKS = ("finite_state", "substitution", "unit_conversion", "ordinal_lookup")
NEAR_DUPLICATE_THRESHOLD = 0.72
HISTORICAL_MANIFEST_PATHS = (
    "artifacts/predictive_working_set/20261013-validated-final/evaluation_manifest.jsonl",
    "artifacts/fused_sparse_swiglu/20261013-validated-final-v2/evaluation_manifest.jsonl",
    "data/photonic_parallel_manifest.jsonl",
)
_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?")
_TOPICS = (
    "coral",
    "moss",
    "pollen",
    "tides",
    "roots",
    "rain",
    "shells",
    "feathers",
    "sand",
    "leaves",
    "seeds",
    "clouds",
    "ice",
    "fungi",
    "nests",
    "rivers",
)
_SETTINGS = ("museum", "garden", "beach", "forest", "school", "island", "farm", "park")


def lexical_fingerprint(prompt: str) -> str:
    """Case/punctuation invariant text, with every numeric token replaced by NUM.

    The regex also splits identifiers such as ZX9 into their lexical and numeric
    components. Thus changing serials, whitespace, case, punctuation, or numbers
    cannot evade the cross-split/historical comparison.
    """
    return " ".join(
        "NUM" if token[0].isdigit() else token for token in _TOKEN_RE.findall(prompt.casefold())
    )


def _matched_example(split: str, task: str, index: int) -> tuple[str, str, str]:
    if task == "classification":
        if split == "calibration":
            temperature = index - 12
            threshold = temperature - 1 if index % 2 else temperature + 1
            return (
                (
                    f"A thermostat selects HOT when the room temperature exceeds its setting, "
                    f"and COOL otherwise. Room temperature: {temperature} degrees; "
                    f"setting: {threshold} degrees. Name the selected mode."
                ),
                "HOT" if temperature > threshold else "COOL",
                "thermostat_rule",
            )
        if split == "development":
            weight = 301 + index * 23
            capacity = weight + 5 if index % 2 else weight - 5
            return (
                (
                    f"Shipping policy assigns AIR to parcels weighing at most {capacity} grams; "
                    f"heavier parcels travel by SEA. A parcel weighs {weight} grams. "
                    "Which shipping class applies? Give its three-letter name."
                ),
                "AIR" if weight <= capacity else "SEA",
                "shipping_threshold",
            )
        lower = 130 + index * 7
        reading = lower + 3 if index % 2 else lower - 3
        return (
            (
                f"The permitted pressure band runs from {lower} through {lower + 10}, inclusive. "
                f"A gauge reads {reading}. Report INSIDE for a reading within the band, "
                "or OUTSIDE for any other reading."
            ),
            "INSIDE" if lower <= reading <= lower + 10 else "OUTSIDE",
            "pressure_band",
        )
    if task == "extraction":
        if split == "calibration":
            departure = f"{5 + index // 60:02d}:{index % 60:02d}"
            return (
                (
                    f"Travel bulletin: the train departs Elmwick at {departure} and arrives "
                    "in Bellhaven at 09:45. Copy the departure time without commentary."
                ),
                departure,
                "departure_time",
            )
        if split == "development":
            page = 41 + index * 3
            return (
                (
                    f"A reader marked page {page} in the biography and page {page + 17} in "
                    "the atlas. Which page was marked in the biography? Supply its digits."
                ),
                str(page),
                "book_page",
            )
        barrels = 203 + index * 11
        return (
            (
                f"Harbor dispatch says the northbound vessel carries {barrels} barrels of oil, "
                f"while the southbound vessel carries {barrels + 26} sacks of flour. "
                "How many barrels are aboard the northbound vessel? Copy that quantity."
            ),
            str(barrels),
            "cargo_quantity",
        )
    if task == "numerical":
        if split == "calibration":
            center = 71 + index * 4
            return (
                (
                    f"Three laboratory readings are {center + 8}, {center - 5}, and {center}. "
                    "What is their median? Write just its digits."
                ),
                str(center),
                "median_reading",
            )
        if split == "development":
            length = 31 + index * 2
            width = 7 + index % 13
            return (
                (
                    f"A rectangular vegetable plot measures {length} meters along its length "
                    f"and {width} meters across. How many meters of fencing surround it? "
                    "Omit the unit in your response."
                ),
                str(2 * (length + width)),
                "fence_perimeter",
            )
        old_price = 500 + index * 17
        reduction = 11 + index % 43
        return (
            (
                f"A shop reduces a bicycle's price from {old_price} coins to "
                f"{old_price - reduction} coins. State the absolute discount in coins "
                "using digits alone."
            ),
            str(reduction),
            "price_discount",
        )
    topic = _TOPICS[index % len(_TOPICS)]
    setting = _SETTINGS[index // len(_TOPICS)]
    if split == "calibration":
        return (
            (
                f"Write a child-friendly analogy about {topic} for a {setting} exhibit. "
                f"Your analogy must contain {topic} and fit within twelve words."
            ),
            topic,
            "nature_analogy",
        )
    if split == "development":
        return (
            (
                f"Draft a curious visitor's question concerning {topic} at a {setting}. "
                f"Include {topic}; keep the entire question below fifteen words."
            ),
            topic,
            "visitor_question",
        )
    return (
        (
            f"An illustrator is making a {setting} poster featuring {topic}. "
            f"Suggest a vivid caption of fewer than eighteen words that includes {topic}."
        ),
        topic,
        "nature_caption",
    )


def _ood_example(task: str, index: int) -> tuple[str, str, str]:
    if task == "finite_state":
        turns = 17 + index
        return (
            (
                f"A mechanical dial begins pointing NORTH. Each click rotates it clockwise "
                f"by one quarter turn. After {turns} clicks, which compass direction "
                "does it face? Answer NORTH, EAST, SOUTH, or WEST."
            ),
            ("NORTH", "EAST", "SOUTH", "WEST")[turns % 4],
            "quarter_turn_dial",
        )
    if task == "substitution":
        alphabet = "ABCD"
        raw = "".join(alphabet[(index // (4**place)) % 4] for place in (3, 2, 1, 0))
        return (
            (
                "A cipher replaces A with M, B with N, C with P, and D with R. "
                f"Encode {raw} by substituting each letter without changing its position. "
                "Provide the encoded four-letter sequence."
            ),
            raw.translate(str.maketrans("ABCD", "MNPR")),
            "letter_cipher",
        )
    if task == "unit_conversion":
        hours = 2 + index // 60
        minutes = index % 60
        return (
            (
                f"A documentary lasts {hours} hours and {minutes} minutes. "
                "Express its complete duration in minutes, giving only digits."
            ),
            str(hours * 60 + minutes),
            "duration_conversion",
        )
    numbers = [137 + index * 19 + offset * 7 for offset in range(6)]
    return (
        (
            f"Read this ordered inventory from left to right: {', '.join(map(str, numbers))}. "
            "Select the fourth entry, counting the first entry as one. Copy it alone."
        ),
        str(numbers[3]),
        "fourth_inventory_entry",
    )


def make_output_gap_manifest() -> list[dict[str, str]]:
    """Construct 1,536 fresh rows without consulting fitted weights or outcomes."""
    rows = []
    for split, size in SPLIT_SIZES.items():
        tasks = OOD_TASKS if split == "heldout_ood" else MATCHED_TASKS
        for task in tasks:
            for index in range(size // len(tasks)):
                prompt, answer, family = (
                    _ood_example(task, index)
                    if split == "heldout_ood"
                    else _matched_example(split, task, index)
                )
                rows.append(
                    {
                        "id": f"og-{split}-{task}-{index:04d}",
                        "split": split,
                        "task": task,
                        "template_family": family,
                        "prompt": prompt,
                        "answer": answer,
                        "judge": "contains" if task == "generation" else "exact",
                    }
                )
    return rows


def load_historical_prompts(repo: Path) -> tuple[list[str], list[dict[str, object]]]:
    """Read only the three registered, existing prompt manifests and hash their bytes.

    A missing source is an error, not silent permission to weaken the exclusion
    check. Historical result files and saved predictions are never read.
    """
    prompts: list[str] = []
    audit: list[dict[str, object]] = []
    for relative_path in HISTORICAL_MANIFEST_PATHS:
        raw = (repo / relative_path).read_bytes()
        rows = [json.loads(line) for line in raw.decode("utf-8-sig").splitlines() if line]
        source_prompts = [row["prompt"] for row in rows]
        if not all(isinstance(prompt, str) and prompt.strip() for prompt in source_prompts):
            raise ValueError(f"invalid historical prompt in {relative_path}")
        prompts.extend(source_prompts)
        audit.append(
            {
                "path": relative_path,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "rows": len(rows),
                "unique_prompts": len(set(source_prompts)),
            }
        )
    return list(dict.fromkeys(prompts)), audit


def _near_duplicate_audit(
    left: dict[str, str], right: dict[str, str], *, label: str
) -> dict[str, object]:
    maximum = 0.0
    witness = None
    left_sets = [(text, set(text.split()), row_id) for text, row_id in left.items()]
    right_sets = [(text, set(text.split()), row_id) for text, row_id in right.items()]
    for left_text, left_tokens, left_id in left_sets:
        for right_text, right_tokens, right_id in right_sets:
            union = left_tokens | right_tokens
            similarity = len(left_tokens & right_tokens) / len(union) if union else 1.0
            if similarity > maximum:
                maximum = similarity
                witness = [left_id, right_id]
            if left_text == right_text or similarity >= NEAR_DUPLICATE_THRESHOLD:
                raise ValueError(
                    f"near-duplicate {label}: {left_id} / {right_id}; "
                    f"normalized lexical Jaccard={similarity:.6f}"
                )
    return {
        "comparison": label,
        "max_jaccard": maximum,
        "witness_ids": witness,
        "fingerprint_pairs": len(left) * len(right),
    }


def validate_output_gap_manifest(
    rows: Sequence[dict[str, str]], *, historical_prompts: Sequence[str] = ()
) -> dict[str, object]:
    """Check exact identities, family isolation, balance, and lexical exclusion.

    Cross-split and historical pairs with identical normalized text OR token-set
    Jaccard >= 0.72 are rejected. We collapse repeated fingerprints only to save
    comparison work; all prompts still enter the check. Repeated fingerprints
    are allowed within a single (split, family), including numeric variants.
    This lexical guard does not establish semantic independence.
    """
    required = {"id", "split", "task", "template_family", "prompt", "answer", "judge"}
    if any(not required <= row.keys() for row in rows):
        raise ValueError("each manifest row requires the seven standard fields")
    if any(
        not isinstance(row[key], str) or not row[key].strip() for row in rows for key in required
    ):
        raise ValueError("manifest fields must be nonempty strings")
    ids = [row["id"] for row in rows]
    prompts = [row["prompt"] for row in rows]
    if len(ids) != len(set(ids)) or len(prompts) != len(set(prompts)):
        raise ValueError("manifest ids and prompts must be unique")
    if {row["split"] for row in rows} != set(SPLIT_SIZES):
        raise ValueError("manifest contains unknown or missing splits")
    if any(row["judge"] not in {"exact", "contains"} for row in rows):
        raise ValueError("unsupported answer judge")
    families: dict[str, str] = {}
    fingerprints: dict[str, dict[str, str]] = {split: {} for split in SPLIT_SIZES}
    fingerprint_families: dict[str, tuple[str, str]] = {}
    task_counts: dict[str, dict[str, int]] = {}
    for split, expected in SPLIT_SIZES.items():
        selected = [row for row in rows if row["split"] == split]
        expected_tasks = OOD_TASKS if split == "heldout_ood" else MATCHED_TASKS
        counts = Counter(row["task"] for row in selected)
        if len(selected) != expected:
            raise ValueError(f"{split} must contain {expected} rows")
        if counts != Counter({task: expected // 4 for task in expected_tasks}):
            raise ValueError(f"{split} must contain its four expected, equally balanced tasks")
        task_counts[split] = dict(sorted(counts.items()))
        for row in selected:
            family = row["template_family"]
            if family in families and families[family] != split:
                raise ValueError("template families must not cross splits")
            families[family] = split
            fingerprint = lexical_fingerprint(row["prompt"])
            previous = fingerprint_families.get(fingerprint)
            if previous is not None and previous != (split, family):
                raise ValueError("normalized fingerprint must not cross splits or families")
            fingerprint_families[fingerprint] = (split, family)
            fingerprints[split].setdefault(fingerprint, row["id"])
    historical = {
        lexical_fingerprint(prompt): f"historical-{index}"
        for index, prompt in enumerate(historical_prompts)
    }
    checks = []
    split_names = list(SPLIT_SIZES)
    for index, split in enumerate(split_names):
        for other in split_names[index + 1 :]:
            checks.append(
                _near_duplicate_audit(
                    fingerprints[split], fingerprints[other], label=f"{split} vs {other}"
                )
            )
        checks.append(
            _near_duplicate_audit(fingerprints[split], historical, label=f"{split} vs historical")
        )
    canonical = json.dumps(list(rows), sort_keys=True, separators=(",", ":")).encode()
    return {
        "manifest_sha256": hashlib.sha256(canonical).hexdigest(),
        "rows": len(rows),
        "split_sizes": dict(SPLIT_SIZES),
        "task_counts": task_counts,
        "template_families": sorted(families),
        "unique_normalized_fingerprints": {s: len(p) for s, p in fingerprints.items()},
        "historical_prompts": len(historical_prompts),
        "historical_unique_fingerprints": len(historical),
        "near_duplicate_threshold": NEAR_DUPLICATE_THRESHOLD,
        "near_duplicate_metric": "casefolded numeric-normalized lexical token-set Jaccard",
        "within_family_numeric_variants_allowed": True,
        "independence_scope": "lexical exclusion; no claim of semantic independence",
        "comparisons": checks,
    }
