"""Fresh, prompt-only data for a fixed-partition causal recognition study.

This reuses the previous study's pinned human-written source and exclusion
definitions, not its selected contexts. Confirmation data is never fitting data.
The enterprise screen is topical and lexical, not a production-workload claim.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from depth_router import expert_audit_data as source

SELECTION_SALT = "DepthRouter-fixed-partition-recognition-data-v1-20261014"
PRIOR_EXPERT_MANIFEST = "data/expert_audit/evaluation_manifest.jsonl"
SPLIT_SIZES = {
    "confirmation": 128,
    "calibration": 768,
    "development": 128,
    "heldout_matched": 512,
    "heldout_ood": 192,
}


def _check_sizes(sizes: dict[str, int]) -> None:
    if set(sizes) != set(SPLIT_SIZES) or any(
        not isinstance(size, int) or isinstance(size, bool) or size <= 0 or size % 8
        for size in sizes.values()
    ) or sizes["confirmation"] % 16:
        raise ValueError("five split sizes must be positive multiples of eight; confirmation of 16")


def load_recognition_exclusions(repo: Path) -> tuple[list[str], list[dict], list[dict]]:
    """Require all registered prior manifests; preserve prompt/source provenance."""
    historical, audits = source.load_historical_prompts(repo)
    raw = (repo / PRIOR_EXPERT_MANIFEST).read_bytes()
    prior = [json.loads(line) for line in raw.decode("utf-8-sig").splitlines() if line]
    if len(prior) != 2304:
        raise ValueError("the complete prior 2304-context expert manifest is required")
    prior_audit = json.loads((repo / "data/expert_audit/data_audit.json").read_text())
    canonical = json.dumps(prior, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(canonical).hexdigest() != prior_audit["manifest_sha256"]:
        raise ValueError("prior expert manifest does not match its frozen audit")
    audits.append({"path": PRIOR_EXPERT_MANIFEST, "sha256": hashlib.sha256(raw).hexdigest(),
                   "rows": len(prior), "unique_prompts": len({row["prompt"] for row in prior}),
                   "canonical_sha256": prior_audit["manifest_sha256"]})
    return historical, prior, audits


def _source_candidates(path: Path) -> list[dict[str, str]]:
    candidates = []
    for row in source._source_candidates(path):
        candidates.append(dict(row, id=f"er-dolly-{int(row['source_row']):05d}",
                               confirmation_stratum="not_confirmation"))
    return sorted(candidates, key=lambda row: source._sha(SELECTION_SALT + row["id"]))


def _prior_features(prior_expert_rows: Sequence[dict]) -> tuple[set, set, list]:
    return (
        {row["source_document_id"] for row in prior_expert_rows},
        {row["template_family"] for row in prior_expert_rows},
        [source._features(row) for row in prior_expert_rows],
    )


def _historical_conflict(feature: tuple, historical: Sequence[frozenset[str]]) -> bool:
    return any(source._jaccard(feature[0], old) >= source.NEAR_DUPLICATE_THRESHOLD
               or source._jaccard(feature[1], old) >= source.NEAR_DUPLICATE_THRESHOLD
               for old in historical)


def _allocations(sizes: dict[str, int]) -> list[tuple[str, str, int, tuple[str, ...]]]:
    # Reserve the scarcer OOD strata first. The order is frozen, not model-selected.
    return [
        ("confirmation", "ood", sizes["confirmation"] // 2, source.OOD_CATEGORIES),
        ("heldout_ood", "not_confirmation", sizes["heldout_ood"], source.OOD_CATEGORIES),
        ("confirmation", "matched", sizes["confirmation"] // 2, source.MATCHED_CATEGORIES),
        ("calibration", "not_confirmation", sizes["calibration"], source.MATCHED_CATEGORIES),
        ("development", "not_confirmation", sizes["development"], source.MATCHED_CATEGORIES),
        ("heldout_matched", "not_confirmation", sizes["heldout_matched"], source.MATCHED_CATEGORIES),
    ]


def audit_enterprise_source_capacity(
    source_path: Path, *, historical_prompts: Sequence[str], prior_expert_rows: Sequence[dict],
) -> dict[str, object]:
    """Explain the prospective budget reduction using source text alone."""
    candidates = [row for row in _source_candidates(source_path)
                  if row["stratum"] == "enterprise_related"]
    historical = list({source._tokens(prompt) for prompt in historical_prompts})
    old_documents, old_families, old_features = _prior_features(prior_expert_rows)
    fresh = []
    for row in candidates:
        feature = source._features(row)
        if row["source_document_id"] in old_documents or row["template_family"] in old_families:
            continue
        if _historical_conflict(feature, historical) or any(
            source._conflict(feature, old) for old in old_features
        ):
            continue
        fresh.append(row)
    return {
        "original_proposed_split_sizes": {"confirmation": 512, "calibration": 1024,
            "development": 256, "heldout_matched": 512, "heldout_ood": 512},
        "length_filtered_enterprise_rows": dict(sorted(Counter(row["task"]
                                                              for row in candidates).items())),
        "strict_fresh_enterprise_rows": dict(sorted(Counter(row["task"] for row in fresh).items())),
        "count_scope": "eligible source rows before new inter-split selection; not model outcomes",
        "original_ood_enterprise_required": 96,
        "original_matched_enterprise_required": 256,
        "decision": "Reduce split sizes before model use; keep source, filters and quota unchanged",
        "frozen_split_sizes": dict(SPLIT_SIZES),
        "model_outputs_accessed": False,
    }


def make_recognition_manifest(
    source_path: Path, *, historical_prompts: Sequence[str] = (),
    prior_expert_rows: Sequence[dict] = (), split_sizes: dict[str, int] | None = None,
) -> list[dict[str, str]]:
    """Freeze fresh contexts with exact quotas and strict prior/cross-split exclusion.

    All source responses are discarded upstream. Matching includes the previous
    experiment's full prompt, instruction family and source passage, regardless
    of which split previously held that context. No predictive outputs are read.
    """
    sizes = SPLIT_SIZES if split_sizes is None else split_sizes
    _check_sizes(sizes)
    candidates = _source_candidates(source_path)
    features = {row["id"]: source._features(row) for row in candidates}
    historical = list({source._tokens(prompt) for prompt in historical_prompts})
    used_documents, used_families, previous_features = _prior_features(prior_expert_rows)
    used_ids: set[str] = set()
    selected_features: list[tuple[str, tuple]] = []
    rows = []
    for split, confirmation_stratum, size, categories in _allocations(sizes):
        for stratum, needed in (("enterprise_related", size // 8),
                                ("general_natural_language", size - size // 8)):
            pool = [row for row in candidates if row["task"] in categories
                    and row["stratum"] == stratum]
            counts: Counter[str] = Counter()
            selected = 0
            while selected < needed:
                made_progress = False
                for category in sorted(categories, key=lambda item: (counts[item], item)):
                    for candidate in pool:
                        if candidate["task"] != category or candidate["id"] in used_ids:
                            continue
                        used_ids.add(candidate["id"])
                        feature = features[candidate["id"]]
                        if (candidate["source_document_id"] in used_documents
                                or candidate["template_family"] in used_families):
                            continue
                        if (_historical_conflict(feature, historical)
                                or any(source._conflict(feature, old) for old in previous_features)):
                            continue
                        if any(other_split != split and source._conflict(feature, other)
                               for other_split, other in selected_features):
                            continue
                        row = dict(candidate, split=split, confirmation_stratum=confirmation_stratum)
                        rows.append(row)
                        selected_features.append((split, feature))
                        used_documents.add(row["source_document_id"])
                        used_families.add(row["template_family"])
                        counts[category] += 1
                        selected += 1
                        made_progress = True
                        break
                    if selected == needed:
                        break
                if not made_progress:
                    raise ValueError(
                        f"insufficient independent {stratum} rows for {split}/{confirmation_stratum}:"
                        f" selected {selected} of {needed}; filters and quota were not relaxed"
                    )
    order = {split: index for index, split in enumerate(sizes)}
    return sorted(rows, key=lambda row: (order[row["split"]], row["confirmation_stratum"],
                                       source._sha(SELECTION_SALT + row["id"])))


def validate_recognition_manifest(
    rows: Sequence[dict[str, str]], *, historical_prompts: Sequence[str] = (),
    prior_expert_rows: Sequence[dict] = (), split_sizes: dict[str, int] | None = None,
) -> dict[str, object]:
    """Independently recompute hashes, quotas, source separation and lexical checks."""
    sizes = SPLIT_SIZES if split_sizes is None else split_sizes
    _check_sizes(sizes)
    required = {"id", "split", "task", "source", "source_revision", "source_document_id",
                "source_instruction", "source_document", "template_family", "prompt",
                "prompt_sha256", "source_row", "source_license", "stratum", "confirmation_stratum"}
    if any(not required <= row.keys() for row in rows):
        raise ValueError("missing required manifest fields")
    if any(not isinstance(row[key], str) or not row[key].strip()
           for row in rows for key in required):
        raise ValueError("manifest fields must be nonempty strings")
    if Counter(row["split"] for row in rows) != Counter(sizes):
        raise ValueError("split sizes differ from frozen recipe")
    for field in ("id", "prompt", "source_document_id", "template_family"):
        if len({row[field] for row in rows}) != len(rows):
            raise ValueError(f"duplicate {field}; source documents and families must be isolated")
    for row in rows:
        if any(key in row for key in ("answer", "response", "output")):
            raise ValueError("response/answer fields are not allowed")
        if row["prompt_sha256"] != source._sha(row["prompt"]):
            raise ValueError("prompt hash mismatch")
        expected_prompt = row["source_instruction"] + (
            "\n\n" + row["source_document"]
            if row["source_document"] != row["source_instruction"] else ""
        )
        if row["prompt"] != expected_prompt:
            raise ValueError("prompt does not match source instruction/document")
        if (len(row["prompt"]) > source.MAX_PROMPT_CHARACTERS
                or len(row["prompt"].split()) < source.MIN_PROMPT_WORDS):
            raise ValueError("prompt length violates frozen filter")
        if (row["source"] != source.SOURCE_ID or row["source_revision"] != source.SOURCE_REVISION
                or row["source_license"] != source.SOURCE_LICENSE):
            raise ValueError("source provenance mismatch")
        if row["id"] != f"er-dolly-{int(row['source_row']):05d}":
            raise ValueError("source row ID mismatch")
        if row["source_document_id"] != "dolly-doc-" + source._sha(
                source.lexical_fingerprint(row["source_document"])):
            raise ValueError("source document hash mismatch")
        if row["template_family"] != "human-instruction-" + source._sha(
                source.lexical_fingerprint(row["source_instruction"])):
            raise ValueError("instruction family hash mismatch")
        group = row["confirmation_stratum"]
        if (row["split"] == "confirmation" and group not in {"matched", "ood"}) or (
            row["split"] != "confirmation" and group != "not_confirmation"
        ):
            raise ValueError("invalid confirmation stratum")
        is_ood = row["split"] == "heldout_ood" or group == "ood"
        if row["task"] not in (source.OOD_CATEGORIES if is_ood else source.MATCHED_CATEGORIES):
            raise ValueError("OOD task category leakage")
        expected_stratum = "enterprise_related" if source._ENTERPRISE_RE.search(
            row["source_instruction"]) else "general_natural_language"
        if row["stratum"] != expected_stratum:
            raise ValueError("enterprise stratum mismatch")
    for split, group, size, _categories in _allocations(sizes):
        selected = [row for row in rows if row["split"] == split
                    and row["confirmation_stratum"] == group]
        if len(selected) != size:
            raise ValueError("confirmation category strata differ from frozen recipe")
        if sum(row["stratum"] == "enterprise_related" for row in selected) != size // 8:
            raise ValueError("enterprise quota differs from frozen recipe")
    old_documents, old_families, prior_features = _prior_features(prior_expert_rows)
    historical = list({source._tokens(prompt) for prompt in historical_prompts})
    features = [source._features(row) for row in rows]
    comparisons = 0
    for index, (row, feature) in enumerate(zip(rows, features, strict=True)):
        if (row["source_document_id"] in old_documents or row["template_family"] in old_families
                or any(source._conflict(feature, old) for old in prior_features)):
            raise ValueError(f"prior expert source/family/prompt reuse: {row['id']}")
        if _historical_conflict(feature, historical):
            raise ValueError(f"historical prompt reuse: {row['id']}")
        for other, previous in zip(rows[:index], features[:index], strict=True):
            if other["split"] != row["split"]:
                comparisons += 1
                if source._conflict(feature, previous):
                    raise ValueError(f"cross-split near duplicate: {row['id']} / {other['id']}")
    canonical = json.dumps(list(rows), sort_keys=True, separators=(",", ":")).encode()
    return {
        "manifest_sha256": hashlib.sha256(canonical).hexdigest(), "rows": len(rows),
        "split_sizes": dict(sizes), "source": source.SOURCE_ID,
        "source_revision": source.SOURCE_REVISION, "source_sha256": source.SOURCE_SHA256,
        "license": source.SOURCE_LICENSE, "selection_salt": SELECTION_SALT,
        "maximum_prompt_characters": source.MAX_PROMPT_CHARACTERS,
        "minimum_prompt_words": source.MIN_PROMPT_WORDS,
        "enterprise_fraction": 1 / source.ENTERPRISE_DENOMINATOR,
        "enterprise_instruction_keywords": list(source.ENTERPRISE_KEYWORDS),
        "enterprise_scope": "instruction-keyword topic screen; not a production benchmark",
        "confirmation_policy": "equal matched/OOD category strata; never fit on confirmation",
        "task_counts": {split: dict(sorted(Counter(row["task"] for row in rows
                        if row["split"] == split).items())) for split in sizes},
        "stratum_counts": {split: dict(Counter(row["stratum"] for row in rows
                           if row["split"] == split)) for split in sizes},
        "confirmation_task_counts": {group: dict(sorted(Counter(row["task"] for row in rows
            if row["split"] == "confirmation" and row["confirmation_stratum"] == group).items()))
            for group in ("matched", "ood")},
        "historical_prompt_count": len(historical_prompts),
        "prior_expert_context_count": len(prior_expert_rows),
        "historical_unique_token_sets": len(historical),
        "cross_split_pairs_checked": comparisons,
        "prompt_and_instruction_token_jaccard_rejection_threshold": source.NEAR_DUPLICATE_THRESHOLD,
        "source_document_five_token_shingle_jaccard_rejection_threshold":
            source.DOCUMENT_SHINGLE_THRESHOLD,
        "source_document_and_exact_instruction_family_globally_unique": True,
        "prior_expert_source_document_instruction_and_prompt_exclusion": True,
        "independence_scope": "lexical exclusion; semantic family/article independence not established",
        "source_responses_used": False,
        "ood_definition": "creative_writing/summarization excluded from calibration and development",
    }
