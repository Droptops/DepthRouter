"""Pinned, human-written contexts for the bounded expert construction audit.

This module only reads prompt text, never model predictions or response labels.
The enterprise stratum is a documented instruction-keyword screen, not a claim
that Dolly is a production enterprise benchmark.  Template families below are
lexical instruction families; semantic independence cannot be guaranteed.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from depth_router.output_gap_data import lexical_fingerprint

SOURCE_ID = "databricks/databricks-dolly-15k"
SOURCE_REVISION = "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"
SOURCE_SHA256 = "2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec"
SOURCE_URL = (
    f"https://huggingface.co/datasets/{SOURCE_ID}/resolve/{SOURCE_REVISION}/"
    "databricks-dolly-15k.jsonl"
)
SOURCE_LICENSE = "CC-BY-SA-3.0"
SPLIT_SIZES = {
    "calibration": 1024,
    "development": 256,
    "heldout_matched": 512,
    "heldout_ood": 512,
}
MATCHED_CATEGORIES = (
    "brainstorming", "classification", "closed_qa", "general_qa",
    "information_extraction", "open_qa",
)
OOD_CATEGORIES = ("creative_writing", "summarization")
HISTORICAL_MANIFEST_PATHS = (
    "artifacts/predictive_working_set/20261013-validated-final/evaluation_manifest.jsonl",
    "artifacts/fused_sparse_swiglu/20261013-validated-final-v2/evaluation_manifest.jsonl",
    "artifacts/output_gap_completion/20261014-controlled-v1/evaluation_manifest.jsonl",
    "data/photonic_parallel_manifest.jsonl",
    "data/expert_audit/historical_alpaca_prompts.jsonl",
)
ALPACA_SOURCE_REVISION = "761dc5bfbdeeffa89b8bff5d038781a4055f796a"
ALPACA_SOURCE_URL = (
    "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/"
    f"{ALPACA_SOURCE_REVISION}/alpaca_data.json"
)
ALPACA_SOURCE_SHA256 = "2eddafc6b977608d778aaab8dfc7e50e547b3af9826dfb9e909d9fc362e4a419"
ALPACA_PROMPTS_SHA256 = "a6867486baba6e273959daae7c1adcb29ccd222fb5f88257d955c4b99237eefb"
HISTORICAL_ALPACA_MANIFEST_SHA256 = "94ab0d22a2cbb0de96037a3b2b60e9fd0d0e27b4d0a532ee5515855d29bbf9e5"
ALPACA_RECORDED_SPLIT_HASHES = {
    20261013: (
        "179d79464522abc570d79fc14e01cec1c8337e07e109e673fdf0c678f747ecec",
        "1fbbb60460087a328e2eb05d6317d2a61b19eab502d71bb6d9e4ba946afc8d34",
    ),
    20261014: (
        "e8730be6138d580d9f22c6fd24185c905e63774b1ed2bcecd290489f81105831",
        "87df512fa0fb3c01fd2bacdf48496c82e2af18bd11fa1c75f643aa8800622607",
    ),
    20261015: (
        "2bff383819dc9ff8557926d97d6b19fd1723d1d9753c4255092225c0d933e4af",
        "c82e88f40a69149d0f8aa5a4e8afd44eb3ed20ebcf2a0bedfab84dd8934e0943",
    ),
}
SELECTION_SALT = "DepthRouter-expert-audit-data-v1-20261014"
MAX_PROMPT_CHARACTERS = 1100
MIN_PROMPT_WORDS = 6
NEAR_DUPLICATE_THRESHOLD = 0.72
DOCUMENT_SHINGLE_THRESHOLD = 0.50
ENTERPRISE_DENOMINATOR = 8
ENTERPRISE_KEYWORDS = (
    "business", "customer", "company", "employee", "manager", "meeting", "office",
    "project", "sales", "invoice", "budget", "marketing", "client", "product",
    "software", "database", "security", "workflow", "contract", "email", "work",
    "leadership", "interview", "team", "data", "program", "service",
)
_ENTERPRISE_RE = re.compile(r"\b(?:" + "|".join(ENTERPRISE_KEYWORDS) + r")\b", re.IGNORECASE)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tokens(text: str) -> frozenset[str]:
    return frozenset(lexical_fingerprint(text).split())


def _shingles(text: str) -> frozenset[str]:
    tokens = lexical_fingerprint(text).split()
    return frozenset(" ".join(tokens[index:index + 5]) for index in range(len(tokens) - 4))


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return float(left == right)
    # No possible intersection can reach the rejection threshold in this case.
    if min(len(left), len(right)) / max(len(left), len(right)) < DOCUMENT_SHINGLE_THRESHOLD:
        return 0.0
    return len(left & right) / len(left | right)


def load_historical_prompts(repo: Path) -> tuple[list[str], list[dict[str, object]]]:
    """Hash and read all registered prompt manifests; missing sources are errors."""
    prompts: list[str] = []
    audit = []
    for relative in HISTORICAL_MANIFEST_PATHS:
        raw = (repo / relative).read_bytes()
        if relative.endswith("historical_alpaca_prompts.jsonl") and hashlib.sha256(
                raw.replace(b"\r\n", b"\n")).hexdigest() != HISTORICAL_ALPACA_MANIFEST_SHA256:
            raise ValueError("historical Alpaca manifest hash mismatch")
        source = [json.loads(line) for line in raw.decode("utf-8-sig").splitlines() if line]
        selected = [row["prompt"] for row in source]
        if not all(isinstance(prompt, str) and prompt.strip() for prompt in selected):
            raise ValueError(f"invalid historical prompt: {relative}")
        prompts.extend(selected)
        audit.append({"path": relative, "sha256": hashlib.sha256(raw).hexdigest(),
                      "rows": len(source), "unique_prompts": len(set(selected))})
    return list(dict.fromkeys(prompts)), audit


def make_historical_alpaca_manifest(source_path: Path) -> tuple[list[dict], dict]:
    """Reconstruct previously inspected prompts, checking recorded split hashes.

    Prior source: sibling checkout research/results/block_lowbit_fallback_alpaca_split_*.json
    and artifacts/overnight/block_causal_alpaca1000_*.json. Only their prompt
    provenance was consulted; no prior model results enter the new experiment.
    Seed 20261012 used up to 800 training and 200 test prompts. The next three
    seeds used 200 calibration and 200 test prompts. Source deduplication has no
    effect (all 52,002 source prompt strings are unique).
    """
    import torch

    raw = source_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ALPACA_SOURCE_SHA256:
        raise ValueError("historical Alpaca source hash mismatch")
    source = json.loads(raw)
    texts = []
    for record in source:
        instruction, extra = record["instruction"].strip(), record.get("input", "").strip()
        texts.append(instruction + ("\n\nInput:\n" + extra if extra else ""))
    unique = list(dict.fromkeys(texts))
    if len(texts) != len(unique) or _sha("\n\x1e\n".join(unique)) != ALPACA_PROMPTS_SHA256:
        raise ValueError("historical source prompt reconstruction does not match recorded hash")
    rows, policies = [], []
    for seed in (20261012, 20261013, 20261014, 20261015):
        order = torch.randperm(len(texts), generator=torch.Generator().manual_seed(seed)).tolist()
        shuffled = [texts[index] for index in order]
        training_count = 800 if seed == 20261012 else 200
        selected = {"training_or_calibration": shuffled[:training_count], "test": shuffled[-200:]}
        hashes = tuple(_sha("\n\x1e\n".join(prompts)) for prompts in selected.values())
        if seed in ALPACA_RECORDED_SPLIT_HASHES and hashes != ALPACA_RECORDED_SPLIT_HASHES[seed]:
            raise ValueError(f"historical Alpaca split {seed} differs from recorded prompt hashes")
        for role, prompts in selected.items():
            for index, prompt in enumerate(prompts):
                rows.append({"id": f"historical-alpaca-{seed}-{role}-{index:04d}",
                             "prompt": prompt, "source_revision": ALPACA_SOURCE_REVISION,
                             "split_seed": seed, "historical_role": role})
        policies.append({"split_seed": seed, "training_or_calibration_count": training_count,
                         "test_count": 200, "training_or_calibration_sha256": hashes[0],
                         "test_sha256": hashes[1],
                         "recorded_hash_validation": seed in ALPACA_RECORDED_SPLIT_HASHES})
    return rows, {
        "source_url": ALPACA_SOURCE_URL, "source_revision": ALPACA_SOURCE_REVISION,
        "source_sha256": ALPACA_SOURCE_SHA256, "source_prompts_sha256": ALPACA_PROMPTS_SHA256,
        "source_prompts": len(texts), "prompt_rows": len(rows),
        "unique_prompts": len({row["prompt"] for row in rows}),
        "split_algorithm": "torch.randperm(cpu Generator.manual_seed); first train/cal; last test",
        "torch_version_for_reconstruction": torch.__version__, "policies": policies,
        "provenance": "Existing sibling DepthRouter research/results/block_lowbit_fallback_"
            "alpaca_split_{20261013,20261014,20261015}.json and artifacts/overnight/"
            "block_causal_alpaca1000_{screen,capacity,balanced_topk}.json",
        "use": "Historical exclusion only; no answers or model outputs included",
        "license": "CC-BY-NC-4.0", "attribution": "Stanford Alpaca, tatsu-lab",
    }


def _features(row: dict[str, str]) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    return (_tokens(row["prompt"]), _tokens(row["source_instruction"]),
            _shingles(row["source_document"]))


def _conflict(left: tuple, right: tuple) -> bool:
    return (
        _jaccard(left[0], right[0]) >= NEAR_DUPLICATE_THRESHOLD
        or _jaccard(left[1], right[1]) >= NEAR_DUPLICATE_THRESHOLD
        or (bool(left[2]) and _jaccard(left[2], right[2]) >= DOCUMENT_SHINGLE_THRESHOLD)
    )


def _source_candidates(source_path: Path) -> list[dict[str, str]]:
    raw = source_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("Dolly source hash does not match the pinned source")
    candidates = []
    for index, line in enumerate(raw.decode("utf-8").splitlines()):
        record = json.loads(line)
        instruction = record["instruction"].strip()
        context = record["context"].strip()
        # Human source text is unmodified; the separator adds no task template.
        prompt = instruction + ("\n\n" + context if context else "")
        if not MIN_PROMPT_WORDS <= len(prompt.split()) or len(prompt) > MAX_PROMPT_CHARACTERS:
            continue
        if record["category"] not in MATCHED_CATEGORIES + OOD_CATEGORIES:
            raise ValueError("unknown Dolly category")
        document = context or instruction
        candidates.append({
            "id": f"ea-dolly-{index:05d}", "split": "", "task": record["category"],
            "source": SOURCE_ID, "source_revision": SOURCE_REVISION,
            "source_row": str(index), "source_license": SOURCE_LICENSE,
            "source_document_id": "dolly-doc-" + _sha(lexical_fingerprint(document)),
            "template_family": "human-instruction-" + _sha(lexical_fingerprint(instruction)),
            "source_instruction": instruction, "source_document": document,
            "stratum": "enterprise_related" if _ENTERPRISE_RE.search(instruction)
            else "general_natural_language",
            "prompt": prompt, "prompt_sha256": _sha(prompt),
        })
    return sorted(candidates, key=lambda row: _sha(SELECTION_SALT + row["id"]))


def make_expert_manifest(
    source_path: Path, *, historical_prompts: Sequence[str] = (),
    split_sizes: dict[str, int] | None = None,
) -> list[dict[str, str]]:
    """Choose fresh source documents and lexical families without model outcomes.

    Each split contains exactly 1/8 enterprise-keyword instructions. Categories
    rotate to avoid one natural-language category dominating. Entire prompts,
    instructions, and source documents are screened across splits; exact source
    documents and exact instruction families are globally unique. OOD reserves
    two task categories entirely. Source responses are deliberately discarded.
    """
    sizes = SPLIT_SIZES if split_sizes is None else split_sizes
    if set(sizes) != set(SPLIT_SIZES) or any(size <= 0 or size % 8 for size in sizes.values()):
        raise ValueError("all four split sizes must be positive multiples of eight")
    candidates = _source_candidates(source_path)
    features = {row["id"]: _features(row) for row in candidates}
    historical = list({_tokens(prompt) for prompt in historical_prompts})
    rows: list[dict[str, str]] = []
    selected_features: list[tuple[str, tuple]] = []
    used_ids, used_documents, used_families = set(), set(), set()
    for split, size in sizes.items():
        categories = OOD_CATEGORIES if split == "heldout_ood" else MATCHED_CATEGORIES
        # Scarcer enterprise examples are allocated first for every split.
        for stratum, needed in (("enterprise_related", size // 8),
                                ("general_natural_language", size - size // 8)):
            pool = [row for row in candidates if row["task"] in categories
                    and row["stratum"] == stratum]
            selected = 0
            counts: Counter[str] = Counter()
            while selected < needed:
                made_progress = False
                for category in sorted(categories, key=lambda task: (counts[task], task)):
                    for candidate in pool:
                        if candidate["task"] != category or candidate["id"] in used_ids:
                            continue
                        used_ids.add(candidate["id"])
                        feature = features[candidate["id"]]
                        if (candidate["source_document_id"] in used_documents
                                or candidate["template_family"] in used_families):
                            continue
                        if any(_jaccard(feature[0], old) >= NEAR_DUPLICATE_THRESHOLD
                               or _jaccard(feature[1], old) >= NEAR_DUPLICATE_THRESHOLD
                               for old in historical):
                            continue
                        if any(previous_split != split and _conflict(feature, previous)
                               for previous_split, previous in selected_features):
                            continue
                        row = dict(candidate, split=split)
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
                    raise ValueError(f"insufficient independent {stratum} source rows for {split}")
    return rows


def validate_expert_manifest(
    rows: Sequence[dict[str, str]], *, historical_prompts: Sequence[str] = (),
    split_sizes: dict[str, int] | None = None,
) -> dict[str, object]:
    """Recompute integrity, split isolation, quotas and lexical leakage checks."""
    sizes = SPLIT_SIZES if split_sizes is None else split_sizes
    required = {"id", "split", "task", "source", "source_revision", "source_document_id",
                "source_instruction", "source_document", "template_family", "prompt",
                "prompt_sha256", "source_row", "source_license", "stratum"}
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
        if row["prompt_sha256"] != _sha(row["prompt"]):
            raise ValueError("prompt hash mismatch")
        expected_prompt = row["source_instruction"] + (
            "\n\n" + row["source_document"]
            if row["source_document"] != row["source_instruction"] else ""
        )
        if row["prompt"] != expected_prompt:
            raise ValueError("prompt does not match the source instruction and document")
        if (row["source"] != SOURCE_ID or row["source_revision"] != SOURCE_REVISION
                or row["source_license"] != SOURCE_LICENSE):
            raise ValueError("source provenance mismatch")
        if row["source_document_id"] != "dolly-doc-" + _sha(
                lexical_fingerprint(row["source_document"])):
            raise ValueError("source document hash mismatch")
        if row["template_family"] != "human-instruction-" + _sha(
                lexical_fingerprint(row["source_instruction"])):
            raise ValueError("instruction family hash mismatch")
        categories = OOD_CATEGORIES if row["split"] == "heldout_ood" else MATCHED_CATEGORIES
        if row["task"] not in categories:
            raise ValueError("OOD task categories overlap the matched task categories")
        expected_stratum = "enterprise_related" if _ENTERPRISE_RE.search(
            row["source_instruction"]) else "general_natural_language"
        if row["stratum"] != expected_stratum:
            raise ValueError("enterprise stratum mismatch")
    for split, size in sizes.items():
        if sum(row["split"] == split and row["stratum"] == "enterprise_related"
               for row in rows) != size // 8:
            raise ValueError("enterprise quota differs from frozen recipe")
    historical = list({_tokens(prompt) for prompt in historical_prompts})
    features = [_features(row) for row in rows]
    comparisons = 0
    for index, (row, feature) in enumerate(zip(rows, features, strict=True)):
        if any(_jaccard(feature[0], old) >= NEAR_DUPLICATE_THRESHOLD
               or _jaccard(feature[1], old) >= NEAR_DUPLICATE_THRESHOLD for old in historical):
            raise ValueError(f"historical prompt reuse: {row['id']}")
        for other, previous in zip(rows[:index], features[:index], strict=True):
            if other["split"] != row["split"]:
                comparisons += 1
                if _conflict(feature, previous):
                    raise ValueError(f"cross-split near duplicate: {row['id']} / {other['id']}")
    canonical = json.dumps(list(rows), sort_keys=True, separators=(",", ":")).encode()
    return {
        "manifest_sha256": hashlib.sha256(canonical).hexdigest(), "rows": len(rows),
        "split_sizes": dict(sizes), "source": SOURCE_ID, "source_revision": SOURCE_REVISION,
        "source_sha256": SOURCE_SHA256, "license": SOURCE_LICENSE,
        "selection_salt": SELECTION_SALT, "maximum_prompt_characters": MAX_PROMPT_CHARACTERS,
        "enterprise_instruction_keywords": list(ENTERPRISE_KEYWORDS),
        "enterprise_fraction": 1 / ENTERPRISE_DENOMINATOR,
        "enterprise_scope": "instruction-keyword topic screen; not a production benchmark",
        "task_counts": {split: dict(sorted(Counter(row["task"] for row in rows
                        if row["split"] == split).items())) for split in sizes},
        "stratum_counts": {split: dict(Counter(row["stratum"] for row in rows
                           if row["split"] == split)) for split in sizes},
        "historical_prompt_count": len(historical_prompts),
        "historical_unique_token_sets": len(historical),
        "cross_split_pairs_checked": comparisons,
        "prompt_and_instruction_token_jaccard_rejection_threshold": NEAR_DUPLICATE_THRESHOLD,
        "source_document_five_token_shingle_jaccard_rejection_threshold":
            DOCUMENT_SHINGLE_THRESHOLD,
        "source_document_and_exact_instruction_family_globally_unique": True,
        "independence_scope": "lexical exclusion; semantic family independence is not established",
        "source_responses_used": False,
        "ood_definition": "creative_writing and summarization categories excluded from fitting",
    }
