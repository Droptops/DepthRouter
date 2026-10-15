from collections import Counter
from pathlib import Path

import pytest

from depth_router.output_gap_data import (
    HISTORICAL_MANIFEST_PATHS,
    NEAR_DUPLICATE_THRESHOLD,
    SPLIT_SIZES,
    lexical_fingerprint,
    load_historical_prompts,
    make_output_gap_manifest,
    validate_output_gap_manifest,
)


def test_fresh_manifest_deterministic_and_excludes_all_historical_sources():
    repo = Path(__file__).resolve().parents[1]
    historical, sources = load_historical_prompts(repo)
    rows = make_output_gap_manifest()
    assert rows == make_output_gap_manifest()
    audit = validate_output_gap_manifest(rows, historical_prompts=historical)
    assert len(rows) == 1536
    assert Counter(row["split"] for row in rows) == SPLIT_SIZES
    assert len(sources) == len(HISTORICAL_MANIFEST_PATHS) == 3
    assert sum(source["rows"] for source in sources) >= 512 + 1536
    assert all(len(source["sha256"]) == 64 for source in sources)
    assert len(audit["manifest_sha256"]) == 64
    assert audit["within_family_numeric_variants_allowed"]
    assert all(check["max_jaccard"] < NEAR_DUPLICATE_THRESHOLD for check in audit["comparisons"])


def test_numeric_case_and_punctuation_edits_cannot_hide_historical_reuse():
    rows = make_output_gap_manifest()
    reused = rows[0]["prompt"].upper().replace("12", "999").replace(".", "!")
    assert lexical_fingerprint(reused) == lexical_fingerprint(rows[0]["prompt"])
    with pytest.raises(ValueError, match="near-duplicate"):
        validate_output_gap_manifest(rows, historical_prompts=[reused])


def test_word_reordering_is_detected_by_jaccard():
    rows = make_output_gap_manifest()
    reused = " ".join(reversed(rows[0]["prompt"].split()))
    with pytest.raises(ValueError, match="near-duplicate"):
        validate_output_gap_manifest(rows, historical_prompts=[reused])


def test_renaming_a_family_does_not_bypass_cross_split_exclusion():
    rows = make_output_gap_manifest()
    heldout = next(row for row in rows if row["split"] == "heldout_matched")
    heldout["prompt"] = rows[0]["prompt"] + " Please."
    heldout["template_family"] = "dishonestly_renamed"
    with pytest.raises(ValueError, match="near-duplicate"):
        validate_output_gap_manifest(rows)


def test_identical_normalized_templates_cannot_cross_families():
    rows = make_output_gap_manifest()
    rows[1]["template_family"] = "another_family"
    with pytest.raises(ValueError, match="normalized fingerprint"):
        validate_output_gap_manifest(rows)


def test_task_balance_is_enforced_and_missing_history_is_fatal(tmp_path):
    rows = make_output_gap_manifest()
    rows[0]["task"] = "extraction"
    with pytest.raises(ValueError, match="balanced"):
        validate_output_gap_manifest(rows)
    with pytest.raises(FileNotFoundError):
        load_historical_prompts(tmp_path)


def test_answer_construction_for_known_cases():
    rows = make_output_gap_manifest()
    selected = {(row["split"], row["task"]): row for row in reversed(rows)}
    assert selected["calibration", "classification"]["answer"] == "COOL"
    assert selected["calibration", "extraction"]["answer"] == "05:00"
    assert selected["calibration", "numerical"]["answer"] == "71"
    assert selected["development", "numerical"]["answer"] == "76"
    assert selected["heldout_matched", "numerical"]["answer"] == "11"
    assert selected["heldout_ood", "finite_state"]["answer"] == "EAST"
    assert selected["heldout_ood", "substitution"]["answer"] == "MMMM"
    assert selected["heldout_ood", "unit_conversion"]["answer"] == "120"
    assert selected["heldout_ood", "ordinal_lookup"]["answer"] == "158"
