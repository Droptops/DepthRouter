import hashlib
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

from depth_router import recognition_data as data

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def tiny_rows(monkeypatch):
    raw = (REPO / "data/expert_audit/evaluation_manifest.jsonl").read_text(encoding="utf-8")
    candidates = [dict(json.loads(line), confirmation_stratum="not_confirmation")
                  for line in raw.splitlines()]
    for row in candidates:
        row["id"] = f"er-dolly-{int(row['source_row']):05d}"
    monkeypatch.setattr(data, "_source_candidates", lambda _path: deepcopy(candidates))
    sizes = {split: 16 if split == "confirmation" else 8 for split in data.SPLIT_SIZES}
    return data.make_recognition_manifest(Path("unused"), split_sizes=sizes), sizes


def test_tiny_manifest_is_deterministic_and_confirmation_is_separate(tiny_rows):
    rows, sizes = tiny_rows
    assert rows == data.make_recognition_manifest(Path("unused"), split_sizes=sizes)
    result = data.validate_recognition_manifest(rows, split_sizes=sizes)
    assert result["rows"] == 48
    assert Counter(row["confirmation_stratum"] for row in rows if row["split"] == "confirmation") == {
        "matched": 8, "ood": 8,
    }
    assert result["source_responses_used"] is False


def test_frozen_recognition_manifest_is_independent_of_every_prior_prompt():
    path = REPO / "data/expert_recognition/evaluation_manifest.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    historical, prior, exclusions = data.load_recognition_exclusions(REPO)
    result = data.validate_recognition_manifest(rows, historical_prompts=historical,
                                               prior_expert_rows=prior)
    saved = json.loads(path.with_name("data_audit.json").read_text())
    assert result["manifest_sha256"] == saved["manifest_sha256"]
    assert result["split_sizes"] == data.SPLIT_SIZES
    assert result["historical_prompt_count"] == 6163
    assert result["prior_expert_context_count"] == 2304
    assert len(exclusions) == 6
    assert {row["id"] for row in rows}.isdisjoint(row["id"] for row in prior)
    assert result["source_responses_used"] is False


def test_prior_source_and_family_reuse_is_rejected(tiny_rows):
    rows, sizes = tiny_rows
    prior = deepcopy(rows[0])
    prior["prompt"] = "Changed text cannot hide the same original source document."
    with pytest.raises(ValueError, match="prior expert source/family/prompt reuse"):
        data.validate_recognition_manifest(rows, prior_expert_rows=[prior], split_sizes=sizes)


def test_historical_case_and_punctuation_edits_are_excluded(tiny_rows):
    rows, sizes = tiny_rows
    with pytest.raises(ValueError, match="historical prompt reuse"):
        data.validate_recognition_manifest(rows, historical_prompts=[rows[0]["prompt"].upper()],
                                           split_sizes=sizes)


def test_ood_category_cannot_enter_calibration(tiny_rows):
    rows, sizes = deepcopy(tiny_rows)
    row = next(row for row in rows if row["split"] == "calibration")
    row["task"] = data.source.OOD_CATEGORIES[0]
    with pytest.raises(ValueError, match="OOD task category leakage"):
        data.validate_recognition_manifest(rows, split_sizes=sizes)


def test_response_fields_are_rejected(tiny_rows):
    rows, sizes = deepcopy(tiny_rows)
    rows[0]["response"] = "A source answer must never become a feature."
    with pytest.raises(ValueError, match="response/answer fields"):
        data.validate_recognition_manifest(rows, split_sizes=sizes)


def test_confirmation_requires_balanced_category_strata(tiny_rows):
    rows, sizes = deepcopy(tiny_rows)
    rows[0]["confirmation_stratum"] = "not_confirmation"
    with pytest.raises(ValueError, match="invalid confirmation stratum"):
        data.validate_recognition_manifest(rows, split_sizes=sizes)


def test_prompt_hash_source_and_document_fields_are_bound(tiny_rows):
    rows, sizes = deepcopy(tiny_rows)
    rows[0]["prompt"] += " altered"
    with pytest.raises(ValueError, match="prompt hash mismatch"):
        data.validate_recognition_manifest(rows, split_sizes=sizes)
    rows[0]["prompt_sha256"] = hashlib.sha256(rows[0]["prompt"].encode()).hexdigest()
    with pytest.raises(ValueError, match="source instruction/document"):
        data.validate_recognition_manifest(rows, split_sizes=sizes)


def test_invalid_budget_is_not_silently_reduced():
    with pytest.raises(ValueError, match="positive multiples"):
        data.make_recognition_manifest(Path("unused"), split_sizes={"calibration": 8})


def test_infeasible_pool_does_not_relax_quota_or_filters(monkeypatch):
    monkeypatch.setattr(data, "_source_candidates", lambda _path: [])
    sizes = {split: 16 if split == "confirmation" else 8 for split in data.SPLIT_SIZES}
    with pytest.raises(ValueError, match="filters and quota were not relaxed"):
        data.make_recognition_manifest(Path("unused"), split_sizes=sizes)


def test_missing_prior_manifests_are_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_recognition_exclusions(tmp_path)


def test_changed_source_bytes_cannot_be_used(tmp_path):
    path = tmp_path / "source.jsonl"
    path.write_text('{"instruction": "A changed prompt is not the pinned data."}\n')
    with pytest.raises(ValueError, match="pinned source"):
        data.make_recognition_manifest(path)
