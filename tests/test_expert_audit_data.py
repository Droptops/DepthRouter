import hashlib
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

import depth_router.expert_audit_data as data

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def frozen_rows():
    path = REPO / "data/expert_audit/evaluation_manifest.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def tiny_rows(frozen_rows, monkeypatch):
    monkeypatch.setattr(data, "_source_candidates", lambda _path: deepcopy(frozen_rows))
    sizes = {split: 8 for split in data.SPLIT_SIZES}
    rows = data.make_expert_manifest(Path("unused"), split_sizes=sizes)
    return rows, sizes


def test_frozen_data_validates_all_sources_and_disjoint_splits(frozen_rows):
    historical, historical_sources = data.load_historical_prompts(REPO)
    audit = data.validate_expert_manifest(frozen_rows, historical_prompts=historical)
    saved = json.loads((REPO / "data/expert_audit/data_audit.json").read_text())
    assert audit["manifest_sha256"] == saved["manifest_sha256"]
    assert audit["rows"] == 2304
    assert Counter(row["split"] for row in frozen_rows) == data.SPLIT_SIZES
    assert len(historical_sources) == 5
    assert audit["cross_split_pairs_checked"] == 1835008
    assert not audit["source_responses_used"]
    assert all("answer" not in row and "response" not in row for row in frozen_rows)
    assert set(data.MATCHED_CATEGORIES).isdisjoint(data.OOD_CATEGORIES)
    for split, size in data.SPLIT_SIZES.items():
        assert audit["stratum_counts"][split]["enterprise_related"] == size // 8


def test_manifest_selection_is_deterministic_without_source_responses(tiny_rows):
    rows, sizes = tiny_rows
    assert rows == data.make_expert_manifest(Path("unused"), split_sizes=sizes)
    assert data.validate_expert_manifest(rows, split_sizes=sizes)["rows"] == 32


def test_pinned_source_rejects_modified_bytes(tmp_path):
    path = tmp_path / "source.jsonl"
    path.write_text('{"instruction": "Changed source text"}\n')
    with pytest.raises(ValueError, match="pinned source"):
        data.make_expert_manifest(path)


def test_historical_case_numeric_and_punctuation_edits_cannot_evade_exclusion(tiny_rows):
    rows, sizes = tiny_rows
    historical = rows[0]["prompt"].upper().replace("?", "!!!")
    with pytest.raises(ValueError, match="historical prompt reuse"):
        data.validate_expert_manifest(rows, split_sizes=sizes, historical_prompts=[historical])
    assert data._tokens("Invoice 137, status PENDING!") == data._tokens(
        "invoice 999; STATUS pending."
    )


def test_duplicate_source_document_cannot_cross_splits(tiny_rows):
    rows, sizes = deepcopy(tiny_rows)
    second_split = next(row for row in rows if row["split"] != rows[0]["split"])
    second_split["source_document_id"] = rows[0]["source_document_id"]
    with pytest.raises(ValueError, match="duplicate source_document_id"):
        data.validate_expert_manifest(rows, split_sizes=sizes)


def test_duplicate_instruction_family_cannot_cross_splits(tiny_rows):
    rows, sizes = deepcopy(tiny_rows)
    second_split = next(row for row in rows if row["split"] != rows[0]["split"])
    second_split["template_family"] = rows[0]["template_family"]
    with pytest.raises(ValueError, match="duplicate template_family"):
        data.validate_expert_manifest(rows, split_sizes=sizes)


def test_ood_task_leakage_is_rejected(tiny_rows):
    rows, sizes = deepcopy(tiny_rows)
    rows[0]["task"] = data.OOD_CATEGORIES[0]
    with pytest.raises(ValueError, match="OOD task categories"):
        data.validate_expert_manifest(rows, split_sizes=sizes)


def test_prompt_hash_and_source_binding_are_checked(tiny_rows):
    rows, sizes = deepcopy(tiny_rows)
    rows[0]["prompt"] += " altered"
    with pytest.raises(ValueError, match="prompt hash mismatch"):
        data.validate_expert_manifest(rows, split_sizes=sizes)
    rows[0]["prompt_sha256"] = hashlib.sha256(rows[0]["prompt"].encode()).hexdigest()
    with pytest.raises(ValueError, match="source instruction and document"):
        data.validate_expert_manifest(rows, split_sizes=sizes)


def test_near_duplicate_documents_and_instruction_families_are_rejected():
    left = {
        "prompt": "Explain the process for terminating an employee contract after review.",
        "source_instruction": "Explain the process for terminating an employee contract after review.",
        "source_document": "A lengthy unique source document has several repeated phrases and "
            "specific content that must not be shared with another evaluation split.",
    }
    changed = dict(left, source_instruction=left["source_instruction"].replace("review", "approval"))
    assert data._conflict(data._features(left), data._features(changed))
    changed = {
        "prompt": "Please summarize this unrelated question about astronomy and comet orbits.",
        "source_instruction": "Please summarize this unrelated question about astronomy and comet orbits.",
        "source_document": left["source_document"] + " Additional trailing text.",
    }
    assert data._conflict(data._features(left), data._features(changed))


def test_historical_sources_are_required(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_historical_prompts(tmp_path)


def test_invalid_split_budget_is_rejected():
    with pytest.raises(ValueError, match="multiples of eight"):
        data.make_expert_manifest(Path("unused"), split_sizes={"calibration": 8})
