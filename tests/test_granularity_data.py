"""Lineage and byte-preserving quarantine checks; no model or GPU calls."""

import hashlib
import json
import shutil
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

from depth_router import granularity_data as data

ROOT = Path(__file__).resolve().parents[1]


def _read_manifest(path):
    return [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


@pytest.fixture(scope="module")
def inherited_rows():
    return _read_manifest(ROOT / data.PRIOR_MANIFEST)


@pytest.fixture(scope="module")
def derived():
    return data.derive_granularity_manifest(ROOT)


@pytest.fixture
def copied_lineage(tmp_path):
    run = ROOT / data.PRIOR_RUN
    shutil.copytree(run, tmp_path / data.PRIOR_RUN)
    registration = json.loads((run / "preregistration.json").read_text())
    for name in registration["source_sha256"]:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    return tmp_path


def test_derived_manifest_is_exact_unopened_subset(derived, inherited_rows):
    rows, audit, quarantine = derived
    assert rows == [row for row in inherited_rows if row["split"] != "confirmation"]
    assert len(rows) == 1600
    assert Counter(row["split"] for row in rows) == data.SPLIT_SIZES
    assert len(quarantine["ids"]) == 128
    assert {row["id"] for row in rows}.isdisjoint(quarantine["ids"])
    assert audit["retained_ids_prompts_splits_and_provenance_unchanged"]
    assert audit["full_inherited_recognition_validation"]["rows"] == 1728
    assert audit["filtered_four_split_validation"]["rows"] == 1600
    assert audit["filtered_exclusion_prompt_count"] == 6163 + 2304 + 128
    assert not audit["model_loaded"] and not audit["gpu_used"] and not audit["source_responses_used"]
    for split, size in data.SPLIT_SIZES.items():
        assert sum(row["split"] == split and row["stratum"] == "enterprise_related" for row in rows) == size // 8


def test_serialized_lines_are_byte_identical_and_match_frozen_artifact(derived):
    rows, audit, _quarantine = derived
    expected = b"".join(line for line in (ROOT / data.PRIOR_MANIFEST).read_bytes().splitlines(keepends=True)
                        if json.loads(line)["split"] != "confirmation")
    actual = data.serialized_manifest_bytes(ROOT, rows)
    assert actual == expected
    assert hashlib.sha256(actual).hexdigest() == audit["manifest_file_sha256"]
    assert (ROOT / "data/expert_granularity/evaluation_manifest.jsonl").read_bytes() == actual
    saved = json.loads((ROOT / "data/expert_granularity/data_audit.json").read_text())
    assert saved == audit


def test_no_changed_prompt_or_reordering_can_hide_in_derivative(derived):
    rows = deepcopy(derived[0])
    rows[0]["prompt"] += " Changed input."
    with pytest.raises(ValueError, match="changed text, provenance, fields or inherited order"):
        data.serialized_manifest_bytes(ROOT, rows)
    with pytest.raises(ValueError, match="changed text, provenance, fields or inherited order"):
        data.serialized_manifest_bytes(ROOT, list(reversed(derived[0])))


def test_quarantine_hashes_bind_every_original_confirmation_row(derived, inherited_rows):
    _rows, audit, quarantine = derived
    old_confirmation = [row for row in inherited_rows if row["split"] == "confirmation"]
    assert quarantine["ids"] == [row["id"] for row in old_confirmation]
    assert quarantine["canonical_rows_sha256"] == data._canonical_sha(old_confirmation)
    assert quarantine["ids_sha256"] == audit["quarantined_ids_sha256"]
    assert quarantine["prompt_sha256_by_id"] == {row["id"]: row["prompt_sha256"] for row in old_confirmation}
    assert len(data.BASE_COMMIT) == 40


def test_only_exact_prior_run_is_inspected(copied_lineage, inherited_rows):
    unrelated = copied_lineage / "artifacts/expert_recognition/unrelated-future-run"
    unrelated.mkdir(parents=True)
    (unrelated / "matrix-heldout_matched.npz").write_bytes(b"Not part of the pinned old run")
    audit = data.verify_prior_run_lineage(copied_lineage, inherited_rows)
    assert audit["no_retained_context_model_outputs_in_prior_run"]
    assert audit["prior_registered_source_count"] == 26
    assert audit["prior_run_files_checked"] == sorted(audit["prior_run_files_checked"])
    (copied_lineage / data.PRIOR_RUN / "matrix-heldout_matched.npz").write_bytes(b"Forbidden output")
    with pytest.raises(ValueError, match="unexpected output in stopped prior run"):
        data.verify_prior_run_lineage(copied_lineage, inherited_rows)


def test_changed_terminal_decision_is_rejected(copied_lineage, inherited_rows):
    path = copied_lineage / data.PRIOR_RUN / "promotion_decision.json"
    record = json.loads(path.read_text())
    record["confirmation_pass"] = True
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="pinned prior-run evidence changed"):
        data.verify_prior_run_lineage(copied_lineage, inherited_rows)


def test_state_output_on_retained_context_is_rejected(copied_lineage, inherited_rows):
    path = copied_lineage / data.PRIOR_RUN / "states-confirmation.jsonl"
    records = _read_manifest(path)
    records[0]["id"] = next(row["id"] for row in inherited_rows if row["split"] == "calibration")
    path.write_text("\n".join(json.dumps(row) for row in records) + "\n")
    with pytest.raises(ValueError, match="prior state metadata contains an unopened context"):
        data.verify_prior_run_lineage(copied_lineage, inherited_rows)


def test_changed_source_snapshot_is_rejected(copied_lineage, inherited_rows):
    path = copied_lineage / data.PRIOR_RUN / "registered_sources" / data.PRIOR_MANIFEST
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="prior registered snapshot changed"):
        data.verify_prior_run_lineage(copied_lineage, inherited_rows)


@pytest.mark.parametrize("name", ["../outside", "C:/outside", "folder/../../outside", "folder\\outside"])
def test_lineage_paths_cannot_escape_root(tmp_path, name):
    with pytest.raises(ValueError, match="unsafe lineage path"):
        data._safe_child(tmp_path, name)
