"""Derive the next audit's data from unopened partitions of the stopped study.

No sampling or text rewriting occurs. The failed 128-context confirmation split
is quarantined in full. Freshness means that the pinned prior execution produced
no model outputs on these retained contexts, not semantic independence or a new
source distribution.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np

from depth_router import expert_audit_data, recognition_data

BASE_COMMIT = "1be53fefd37c344dbc4cdc44bb708e3e226ae0e9"
PRIOR_RUN = "artifacts/expert_recognition/20261014-recognition-v1"
PRIOR_MANIFEST = "data/expert_recognition/evaluation_manifest.jsonl"
PRIOR_MANIFEST_SHA256 = "c4eb5654b713927c2924195bb2cd98435306214b32d98c402d0d42ff8e251fe6"
PRIOR_EVIDENCE_SHA256 = {
    "preregistration.json": "493f8717c4da4176e09c0cdc88e0d9f344bf8fed49f7a42de176730b88f46b54",
    "promotion_decision.json": "98d6d9ca83136c36c31efa4d8f4d1c00c10d1eac969658f8a86993d3186bba4a",
    "confirmation_decision.json": "210120aa99579ba5fc4c78a1ca8477a9fe39892e44dd35d82b21fe600e41f397",
    "events.jsonl": "8924e83ef40dcfde3a98167ff112a5b1b6a826c879f91b44ab4d28144a3529fc",
    "frozen_partition.json": "f827d04b841f1d1997cd700f6ee95e9c5d86423c4d5048fb0bb71c5f103b4a5d",
    "frozen_fixed_selection.json": "f69e12cdb710bd5f8883218b8b085cc72a50d53d346f2935396ca35cb105e704",
}
SPLIT_SIZES = {"calibration": 768, "development": 128, "heldout_matched": 512, "heldout_ood": 192}
_ALLOWED_PRIOR_FILES = {
    *PRIOR_EVIDENCE_SHA256, "capture-confirmation.json", "states-confirmation.jsonl",
    "matrix-confirmation.npz", "cache-confirmation.pt", "environment.json",
    "full_model_cached_validation.json", "postrun_integrity_verification.json", "DRAFT_PR.md",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_sha(value) -> str:
    return _sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_child(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    _require(not relative.is_absolute() and relative.parts and all(
        part not in (".", "..") and ":" not in part and "\\" not in part
        for part in relative.parts), f"unsafe lineage path: {name}")
    child = (root / name).resolve()
    _require(child.is_relative_to(root.resolve()), f"lineage path escapes root: {name}")
    return child


def serialized_manifest_bytes(repo: Path, rows: list[dict]) -> bytes:
    """Retain the exact inherited JSONL line bytes and their original order."""
    raw = (repo / PRIOR_MANIFEST).read_bytes()
    _require(_sha(raw) == PRIOR_MANIFEST_SHA256, "inherited recognition manifest hash changed")
    selected = {row["id"] for row in rows}
    source = [(json.loads(line), line) for line in raw.splitlines(keepends=True) if line.strip()]
    inherited = [row for row, _line in source if row["id"] in selected]
    _require(rows == inherited, "derived rows changed text, provenance, fields or inherited order")
    return b"".join(line for row, line in source if row["id"] in selected)


def verify_prior_run_lineage(repo: Path, source_rows: list[dict]) -> dict:
    """Audit only the specifically pinned prior run, not other study directories."""
    run = repo / PRIOR_RUN
    for name, expected in PRIOR_EVIDENCE_SHA256.items():
        _require(_sha((run / name).read_bytes()) == expected, f"pinned prior-run evidence changed: {name}")
    registration = _json(run / "preregistration.json")
    _require(registration["before_model_load_fit_and_fresh_outputs"] is True,
             "prior run was not prospectively registered")
    inherited_hashes = registration["source_sha256"]
    for name, expected in inherited_hashes.items():
        snapshot = _safe_child(run / "registered_sources", name)
        _require(_sha(snapshot.read_bytes()) == expected, f"prior registered snapshot changed: {name}")
        current = _safe_child(repo, name)
        _require(_sha(current.read_bytes()) == expected, f"inherited scientific source changed: {name}")
    _require(inherited_hashes[PRIOR_MANIFEST] == PRIOR_MANIFEST_SHA256,
             "prior registered manifest is not the inherited manifest")
    config = registration["config"]
    _require(config["partition_name"] == "C-29" and config["fixed_candidate_index"] == 8,
             "prior discovery partition/fixed candidate changed")
    decision = _json(run / "promotion_decision.json")
    _require(decision["confirmation_pass"] is False and decision["promote"] is False
             and decision["recognition"] == "not_run_confirmation_failed"
             and decision["runtime"] == "not_run_confirmation_failed"
             and decision["original_weights_unchanged"] is True,
             "prior run did not stop at failed confirmation")
    _require(_json(run / "confirmation_decision.json")["point_pass"] is False,
             "prior confirmation result contradicts its stop decision")
    observed = []
    # WindowsPath ordering case-folds, while PosixPath ordering does not.
    # Serialized provenance must use the same explicit string order on both.
    for path in sorted(run.rglob("*"), key=lambda item: item.relative_to(run).as_posix()):
        relative = path.relative_to(run)
        if not path.is_file() or relative.parts[0] == "registered_sources":
            continue
        _require(len(relative.parts) == 1 and path.name in _ALLOWED_PRIOR_FILES,
                 f"unexpected output in stopped prior run: {relative.as_posix()}")
        observed.append(relative.as_posix())
    _require(not (run / "runtime").exists(), "runtime directory exists after failed confirmation")
    confirmation_ids = {row["id"] for row in source_rows if row["split"] == "confirmation"}
    _require(len(confirmation_ids) == 128, "quarantine must contain all 128 confirmation contexts")
    states = [json.loads(line) for line in (run / "states-confirmation.jsonl").read_text().splitlines()]
    _require(all(row["split"] == "confirmation" and row["id"] in confirmation_ids for row in states),
             "prior state metadata contains an unopened context")
    labels = np.asarray(_json(run / "frozen_partition.json"))
    _require(labels.shape == (4864,) and np.array_equal(np.bincount(labels, minlength=8), np.full(8, 608)),
             "prior partition is not eight balanced 608-neuron experts")
    fixed = _json(run / "frozen_fixed_selection.json")
    combinations = np.asarray(list(itertools.combinations(range(8), 6)))
    _require(fixed["index"] == 8 and fixed["experts"] == combinations[8].tolist()
             and fixed["selected_on"] == "development", "prior fixed choice changed")
    with np.load(run / "matrix-confirmation.npz", allow_pickle=False) as matrix:
        _require(np.array_equal(matrix["context_ids"], [row["id"] for row in states])
                 and set(matrix["context_ids"].tolist()) <= confirmation_ids,
                 "prior candidate matrix contains an unopened context or differs from state metadata")
        _require(np.array_equal(matrix["partition"], labels)
                 and np.array_equal(matrix["selected_experts"], combinations),
                 "prior confirmation matrix does not use the frozen C-29/28-candidate audit")
        state_count = len(matrix["context_ids"])
    capture = _json(run / "capture-confirmation.json")
    _require(capture["states"] == state_count == len(states) and capture["contexts_requested"] == 128,
             "prior capture count differs from matrix/metadata")
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    _require(events[-1]["event"] == "cycle_complete" and events[-1]["promote"] is False,
             "prior event log lacks terminal stop")
    _require(all(row.get("split", "confirmation") == "confirmation"
                 and row.get("matrix", "matrix-confirmation.npz") == "matrix-confirmation.npz"
                 and row["event"] in {"capture_progress", "candidate_progress", "confirmation_complete", "cycle_complete"}
                 for row in events), "prior event log contains work beyond confirmation")
    return {
        "base_commit": BASE_COMMIT, "prior_run": PRIOR_RUN,
        "prior_evidence_sha256": dict(PRIOR_EVIDENCE_SHA256),
        "inherited_source_sha256": inherited_hashes,
        "prior_registered_source_count": len(inherited_hashes),
        "all_inherited_current_and_snapshot_hashes_exact": True,
        "prior_run_files_checked": observed,
        "prior_output_scope": "confirmation only; exact pinned run directory inspected",
        "prior_candidate_audit": {"partition": "C-29", "fixed_candidate_index": 8,
            "candidate_count": 28, "states": state_count,
            "matrix_sha256": _sha((run / "matrix-confirmation.npz").read_bytes()),
            "state_metadata_sha256": _sha((run / "states-confirmation.jsonl").read_bytes())},
        "no_retained_context_model_outputs_in_prior_run": True,
        "claim_limit": "provenance and artifact audit, not semantic independence or proof about unrecorded external runs",
    }


def derive_granularity_manifest(repo: Path) -> tuple[list[dict], dict, dict]:
    """Validate inherited data, quarantine confirmation, and derive 1,600 rows."""
    repo = Path(repo).resolve()
    source_raw = (repo / PRIOR_MANIFEST).read_bytes()
    _require(_sha(source_raw) == PRIOR_MANIFEST_SHA256, "inherited recognition manifest hash changed")
    inherited = [json.loads(line) for line in source_raw.splitlines() if line.strip()]
    lineage = verify_prior_run_lineage(repo, inherited)
    historical, prior_expert, exclusion_sources = recognition_data.load_recognition_exclusions(repo)
    full_audit = recognition_data.validate_recognition_manifest(
        inherited, historical_prompts=historical, prior_expert_rows=prior_expert)
    saved = _json(repo / "data/expert_recognition/data_audit.json")
    _require(full_audit["manifest_sha256"] == saved["manifest_sha256"],
             "full inherited recognition manifest fails its frozen canonical hash")
    quarantined = [row for row in inherited if row["split"] == "confirmation"]
    retained = [row for row in inherited if row["split"] != "confirmation"]
    _require(Counter(row["split"] for row in retained) == Counter(SPLIT_SIZES),
             "derived split counts differ from the frozen unopened partitions")
    _require(all(row["confirmation_stratum"] == "not_confirmation" for row in retained),
             "confirmation row survived quarantine")
    exclusions = [*historical, *(row["prompt"] for row in prior_expert),
                  *(row["prompt"] for row in quarantined)]
    filtered_audit = expert_audit_data.validate_expert_manifest(
        retained, historical_prompts=exclusions, split_sizes=SPLIT_SIZES)
    raw = serialized_manifest_bytes(repo, retained)
    quarantine_ids = [row["id"] for row in quarantined]
    quarantined_bytes = b"".join(line for line in source_raw.splitlines(keepends=True)
                                 if line.strip() and json.loads(line)["split"] == "confirmation")
    quarantine = {
        "reason": "failed prior confirmation already produced model outputs; never reused",
        "context_count": len(quarantined), "ids": quarantine_ids,
        "ids_sha256": _canonical_sha(quarantine_ids),
        "source_row_bytes_sha256": _sha(quarantined_bytes),
        "canonical_rows_sha256": _canonical_sha(quarantined),
        "prompt_sha256_by_id": {row["id"]: row["prompt_sha256"] for row in quarantined},
    }
    return retained, {
        "rows": len(retained), "split_sizes": dict(SPLIT_SIZES),
        "manifest_file_sha256": _sha(raw), "manifest_sha256": _canonical_sha(retained),
        "source_manifest": PRIOR_MANIFEST, "source_manifest_file_sha256": PRIOR_MANIFEST_SHA256,
        "source_manifest_rows": len(inherited), "selection": "filter split != confirmation; no new sampling",
        "retained_jsonl_lines_byte_identical": True, "retained_ids_prompts_splits_and_provenance_unchanged": True,
        "source_responses_used": False, "model_loaded": False, "gpu_used": False,
        "freshness_scope": "no model outputs produced for retained contexts in the pinned prior run",
        "quarantined_context_count": len(quarantined), "quarantined_ids_sha256": quarantine["ids_sha256"],
        "full_inherited_recognition_validation": full_audit,
        "filtered_four_split_validation": filtered_audit,
        "filtered_exclusion_prompt_count": len(exclusions), "historical_exclusion_sources": exclusion_sources,
        "lineage": lineage,
    }, quarantine
