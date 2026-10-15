"""Report checks using tiny stored artifacts, without loading a language model."""

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from depth_router.expert_audit_metrics import (
    candidate_combinations,
    paired_context_difference,
    portfolio_gate,
    select_fixed_candidate,
    summarize_candidates,
)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def report(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments"))
    return importlib.import_module("expert_construction_report")


@pytest.fixture
def stored_run(tmp_path):
    run = tmp_path / "artifacts" / "expert_construction" / "test-run"
    run.mkdir(parents=True)
    config = {
        "model": "fixture/Qwen", "revision": "model-pin", "base_commit": "research-pin",
        "seeds": [17], "assignment": {"steps": 1}, "coactivation": {"sketch_dim": 2},
        "continuation_positions": [1, 3, 7, 11], "max_prompt_tokens": 512,
        "batch_size": 2, "gpu_wall_budget_seconds": 14400,
        "bootstrap_repetitions": 40, "bootstrap_seed": 17,
    }
    _write(run / "preregistration.json", {"config": config, "source_sha256": {}})
    _write(run / "environment.json", {"gpu": "fixture GPU", "dtype": "torch.bfloat16"})
    _write(run / "registered_sources/data/expert_audit/source_provenance.json", {
        "dataset": "test-source", "revision": "source-pin", "license": "test-license",
    })
    _write(run / "registered_sources/data/expert_audit/data_audit.json", {
        "historical_prompt_count": 10, "manifest_file_sha256": "manifest-hash",
    })
    splits = ("calibration", "development", "heldout_matched", "heldout_ood")
    kl = np.full((4, 28), 0.1)
    tokens = np.ones((4, 28), dtype=np.int64)
    # Exactly half the observed states have an acceptable candidate. This fails
    # coverage and has zero portfolio benefit over the development fixed subset.
    kl[:2, 0] = 0.001
    tokens[:2, 0] = 0
    matrix = {
        "kl": kl, "candidate_token": tokens, "dense_token": np.zeros(4, dtype=np.int64),
        "hidden_mse": np.ones((4, 28)), "context_ids": np.array(["a", "a", "b", "b"]),
        "positions": np.array([1, 3, 1, 3]), "selected_experts": candidate_combinations(),
        "dense_margin": np.ones(4), "control_kl": np.zeros(4),
        "control_token": np.zeros(4, dtype=np.int64),
        "permuted_control_kl": np.zeros(4),
        "permuted_control_token": np.array([0, 0, 0, 1]),
    }
    selection = select_fixed_candidate(kl, tokens == 0)
    summaries = {f"{arm}-17": {} for arm in "ABCD"}
    fixed = {key: selection for key in summaries}
    _write(run / "fixed_selection.json", fixed)
    for split in splits:
        rows = [{"id": context, "position": int(position), "task": "general_qa"}
                for context, position in zip(matrix["context_ids"], matrix["positions"], strict=True)]
        (run / f"states-{split}.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
        )
        _write(run / f"capture-{split}.json", {
            "contexts_requested": 2, "contexts_evaluated": 2, "states": 4,
            "zero_state_context_ids": [], "prompt_truncations": 1,
        })
        for key, split_summaries in summaries.items():
            np.savez_compressed(run / f"matrix-{split}-{key}.npz", **matrix)
            if split == "calibration":
                continue
            split_summaries[split] = summarize_candidates(
                kl, tokens == 0, matrix["hidden_mse"], matrix["context_ids"], matrix["dense_margin"],
                fixed_selection=selection, split=split, control_kl=matrix["control_kl"],
                control_agreement=matrix["control_token"] == matrix["dense_token"],
                bootstrap_repetitions=40,
            )
    _write(run / "all_summaries.json", summaries)
    zero_difference = paired_context_difference(np.zeros((1, 4)), np.zeros((1, 4)),
                                                matrix["context_ids"], bootstrap_repetitions=40)
    _write(run / "paired_D_minus_C.json", {
        split: {metric: zero_difference for metric in ("cheap_coverage", "portfolio_advantage")}
        for split in splits
    })
    _write(run / "promotion_decision.json", {
        "phase1_complete": True, "phase1_gates": {
            key: portfolio_gate(summaries[key]) for key in summaries
        }, "development_selected_family": "A", "selected_family_eligible": False,
        "phase2": "skipped_phase1_gate_failed", "phase3": "not_run_requires_phase2_quality_pass",
        "promote": False, "original_weights_unchanged": True,
        "measured_latency": None, "measured_dram_bytes": None,
        "physical_counter_attempt": "not_attempted_prerequisite_quality_gate",
        "single_gpu_wall_elapsed_seconds": 30.0,
    })
    return run


def test_report_preserves_negative_gates_and_distinguishes_arithmetic_from_costs(report, stored_run):
    full, concise = report.render_reports(stored_run, root=stored_run.parents[2])
    assert "16 complete candidate matrices" in full
    assert "1,792 state/candidate measurements" in full
    assert "failed the predeclared additional family-level continuation criterion" in concise
    assert "insufficient acceptable cheap candidates" in full
    assert "insufficient portfolio advantage over the fixed subset" in full
    assert "skipped_phase1_gate_failed" in full
    assert "Measured latency: `None`" in full
    assert "Null means unknown, not zero" in full
    assert "1.04%" in full
    assert "never a deployable selector" in concise
    assert "| D-17 | heldout_ood |" in full
    assert "fit-data diagnostic" in full
    assert "75.000 [" in full  # Permuted control has a mismatch and is reported.
    assert "| permuted_control |" in full
    assert "| control |" in full
    assert "Missing planned states (EOS)" in full
    assert "| calibration | 2 | 2 | 4 | 4 | 1 |" in full
    assert "DOT-MoE" in full and "https://arxiv.org/abs/1603.00336" in full
    assert "raw matrix contexts" not in full


def test_outcome_preserves_individual_passes_and_uses_selected_family_failure_only(report):
    gates = {}
    passing = {"B-17", "B-29", "C-17", "C-29"}
    for arm in "ABCD":
        for seed in (17, 29, 43):
            key = f"{arm}-{seed}"
            gates[key] = {
                "point_pass": key in passing, "statistically_resolved_pass": key in passing,
                "failure_reasons": [] if key in passing else
                ["ood:insufficient_cheap_candidate_coverage"] +
                ([] if arm == "C" else ["ood:insufficient_advantage_over_fixed_subset"]),
            }
    outcome = report._outcome({
        "complete": True, "config": {"seeds": [17, 29, 43]},
        "promotion": {"development_selected_family": "C", "selected_family_eligible": False,
                      "phase1_gates": gates},
    })
    assert "4 of 12 individual partitions passed" in outcome
    assert "B-17, B-29, C-17, C-29" in outcome
    assert "4 also passed" in outcome
    assert "additional family-level continuation criterion" in outcome
    assert "insufficient acceptable cheap candidates" in outcome
    assert "insufficient portfolio advantage" not in outcome


def test_report_rejects_summary_raw_disagreement(report, stored_run):
    path = stored_run / "all_summaries.json"
    summaries = json.loads(path.read_text())
    summaries["D-17"]["heldout_ood"]["cheap_coverage"] = 1.0
    _write(path, summaries)
    with pytest.raises(ValueError, match="stored summary disagrees with raw matrix"):
        report.render_reports(stored_run)


def test_complete_report_requires_every_raw_matrix(report, stored_run):
    # Remove this fixture file only; a completed experiment cannot conceal a
    # failed arm or silently replace missing evidence with a reported summary.
    (stored_run / "matrix-heldout_ood-D-17.npz").unlink()
    with pytest.raises(ValueError, match="missing raw candidate evidence"):
        report.render_reports(stored_run)


def test_partial_run_is_reported_as_incomplete_without_capability_success(report, tmp_path):
    run = tmp_path / "partial"
    run.mkdir()
    _write(run / "preregistration.json", {"config": {
        "model": "fixture/Qwen", "revision": "pin", "base_commit": "base", "seeds": [17],
        "assignment": {}, "coactivation": {}, "continuation_positions": [1, 3, 7, 11],
        "max_prompt_tokens": 512, "batch_size": 16, "gpu_wall_budget_seconds": 14400,
        "bootstrap_repetitions": 40, "bootstrap_seed": 17,
    }})
    _write(run / "incomplete.json", {
        "error_type": "TimeoutError", "error": "preregistered ceiling reached",
        "single_gpu_wall_elapsed_seconds": 14401,
    })
    full, concise = report.render_reports(run, root=tmp_path)
    assert "capability audit is incomplete" in concise
    assert "no capability or deployment success is established" in full
    assert "TimeoutError: preregistered ceiling reached" in full
    assert "0 complete candidate matrices" in full
    assert "14401" in full


def test_report_rejects_context_or_position_misalignment(report, stored_run):
    path = stored_run / "states-heldout_ood.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["id"] = "wrong-context"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="contexts do not align"):
        report.render_reports(stored_run)
