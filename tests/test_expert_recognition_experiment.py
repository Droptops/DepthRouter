"""CPU-only checks of the fresh confirmation and recognition stop rules."""

import copy
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from depth_router.expert_audit_metrics import candidate_combinations, select_fixed_candidate


@pytest.fixture
def experiment(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments"))
    old_threads = torch.get_num_threads()
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    yield importlib.import_module("expert_recognition")
    torch.set_num_threads(old_threads)
    torch.backends.cuda.matmul.allow_tf32 = old_tf32
    torch.backends.cudnn.allow_tf32 = old_cudnn_tf32


def confirmation_fixture(*, matched_coverage=18, matched_fixed=10,
                         ood_coverage=18, ood_fixed=10):
    count = 40
    kl = np.full((count, 28), 0.1)
    tokens = np.full((count, 28), 2)
    records, rows = [], []
    for stratum, offset, coverage, fixed in (
        ("matched", 0, matched_coverage, matched_fixed),
        ("ood", 20, ood_coverage, ood_fixed),
    ):
        for index in range(5):
            rows.append({"id": f"{stratum}-{index}", "confirmation_stratum": stratum})
        records.extend({"id": f"{stratum}-{index // 4}"} for index in range(20))
        kl[offset:offset + fixed, 0] = 0.005
        tokens[offset:offset + fixed, 0] = 1
        kl[offset + fixed:offset + coverage, 1] = 0.005
        tokens[offset + fixed:offset + coverage, 1] = 1
    matrix = {
        "kl": kl, "candidate_token": tokens, "dense_token": np.ones(count, dtype=int),
        "hidden_mse": np.ones((count, 28)), "dense_margin": np.ones(count),
        "context_ids": np.array([row["id"] for row in records]),
        "positions": np.tile([1, 3, 7, 11], 10),
        "control_kl": np.zeros(count), "control_token": np.ones(count, dtype=int),
        "selected_experts": np.array(candidate_combinations()),
        "partition": np.arange(4864) % 8,
    }
    old_kl = np.full((8, 28), 0.1)
    old_agreement = np.zeros((8, 28), dtype=bool)
    old_kl[:, 0] = 0.001
    old_agreement[:, 0] = True
    fixed = select_fixed_candidate(old_kl, old_agreement)
    config = {"bootstrap_repetitions": 100, "bootstrap_seed": 17}
    return matrix, records, rows, fixed, config


def recognition_fixture():
    passing = {
        "greedy_agreement": 1.0, "mean_kl": 0.001, "cheap_fraction": 0.5,
        "quality_gate": {"point_pass": True, "statistically_resolved_pass": True},
        "has_nonzero_cheap_use": True,
    }
    splits = ("development", "heldout_matched", "heldout_ood")
    metrics = {family: {str(seed): {split: copy.deepcopy(passing) for split in splits}
                        for seed in (17, 29, 43)} for family in ("portfolio", "binary")}
    paired = {split: {"cheap_fraction": {"estimate": 0.1, "lower": 0.02, "upper": 0.18}}
              for split in splits}
    config = {"seeds": [17, 29, 43], "primary_runtime_seed": 17}
    return metrics, paired, config


def test_slice_matrix_preserves_metadata_when_its_first_axis_equals_state_count(experiment):
    matrix = {
        "kl": np.arange(28 * 28).reshape(28, 28),
        "context_ids": np.array([f"context-{index}" for index in range(28)]),
        "positions": np.ones(28),
        "selected_experts": np.array(candidate_combinations()),
        "partition": np.arange(28),
        "complete": np.array(True),
    }
    selected = np.arange(28) % 2 == 0
    sliced = experiment.slice_matrix(matrix, selected)
    assert sliced["kl"].shape == (14, 28)
    np.testing.assert_array_equal(sliced["context_ids"], matrix["context_ids"][selected])
    assert sliced["selected_experts"] is matrix["selected_experts"]
    assert sliced["partition"] is matrix["partition"]
    assert sliced["complete"] is matrix["complete"]
    assert matrix["kl"].shape == (28, 28)


def test_confirmation_requires_each_stratum_not_pooled_success(experiment):
    inputs = confirmation_fixture(ood_coverage=15)
    result = experiment.confirmation_decision(*inputs)
    assert result["strata"]["matched"]["engineering_gate"]["point_pass"]
    assert not result["strata"]["ood"]["engineering_gate"]["point_pass"]
    assert not result["point_pass"]
    assert not result["confirmation_is_training_data"]
    assert result["failure_action"] == "stop_without_switching_partition_or_consuming_other_splits"


def test_confirmation_checks_portfolio_advantage_separately_from_coverage(experiment):
    result = experiment.confirmation_decision(*confirmation_fixture(ood_fixed=16))
    assert result["strata"]["ood"]["cheap_coverage"] == 0.9
    assert result["strata"]["ood"]["portfolio_advantage"] == pytest.approx(0.025)
    assert not result["point_pass"]
    passing = experiment.confirmation_decision(*confirmation_fixture())
    assert passing["point_pass"]
    assert set(passing["strata"]) == {"matched", "ood"}


def test_confirmation_numerical_failure_is_not_a_sparse_success(experiment):
    matrix, records, rows, fixed, config = confirmation_fixture()
    matrix["control_token"][20] = 2
    result = experiment.confirmation_decision(matrix, records, rows, fixed, config)
    assert not result["point_pass"]


def test_confirmation_missing_stratum_is_an_error(experiment):
    matrix, records, rows, fixed, config = confirmation_fixture()
    for row in rows:
        row["confirmation_stratum"] = "matched"
    with pytest.raises(ValueError, match="missing ood"):
        experiment.confirmation_decision(matrix, records, rows, fixed, config)


def test_recognition_requires_three_seeds_and_all_splits(experiment):
    metrics, paired, config = recognition_fixture()
    assert experiment.recognition_decision(metrics, paired, config)["portfolio"][
        "eligible_for_runtime_quality"]
    metrics["portfolio"]["43"]["heldout_ood"]["quality_gate"]["point_pass"] = False
    decision = experiment.recognition_decision(metrics, paired, config)
    assert not decision["portfolio"]["eligible_for_runtime_quality"]
    assert decision["binary"]["eligible_for_runtime_quality"]
    assert decision["portfolio"]["individual_seeds"]["17"]["quality_point_pass"]
    assert decision["portfolio"]["primary_runtime_seed"] == 17
    del metrics["binary"]["29"]["development"]
    with pytest.raises(ValueError, match="every preregistered evaluation split"):
        experiment.recognition_decision(metrics, paired, config)


def test_all_dense_is_not_promoted_even_with_perfect_fidelity(experiment):
    metrics, paired, config = recognition_fixture()
    for family in metrics:
        metrics[family]["29"]["development"].update(
            {"cheap_fraction": 0, "has_nonzero_cheap_use": False})
    result = experiment.recognition_decision(metrics, paired, config)
    assert all(not result[family]["eligible_for_runtime_quality"] for family in result)
    assert all(result[family]["all_seed_quality_point_pass"] for family in result)


@pytest.mark.parametrize("lower", [0.0, -0.01])
def test_portfolio_needs_resolved_paired_advantage_on_both_heldouts(experiment, lower):
    metrics, paired, config = recognition_fixture()
    paired["heldout_ood"]["cheap_fraction"]["lower"] = lower
    result = experiment.recognition_decision(metrics, paired, config)
    assert not result["portfolio"]["eligible_for_runtime_quality"]
    assert not result["portfolio"]["portfolio_advantage_pass"]
    assert result["binary"]["eligible_for_runtime_quality"]
    assert result["binary"]["portfolio_advantage_pass"] is None


def test_feature_order_tracks_valid_matrix_state_order(experiment):
    groups = [
        {"x": torch.arange(12).reshape(3, 1, 4), "valid": torch.tensor([True, False, True])},
        {"x": torch.arange(12, 24).reshape(3, 1, 4), "valid": torch.tensor([False, True, True])},
    ]
    expected = torch.tensor([[0, 1, 2, 3], [8, 9, 10, 11], [16, 17, 18, 19], [20, 21, 22, 23]])
    torch.testing.assert_close(experiment.features(groups), expected)


class FakeModel(nn.Module):
    """Only model metadata; its overridden cuda never touches a CUDA device."""

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Identity() for _ in range(24)])
        self.model.layers[-1].mlp = nn.Module()
        self.model.layers[-1].mlp.gate_proj = nn.Linear(1, 4864, bias=False)
        self.config = SimpleNamespace(_attn_implementation="cpu_mock")

    def cuda(self, *args, **kwargs):
        return self


def test_failed_confirmation_exits_before_training_or_other_split_access(
    experiment, monkeypatch, tmp_path
):
    model = FakeModel()
    fake_transformers = SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *args, **kwargs: model),
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: SimpleNamespace()),
        __version__="cpu-test-no-model-load",
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(experiment, "ROOT", tmp_path)
    monkeypatch.setattr(experiment, "verify_sources", lambda *_args: None)
    monkeypatch.setattr(experiment.torch.cuda, "get_device_name", lambda: "CPU mocked metadata")
    monkeypatch.setattr(experiment.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=""))
    output = tmp_path / "run"
    output.mkdir()
    experiment.write_json(output / "frozen_partition.json", (np.arange(4864) % 8).tolist())
    experiment.write_json(output / "frozen_fixed_selection.json", {"index": 8})
    # A single JSON object is also one JSONL row; no real data/model access.
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"id": "c0", "split": "confirmation"}) + "\n")
    seen = []

    def evaluate(*args):
        split = args[5]
        seen.append(split)
        assert split == "confirmation"
        return [], [], {}, {}

    monkeypatch.setattr(experiment, "evaluate_split", evaluate)
    monkeypatch.setattr(experiment, "validate_cached_candidates", lambda *args: [])
    monkeypatch.setattr(experiment, "confirmation_decision", lambda *args: {"point_pass": False})
    monkeypatch.setattr(experiment, "train_selector_pair", lambda *args, **kwargs:
                        pytest.fail("training ran after failed confirmation"))
    monkeypatch.setattr(experiment, "run_runtime", lambda *args, **kwargs:
                        pytest.fail("runtime ran after failed confirmation"))
    config = {"gpu_wall_budget_seconds": 60, "model": "fake", "revision": "fake",
              "attention": "sdpa", "manifest": "manifest.jsonl"}
    experiment.run(config, output)
    assert seen == ["confirmation"]
    result = json.loads((output / "promotion_decision.json").read_text())
    assert result["confirmation_pass"] is False and result["promote"] is False
    assert result["recognition"] == result["runtime"] == "not_run_confirmation_failed"
    assert result["original_weights_unchanged"] is True
    assert result["measured_latency"] is None and result["measured_dram_bytes"] is None
    assert not (output / "incomplete.json").exists()
