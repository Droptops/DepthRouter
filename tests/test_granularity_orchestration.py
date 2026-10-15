"""CPU-only tests of preregistration locks, selection order and seal boundaries."""

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

PROJECT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((PROJECT / "configs/expert_granularity.json").read_text())


@pytest.fixture
def experiment(monkeypatch):
    monkeypatch.syspath_prepend(str(PROJECT / "experiments"))
    old_threads = torch.get_num_threads()
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    yield importlib.import_module("expert_granularity")
    torch.set_num_threads(old_threads)
    torch.backends.cuda.matmul.allow_tf32 = old_tf32
    torch.backends.cudnn.allow_tf32 = old_cudnn_tf32


def alter(value):
    if isinstance(value, bool):
        return not value
    if isinstance(value, (float, int)):
        return value + 1
    if isinstance(value, str):
        return value + "-changed"
    if isinstance(value, list):
        return value[::-1]
    if isinstance(value, dict):
        modified = copy.deepcopy(value)
        first = next(iter(modified))
        modified[first] = alter(modified[first])
        return modified
    raise AssertionError("unhandled config type")


@pytest.mark.parametrize("field", list(CONFIG))
def test_every_registered_configuration_field_is_locked(experiment, field):
    changed = copy.deepcopy(CONFIG)
    changed[field] = alter(changed[field])
    with pytest.raises(ValueError, match="protocol"):
        experiment.validate_config(changed)


def test_missing_and_unknown_configuration_fields_are_rejected(experiment):
    experiment.validate_config(copy.deepcopy(CONFIG))
    changed = copy.deepcopy(CONFIG)
    changed.pop("revision")
    with pytest.raises(ValueError, match="protocol"):
        experiment.validate_config(changed)
    changed = {**CONFIG, "unregistered_threshold": 0.02}
    with pytest.raises(ValueError, match="protocol"):
        experiment.validate_config(changed)


def test_partition_and_parity_sampling_are_frozen_before_outputs(experiment):
    parent = np.arange(4864, dtype=np.int64) // 608
    first = experiment.frozen_partitions(parent, [17, 29, 43])
    second = experiment.frozen_partitions(parent, [17, 29, 43])
    assert first == second
    original = np.asarray(first["original_candidate_ids"])
    new = np.setdiff1d(np.arange(1820), original)
    for seed in [17, 29, 43]:
        labels = np.asarray(first["children"][str(seed)])
        np.testing.assert_array_equal(labels // 2, parent)
        parity = first["parity_ids"][str(seed)]
        expected = set(original) | {new[0], new[-1]} | set(
            np.random.default_rng(seed).choice(new, 16, replace=False))
        assert parity == sorted(expected)


def test_seal_files_detects_mutation_without_accessing_model(experiment, tmp_path):
    file = tmp_path / "portfolio_selection.json"
    file.write_text('{"17": {"candidate_ids": [1, 2]}}')
    seal = experiment.seal_files(tmp_path, [file.name], "calibration_seal.json")
    experiment.verify_seal(seal, tmp_path)
    file.write_text('{"17": {"candidate_ids": [2, 3]}}')
    with pytest.raises(RuntimeError, match="seal changed"):
        experiment.verify_seal(seal, tmp_path)


def test_verify_sources_checks_both_working_source_snapshot_and_frozen_artifacts(
    experiment, monkeypatch, tmp_path
):
    root, output = tmp_path / "repo", tmp_path / "run"
    root.mkdir()
    snapshot = output / "registered_sources"
    snapshot.mkdir(parents=True)
    (root / "source.py").write_text("frozen source")
    (snapshot / "source.py").write_text("frozen source")
    (output / "partitions.json").write_text("frozen partition")
    registration = {
        "source_sha256": {"source.py": experiment.sha_file(root / "source.py")},
        "frozen_artifact_sha256": {"partitions.json": experiment.sha_file(output / "partitions.json")},
    }
    experiment.write_json(output / "preregistration.json", registration)
    monkeypatch.setattr(experiment, "ROOT", root)
    experiment.verify_sources(output)
    (snapshot / "source.py").write_text("modified snapshot")
    with pytest.raises(RuntimeError, match="source changed"):
        experiment.verify_sources(output)
    (snapshot / "source.py").write_text("frozen source")
    (root / "source.py").write_text("modified working source")
    with pytest.raises(RuntimeError, match="source changed"):
        experiment.verify_sources(output)
    (root / "source.py").write_text("frozen source")
    (output / "partitions.json").write_text("modified partition")
    with pytest.raises(RuntimeError, match="seal changed"):
        experiment.verify_sources(output)


class FakeModel(nn.Module):
    """Metadata only; cuda is overridden and no decoder is ever executed."""

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Identity() for _ in range(24)])
        self.model.layers[-1].mlp = nn.Module()
        self.model.layers[-1].mlp.gate_proj = nn.Linear(1, 4864, bias=False)
        self.config = SimpleNamespace(_attn_implementation="cpu-mocked")

    def cuda(self, *args, **kwargs):
        return self


def prepare_mocked_run(experiment, monkeypatch, tmp_path, *, tamper_before_heldout=False):
    model = FakeModel()
    fake_transformers = SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *args, **kwargs: model),
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: SimpleNamespace()),
        __version__="unit-test-no-model-execution",
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(experiment, "ROOT", tmp_path)
    monkeypatch.setattr(experiment, "verify_sources", lambda *_args: None)
    monkeypatch.setattr(experiment.torch.cuda, "get_device_name", lambda: "CPU mocked metadata")
    monkeypatch.setattr(experiment.platform, "platform", lambda: "unit-test")
    monkeypatch.setattr(experiment.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=""))
    # Avoid even allocating the 3 x 1820 x 3648 index arrays in a CPU orchestration
    # test; the real packed-index evaluator is tested separately.
    monkeypatch.setattr(experiment, "partition_indices", lambda *args: np.array([0], dtype=np.int64))
    monkeypatch.setattr(experiment, "validate_cached_candidates", lambda *args, **kwargs: [])
    output = tmp_path / "run"
    output.mkdir()
    config = copy.deepcopy(CONFIG)
    config["manifest"] = "manifest.jsonl"
    config["bootstrap_repetitions"] = 20  # Test-only direct run, never registration.
    parent = np.arange(4864, dtype=np.int64) // 608
    experiment.write_json(output / "partitions.json", experiment.frozen_partitions(parent, [17, 29, 43]))
    experiment.write_json(output / "preregistration.json", {"test_only": True})
    rows = [{"id": f"{split}-{index}", "split": split} for split in
            ("calibration", "development", "heldout_matched", "heldout_ood") for index in range(2)]
    (tmp_path / "manifest.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    captured, selected_on, fixed_on = [], [], []

    def collect(_model, _tokenizer, _dense, split_rows, _config, _output, _log, _budget):
        split = split_rows[0]["split"]
        captured.append(split)
        if split != "calibration":
            assert (output / "calibration_seal.json").exists()
        if split.startswith("heldout"):
            assert (output / "heldout_seal.json").exists()
        records = [{"id": f"{split}-{index // 4}", "position": [1, 3, 7, 11][index % 4]}
                   for index in range(8)]
        return [], records, {"split": split}

    old_ids = experiment.original_candidate_indices()
    new_id = next(index for index in range(1820) if index not in old_ids)

    def evaluate(_model, _dense, _indices, _groups, records, _path, _budget, _log, **kwargs):
        ids = np.asarray(kwargs["candidate_ids"])
        kl = np.full((8, len(ids)), 0.1)
        token = np.zeros((8, len(ids)), dtype=np.int64)
        for target, positions in [(old_ids[0], [0, 2, 4, 6]), (new_id, [1, 3, 5, 7])]:
            if target in ids:
                column = int(np.flatnonzero(ids == target)[0])
                kl[positions, column] = 0.001
                token[positions, column] = 1
        return {"kl": kl, "candidate_token": token, "dense_token": np.ones(8, dtype=np.int64),
                "candidate_ids": ids, "context_ids": np.array([row["id"] for row in records]),
                "positions": np.array([row["position"] for row in records]),
                "partition": kwargs["partition"], "partition_sha256": kwargs["partition_sha256"],
                "selected_experts": kwargs["selected_experts"], "split": kwargs["split"],
                "control_kl": np.zeros(8), "control_token": np.ones(8, dtype=np.int64)}

    original_select = experiment.select_calibration_portfolio
    original_fixed = experiment.select_development_fixed

    def select(matrix):
        selected_on.append(matrix["split"])
        return original_select(matrix)

    def fixed(matrix, ids):
        fixed_on.append(matrix["split"])
        return original_fixed(matrix, ids)

    def log_factory(_output):
        def log(event, **_fields):
            if tamper_before_heldout and event == "fixed_comparators_sealed_before_heldouts":
                (output / "portfolio_selection.json").write_text("tampered after sealing")
        return log

    monkeypatch.setattr(experiment, "collect_states", collect)
    monkeypatch.setattr(experiment, "evaluate_candidates", evaluate)
    monkeypatch.setattr(experiment, "select_calibration_portfolio", select)
    monkeypatch.setattr(experiment, "select_development_fixed", fixed)
    monkeypatch.setattr(experiment, "Log", log_factory)
    return config, output, captured, selected_on, fixed_on


def test_orchestrator_fits_calibration_only_and_freezes_fixed_before_heldout(
    experiment, monkeypatch, tmp_path
):
    config, output, captured, selected, fixed = prepare_mocked_run(experiment, monkeypatch, tmp_path)
    experiment.run(config, output)
    assert captured == ["calibration", "development", "heldout_matched", "heldout_ood"]
    assert selected == ["calibration"] * 3
    assert fixed == ["development"] * 7  # old28 + selected28/full1820 for each seed.
    decision = json.loads((output / "promotion_decision.json").read_text())
    assert decision["status"] == "completed" and decision["capability_pass"]
    assert decision["original_weights_unchanged"]
    assert decision["promote"] is False and decision["global_bandwidth_solver_promotion"] is False
    assert decision["runtime_training"] == decision["runtime_timing"] == "not_run_out_of_scope"
    assert decision["measured_latency"] is None and decision["measured_dram_bytes"] is None
    assert (output / "final_weight_sha256.json").exists()


def test_mutated_calibration_seal_aborts_before_any_heldout_capture(
    experiment, monkeypatch, tmp_path
):
    config, output, captured, selected, fixed = prepare_mocked_run(
        experiment, monkeypatch, tmp_path, tamper_before_heldout=True)
    with pytest.raises(RuntimeError, match="seal changed"):
        experiment.run(config, output)
    assert captured == ["calibration", "development"]
    assert selected == ["calibration"] * 3 and fixed == ["development"] * 7
    assert (output / "incomplete.json").exists()
    assert not (output / "promotion_decision.json").exists()
