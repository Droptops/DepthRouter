"""CPU numerical and failure-preservation checks for the tiled capability audit."""

import importlib
import json
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from depth_router.expert_partitions import (
    candidate_combinations,
    partition_indices,
    random_partition,
)


@pytest.fixture
def evaluation(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments"))
    return importlib.import_module("expert_granularity_evaluation")


class TinySwiGLU(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.gate_proj = nn.Linear(4, width, bias=False)
        self.up_proj = nn.Linear(4, width, bias=False)
        self.down_proj = nn.Linear(width, 4, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = TinySwiGLU()
        self.post_attention_layernorm = nn.LayerNorm(4)

    def forward(self, x):
        return x + self.mlp(self.post_attention_layernorm(x))


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TinyLayer()])
        self.model.norm = nn.LayerNorm(4)
        self.lm_head = nn.Linear(4, 19, bias=False)

    def forward(self, input_ids, logits_to_keep=1, **_kwargs):
        hidden = self.model.layers[-1](input_ids)
        return SimpleNamespace(logits=self.lm_head(self.model.norm(hidden))[:, -logits_to_keep:])


class UnlimitedBudget:
    def check(self):
        pass


def fixture_inputs(evaluation):
    torch.manual_seed(17)
    model = TinyModel().to(torch.bfloat16).eval().requires_grad_(False)
    dense = model.model.layers[-1].mlp
    labels = random_partition(16, seed=29)
    old = torch.stack([partition_indices(labels, ids).sort().values for ids in candidate_combinations()])
    groups, records = [], []
    replay = None
    for group_index, valid in enumerate(([True, True], [False, True])):
        replay = {"input_ids": torch.randn(2, 1, 4, dtype=torch.bfloat16)}
        result, capture = evaluation.capture_decode(model, dense, replay)
        capture["valid"] = torch.tensor(valid)
        groups.append(capture)
        top = result.logits[:, -1].float().topk(2, dim=-1).values
        for slot, keep in enumerate(valid):
            if keep:
                records.append({"group": group_index, "slot": slot,
                                "id": f"context-{slot}", "position": group_index + 1,
                                "dense_token": int(result.logits[slot, -1].argmax()),
                                "dense_margin": float(top[slot, 0] - top[slot, 1])})
    old_tuples = {tuple(row.tolist()) for row in old}
    extras = [list(row) for row in combinations(range(16), 12) if row not in old_tuples][:7]
    expanded = torch.cat((old, torch.tensor(extras)), dim=0)
    return model, dense, labels, old, expanded, groups, records, replay


def test_generic_tiled_evaluator_exactly_reproduces_original_measurements(evaluation, tmp_path):
    from expert_construction import evaluate_partition

    model, dense, labels, old, _, groups, records, _ = fixture_inputs(evaluation)
    legacy = evaluate_partition(model, dense, labels, groups, records, tmp_path / "legacy.npz",
                                UnlimitedBudget(), lambda *_args, **_kwargs: None)
    actual = evaluation.evaluate_candidates(
        model, dense, old, groups, records, tmp_path / "generic.npz",
        UnlimitedBudget(), lambda *_args, **_kwargs: None,
        tile_size=5, control_order=partition_indices(labels),
    )
    for name in (*evaluation.CANDIDATE_FIELDS, *evaluation.CONTROL_FIELDS):
        assert np.array_equal(actual[name], legacy[name]), name
    assert actual["complete"].item() is True
    assert actual["measured"].shape == (3, 28)
    assert actual["measured"].all() and actual["control_measured"].all()
    assert np.array_equal(actual["neuron_indices"], old.numpy())
    assert actual["positions"].tolist() == [1, 1, 2]
    assert actual["groups"].tolist() == [0, 0, 1]
    assert actual["slots"].tolist() == [0, 1, 1]
    saved = evaluation.read_candidate_matrix(tmp_path / "generic.json")
    assert np.array_equal(saved["kl"], actual["kl"])
    assert saved["complete"].item() is True


def test_nested_raw_fields_remain_exact_for_different_tile_boundaries(evaluation, tmp_path):
    model, dense, labels, old, expanded, groups, records, _ = fixture_inputs(evaluation)
    common = {"control_order": partition_indices(labels)}
    baseline = evaluation.evaluate_candidates(
        model, dense, old, groups, records, tmp_path / "old.npz", UnlimitedBudget(),
        lambda *_args, **_kwargs: None, tile_size=28, **common)
    refined = evaluation.evaluate_candidates(
        model, dense, expanded, groups, records, tmp_path / "refined.npz", UnlimitedBudget(),
        lambda *_args, **_kwargs: None, tile_size=6, inherited_matrix=baseline,
        inherited_candidate_ids=np.arange(28), **common)
    assert refined["kl"].shape == (3, 35)
    assert refined["inherited_exact"].item() is True
    control = evaluation.assert_inherited_equal(refined, baseline, np.arange(28))
    assert control["monotone_oracle_coverage"] is True
    assert control["new_acceptable_states"] >= control["old_acceptable_states"]
    refined["hidden_mse"][0, 0] += 1
    with pytest.raises(AssertionError, match="hidden_mse"):
        evaluation.assert_inherited_equal(refined, baseline, np.arange(28))


def test_timeout_retains_exact_measured_cells_and_marks_every_other_value(evaluation, tmp_path):
    model, dense, _, old, _, groups, records, _ = fixture_inputs(evaluation)
    progress = []

    class StopAfterFirstTile:
        def check(self):
            if progress:
                raise TimeoutError("registered GPU wall ceiling test")

    path = tmp_path / "partial.npz"
    with pytest.raises(TimeoutError, match="ceiling test"):
        evaluation.evaluate_candidates(
            model, dense, old, groups, records, path, StopAfterFirstTile(),
            lambda *_args, **kwargs: progress.append(kwargs), tile_size=4)
    assert not path.exists()
    partial_path = tmp_path / "partial.incomplete.json"
    saved = evaluation.read_candidate_matrix(partial_path)
    assert saved["complete"].item() is False
    assert saved["measured"].shape == (3, 28)
    assert saved["measured"][:, :4].all()
    assert not saved["measured"][:, 4:].any()
    assert saved["control_measured"].all()
    assert saved["candidate_cells_completed"].item() == 12
    for name in evaluation.FLOAT_FIELDS:
        assert np.isfinite(saved[name][:, :4]).all()
        assert np.isnan(saved[name][:, 4:]).all()
    assert (saved["candidate_token"][:, 4:] == -1).all()
    assert saved["error_type"].item() == "TimeoutError"
    assert saved["context_ids"].tolist() == [row["id"] for row in records]
    with pytest.raises(FileExistsError, match="overwrite"):
        evaluation.evaluate_candidates(model, dense, old, groups, records, path,
                                       UnlimitedBudget(), lambda *_args, **_kwargs: None)


def test_control_failure_preserves_partial_artifact_and_never_interprets_sparse_data(evaluation, tmp_path):
    model, dense, _, old, _, groups, records, _ = fixture_inputs(evaluation)
    records[0]["dense_token"] += 1
    with pytest.raises(AssertionError, match="dense-token metadata"):
        evaluation.evaluate_candidates(model, dense, old, groups, records, tmp_path / "bad.npz",
                                       UnlimitedBudget(), lambda *_args, **_kwargs: None)
    partial = evaluation.read_candidate_matrix(tmp_path / "bad.incomplete.json")
    assert partial["interrupted_phase"].item() == "controls"
    assert not partial["measured"].any()
    assert not partial["control_measured"].any()


def test_parity_checks_selected_generic_masks_and_restores_model(evaluation):
    model, dense, _, _, expanded, _, _, replay = fixture_inputs(evaluation)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    checks = evaluation.validate_cached_candidates(
        model, dense, expanded, replay, UnlimitedBudget(), candidate_ids=[0, 27, 28, 34])
    assert [row["candidate_id"] for row in checks] == [0, 27, 28, 34]
    assert all(row["exact_logits"] and row["max_abs_logit_error"] == 0 for row in checks)
    assert model.model.layers[-1].mlp is dense
    assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())
    assert not dense._forward_hooks and not dense._forward_pre_hooks


def test_failed_fullmodel_substitution_restores_original_layer(evaluation, monkeypatch):
    model, dense, _, old, _, _, _, replay = fixture_inputs(evaluation)
    original = model.forward

    def fail(**kwargs):
        if model.model.layers[-1].mlp is not dense:
            raise RuntimeError("injected fullmodel error")
        return original(**kwargs)

    monkeypatch.setattr(model, "forward", fail)
    with pytest.raises(RuntimeError, match="injected fullmodel"):
        evaluation.validate_cached_candidates(model, dense, old, replay, UnlimitedBudget(),
                                              candidate_ids=[0])
    assert model.model.layers[-1].mlp is dense
    assert not dense._forward_hooks and not dense._forward_pre_hooks


@pytest.mark.parametrize("indices,error", [
    (torch.tensor([[0.0, 1.0]]), "int64"),
    (torch.tensor([[1, 0]]), "ascending"),
    (torch.tensor([[0, 0]]), "ascending"),
    (torch.tensor([[0, 16]]), "outside"),
    (torch.tensor([[0, 1], [0, 1]]), "unique"),
])
def test_noncanonical_masks_are_rejected(evaluation, indices, error):
    with pytest.raises(ValueError, match=error):
        evaluation.validate_indices(indices, 16)


def test_invalid_tile_size_or_record_order_fails_before_artifact_creation(evaluation, tmp_path):
    model, dense, _, old, _, groups, records, _ = fixture_inputs(evaluation)
    for size in (0, 29, True, 1.5):
        with pytest.raises(ValueError, match="tile_size"):
            evaluation.evaluate_candidates(model, dense, old, groups, records, tmp_path / "none.npz",
                                           UnlimitedBudget(), lambda *_args, **_kwargs: None,
                                           tile_size=size)
    with pytest.raises(ValueError, match="group/slot order"):
        evaluation.evaluate_candidates(model, dense, old, groups, records[::-1], tmp_path / "none.npz",
                                       UnlimitedBudget(), lambda *_args, **_kwargs: None)
    assert list(tmp_path.iterdir()) == []


def test_chunk_archive_is_lossless_shared_layout_and_detects_corruption(evaluation, tmp_path):
    model, dense, labels, old, _, groups, records, _ = fixture_inputs(evaluation)
    matrix = evaluation.evaluate_candidates(
        model, dense, old, groups, records, tmp_path / "source.json", UnlimitedBudget(),
        lambda *_args, **_kwargs: None, partition=labels.numpy(), partition_sha256="a" * 64,
        selected_experts=np.asarray(candidate_combinations()), split="calibration")
    manifest = evaluation.write_candidate_matrix(tmp_path / "small-chunks.json", matrix, chunk_size=5)
    assert len(manifest["chunks"]) == 6
    assert all(item["column_stop"] - item["column_start"] <= 5 for item in manifest["chunks"])
    assert len(list(tmp_path.glob("candidate-layout-*.npz"))) == 1
    reconstructed = evaluation.read_candidate_matrix(tmp_path / "small-chunks.json")
    assert set(reconstructed) == set(matrix)
    for key, value in matrix.items():
        assert np.array_equal(reconstructed[key], value), key
        assert reconstructed[key].dtype == np.asarray(value).dtype
    chunk = tmp_path / manifest["chunks"][0]["file"]
    chunk.write_bytes(chunk.read_bytes() + b"corruption")
    with pytest.raises(AssertionError, match="archive hash mismatch"):
        evaluation.read_candidate_matrix(tmp_path / "small-chunks.json")


def test_reader_rejects_manifest_column_gaps_and_parent_path_references(evaluation, tmp_path):
    model, dense, _, old, _, groups, records, _ = fixture_inputs(evaluation)
    evaluation.evaluate_candidates(model, dense, old, groups, records, tmp_path / "matrix.json",
                                   UnlimitedBudget(), lambda *_args, **_kwargs: None)
    path = tmp_path / "matrix.json"
    original = json.loads(path.read_text())
    malformed = json.loads(path.read_text())
    malformed["chunks"][0]["column_start"] = 1
    path.write_text(json.dumps(malformed))
    with pytest.raises(ValueError, match="exact bounded partition"):
        evaluation.read_candidate_matrix(path)
    original["layout"]["file"] = "../outside.npz"
    path.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="sibling filenames"):
        evaluation.read_candidate_matrix(path)


def test_global_candidate_ids_can_be_sparse_and_parity_includes_both_controls(evaluation, tmp_path):
    model, dense, labels, old, _, groups, records, replay = fixture_inputs(evaluation)
    global_ids = np.arange(28, dtype=np.int64) * 17
    matrix = evaluation.evaluate_candidates(
        model, dense, old, groups, records, tmp_path / "global.json", UnlimitedBudget(),
        lambda *_args, **_kwargs: None, candidate_ids=global_ids)
    result = evaluation.assert_inherited_equal(matrix, matrix, global_ids)
    assert result["exact_all_raw_candidate_fields"] is True
    checks = evaluation.validate_cached_candidates(
        model, dense, old, replay, UnlimitedBudget(), candidate_ids=[0, 27],
        control_order=partition_indices(labels))
    assert [row["kind"] for row in checks] == ["cheap", "cheap", "control", "permuted_control"]
    assert all(row["exact_logits"] for row in checks)
