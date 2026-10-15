"""CPU tests for timing authorization, causal dispatch and numerical controls."""

import copy
import hashlib
import importlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from depth_router.expert_partitions import random_partition


@pytest.fixture
def timing(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments"))
    return importlib.import_module("expert_recognition_timing")


def passing_evidence():
    row = {"greedy_agreement": 0.995, "mean_kl": 0.002, "cheap_fraction": 0.4,
           "quality_gate": {"point_pass": True}}
    splits = ("development", "heldout_matched", "heldout_ood")
    seed_metrics = {seed: {split: copy.deepcopy(row) for split in splits} for seed in (17, 29, 43)}
    runtime_metrics = {split: {**copy.deepcopy(row), "batch_size": 1,
                              "source": "fresh_batch1_fullmodel_dense_prefix_capture"}
                       for split in splits}
    advantage = {"heldout_matched": 0.02, "heldout_ood": 0.01}
    return seed_metrics, runtime_metrics, advantage


def test_gate_requires_all_seed_and_runtime_evidence(timing):
    seeds, runtime, advantage = passing_evidence()
    gate = timing.require_quality_gate(seeds, runtime, policy_family="portfolio",
                                      portfolio_advantage_ci_low=advantage)
    assert gate["passed"] is True
    assert set(gate["seed_evidence"]) == {"17", "29", "43"}
    assert gate["portfolio_advantage_ci95_lower"] == advantage
    binary = timing.require_quality_gate(seeds, runtime, policy_family="binary")
    assert binary["passed"] is True
    assert binary["portfolio_advantage_ci95_lower"] == {}


@pytest.mark.parametrize("field,value,error", [
    ("greedy_agreement", 0.9899, "agreement"),
    ("mean_kl", 0.01, "KL"),
    ("mean_kl", float("nan"), "nonfinite"),
    ("mean_kl", -0.1, "KL"),
    ("cheap_fraction", 0, "cheap-use"),
    ("cheap_fraction", 1.1, "cheap-use"),
    ("quality_gate", {"point_pass": False}, "explicit point"),
])
def test_any_seed_failure_stops_timing(timing, field, value, error):
    seeds, runtime, advantage = passing_evidence()
    seeds[43]["heldout_ood"][field] = value
    with pytest.raises(ValueError, match=error):
        timing.require_quality_gate(seeds, runtime, policy_family="portfolio",
                                    portfolio_advantage_ci_low=advantage)


@pytest.mark.parametrize("mutation,error", [
    (lambda seed, runtime, adv: seed.pop(29), "three frozen router seeds"),
    (lambda seed, runtime, adv: runtime.pop("development"), "three development"),
    (lambda seed, runtime, adv: adv.update(heldout_ood=0), "resolved portfolio"),
    (lambda seed, runtime, adv: runtime["heldout_matched"].update(batch_size=16), "genuine batch1"),
    (lambda seed, runtime, adv: runtime["heldout_matched"].update(source="sliced_batch16"), "genuine fullmodel"),
])
def test_missing_or_leaky_runtime_evidence_is_rejected(timing, mutation, error):
    seeds, runtime, advantage = passing_evidence()
    mutation(seeds, runtime, advantage)
    with pytest.raises(ValueError, match=error):
        timing.require_quality_gate(seeds, runtime, policy_family="portfolio",
                                    portfolio_advantage_ci_low=advantage)


class TinySwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4, 8, bias=False)
        self.up_proj = nn.Linear(4, 8, bias=False)
        self.down_proj = nn.Linear(8, 4, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SignDecision(nn.Module):
    def __init__(self, candidate=8):
        super().__init__()
        self.candidate = candidate

    def forward(self, x):
        return torch.where(x[:, 0] > 0, self.candidate, 28).long()


def test_packed_dispatch_keeps_original_weights_and_only_accepts_x(timing):
    torch.manual_seed(17)
    dense = TinySwiGLU().to(torch.bfloat16).eval().requires_grad_(False)
    original = {key: value.clone() for key, value in dense.state_dict().items()}
    routed = timing.RoutedPackedMLP(dense, random_partition(8, seed=29), SignDecision())
    assert list(inspect.signature(routed.forward).parameters) == ["x"]
    assert len(routed.candidates) == 28
    x = torch.randn(5, 1, 4, dtype=torch.bfloat16)
    x[:, :, 0] = torch.tensor([1, -1, 1, -1, 1], dtype=torch.bfloat16)[:, None]
    actual = routed(x)
    expected = torch.stack([
        (routed.candidates["8"] if index % 2 == 0 else dense)(x[index])
        for index in range(len(x))
    ])
    # CPU BF16 row-vs-cohort GEMM is numerically equivalent for this tiny test.
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(routed(x[:1]), routed.candidates["8"](x[:1]), atol=0, rtol=0)
    assert routed.routes(x).tolist() == [8, 28, 8, 28, 8]
    assert all(torch.equal(value, original[key]) for key, value in dense.state_dict().items())
    assert all(not parameter.requires_grad for parameter in routed.dense.parameters())
    assert all(not value.requires_grad for module in routed.candidates.values() for value in module.buffers())


def test_binary_packs_only_its_single_fixed_candidate(timing):
    dense = TinySwiGLU()
    labels = random_partition(8, seed=29)
    binary = timing.RoutedPackedMLP(dense, labels, SignDecision(), fixed_candidate=8)
    portfolio = timing.RoutedPackedMLP(dense, labels, SignDecision())
    assert list(binary.candidates) == ["8"]
    binary_bytes = binary.weight_accounting()
    portfolio_bytes = portfolio.weight_accounting()
    assert binary_bytes["candidate_banks"] == 1
    assert portfolio_bytes["candidate_banks"] == 28
    assert portfolio_bytes["duplicated_candidate_weight_bytes"] == 28 * binary_bytes["duplicated_candidate_weight_bytes"]
    assert binary_bytes["duplicated_candidate_weight_bytes"] == 0.75 * binary_bytes["original_dense_weight_bytes"]
    assert binary_bytes["physical_dram_bytes"] is None
    assert portfolio_bytes["physical_dram_bytes"] is None
    x = torch.ones(1, 1, 4)
    torch.testing.assert_close(binary(x), portfolio(x), atol=0, rtol=0)
    binary.decision = SignDecision(candidate=9)
    with pytest.raises(ValueError, match="out-of-range"):
        binary(x)


def test_incorrect_decision_shape_and_dtype_are_rejected(timing):
    routed = timing.RoutedPackedMLP(TinySwiGLU(), random_partition(8, seed=17), nn.Identity())
    with pytest.raises(ValueError, match="one int64 route"):
        routed(torch.randn(1, 1, 4))
    with pytest.raises(ValueError, match="input width"):
        routed(torch.randn(1, 1, 3))
    with pytest.raises(ValueError, match="partition and dense width"):
        timing.RoutedPackedMLP(TinySwiGLU(), torch.arange(16), SignDecision())


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

    def forward(self, input_ids, **_kwargs):
        return SimpleNamespace(logits=self.lm_head(self.model.norm(self.model.layers[-1](input_ids))))


def test_cached_dispatch_validates_actual_fullmodel_and_restores_state(timing):
    torch.manual_seed(43)
    model = TinyModel().to(torch.bfloat16).eval().requires_grad_(False)
    dense = model.model.layers[-1].mlp
    routed = timing.RoutedPackedMLP(dense, random_partition(8, seed=29), SignDecision())
    result = timing.validate_fullmodel_replay(
        model, dense, routed, {"input_ids": torch.randn(3, 1, 4, dtype=torch.bfloat16)},
    )
    assert result["exact_cached_vs_fullmodel_logits"] is True
    assert result["runtime_shape"] == [3, 1, 4]
    assert result["states"] == 3
    assert model.model.layers[-1].mlp is dense
    assert not dense._forward_pre_hooks
    assert not model.model.layers[-1].post_attention_layernorm._forward_pre_hooks


def test_replacement_restores_original_module_after_failure(timing):
    layer = TinyLayer()
    original = layer.mlp
    with pytest.raises(RuntimeError, match="injected"), timing.replace_mlp(layer, nn.Identity()):
        raise RuntimeError("injected")
    assert layer.mlp is original


def test_deterministic_replay_selection_does_not_look_at_quality_or_route(timing):
    records = [{"id": context, "position": position, "route": route, "quality": 1000 - position}
               for context, route in (("b", 28), ("a", 1)) for position in (3, 1)]
    selected = timing.sequence_selection(records, 3)
    assert [(row["id"], row["position"]) for row in selected] == [("a", 1), ("b", 1), ("a", 3)]
    for row in records:
        row["route"], row["quality"] = 0, -999
    assert [(row["id"], row["position"]) for row in timing.sequence_selection(records, 3)] == [("a", 1), ("b", 1), ("a", 3)]
    with pytest.raises(ValueError, match="nonempty"):
        timing.sequence_selection([], 4)


def test_speedup_interval_requires_paired_positive_timing_samples(timing):
    result = timing.paired_speedup([2, 4, 6], [1, 2, 3], resamples=100)
    assert result["ratio_of_means"] == pytest.approx(2)
    assert result["paired_block_bootstrap_ci95"] == pytest.approx([2, 2])
    assert result["resolved_savings"] is True
    for left, right in (([1], [1]), ([1, 2], [1]), ([0, 1], [1, 1]), ([1, np.nan], [1, 1])):
        with pytest.raises(ValueError):
            timing.paired_speedup(left, right, resamples=100)


def test_gate_failure_precedes_cuda_access_and_output_creation(timing, monkeypatch, tmp_path):
    seeds, runtime, advantage = passing_evidence()
    seeds[17]["development"]["cheap_fraction"] = 0
    monkeypatch.setattr(timing.torch.cuda, "is_available", lambda: pytest.fail("CUDA checked before gate"))
    target = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="cheap-use"):
        timing.run_isolated_timing(None, None, [], [], target, seed_metrics=seeds,
                                   runtime_metrics=runtime, policy_family="portfolio", budget=None,
                                   config={}, portfolio_advantage_ci_low=advantage)
    assert not target.exists()


def test_no_profiler_means_unknown_traffic_with_exact_detection_reason(timing, monkeypatch, tmp_path):
    monkeypatch.setattr(timing.shutil, "which", lambda _name: None)
    result = timing.attempt_physical_counters(["python", "frozen_policy.py"], tmp_path)
    assert result["status"] == "unavailable"
    assert result["physical_dram_bytes"] is None
    assert "returned None" in result["reason"]
    assert json.loads((tmp_path / "counter_status.json").read_text()) == result


def test_profiler_failure_preserves_hardware_error_without_inventing_bytes(timing, monkeypatch, tmp_path):
    monkeypatch.setattr(timing.shutil, "which", lambda _name: "/usr/bin/ncu")
    commands = []

    def fail(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=1, stdout="", stderr="ERR_NVGPUCTRPERM exact test")

    monkeypatch.setattr(timing.subprocess, "run", fail)
    result = timing.attempt_physical_counters(["python", "frozen_policy.py"], tmp_path)
    assert result["status"] == "failed"
    assert result["physical_dram_bytes"] is None
    assert result["exact_failure"] == "ERR_NVGPUCTRPERM exact test"
    assert "dram__bytes_read.sum,dram__bytes_write.sum" in commands[0]
    assert (tmp_path / "ncu.log").read_text() == result["exact_failure"]


def test_profile_evidence_rechecks_seal_quality_and_runtime_cache_hash(timing, tmp_path):
    seeds, runtime, advantage = passing_evidence()
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    partition = tmp_path / "frozen_partition.json"
    partition.write_text("[0,1,2,3,4,5,6,7]", encoding="utf-8")
    cache = runtime_dir / "cache-heldout_matched.pt"
    cache.write_bytes(b"CPU fixture hash target, never torch-loaded")

    def write(name, value):
        (tmp_path / name).write_text(json.dumps(value), encoding="utf-8")

    write("preregistration.json", {"source_sha256": {}, "config": {"test": True}})
    write("heldout_seal.json", {"frozen_partition.json": hashlib.sha256(partition.read_bytes()).hexdigest()})
    write("recognition_decision.json", {"portfolio": {"eligible_for_runtime_quality": True}})
    write("recognition_metrics.json", {"portfolio": seeds})
    write("runtime/runtime_quality.json", {"portfolio": {"quality": runtime}})
    write("paired_portfolio_minus_binary.json", {
        split: {"cheap_fraction": {"lower": lower}} for split, lower in advantage.items()})
    write("runtime/capture-heldout_matched.json", {"cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest()})
    config, gate = timing.profile_evidence(tmp_path, "portfolio")
    assert config == {"test": True} and gate["passed"] is True
    cache.write_bytes(b"changed")
    with pytest.raises(ValueError, match="runtime cache hash"):
        timing.profile_evidence(tmp_path, "portfolio")
    partition.write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="seal changed"):
        timing.profile_evidence(tmp_path, "portfolio")


def test_static_cache_small_subset_gate_is_additional_and_fail_closed(timing):
    row = {"dense_matches_stored_token": True, "dense_graph_exact": True,
           "routed_reconstruction_exact": True, "greedy_agreement": True, "kl": 0.002}
    numerical = [dict(row) for _ in range(8)]
    result = timing.static_cache_subset_gate(numerical, expected_states=8)
    assert result["point_pass"] is True
    assert result["states"] == 8
    assert result["replaces_full_heldout_runtime_gate"] is False
    numerical[-1]["greedy_agreement"] = False
    assert timing.static_cache_subset_gate(numerical, expected_states=8)["point_pass"] is False
    numerical[-1]["greedy_agreement"] = True
    for item in numerical:
        item["kl"] = 0.01
    assert timing.static_cache_subset_gate(numerical, expected_states=8)["point_pass"] is False
    with pytest.raises(ValueError, match="complete frozen"):
        timing.static_cache_subset_gate(numerical[:-1], expected_states=8)
    numerical[-1]["dense_matches_stored_token"] = False
    with pytest.raises(RuntimeError, match="technical control"):
        timing.static_cache_subset_gate(numerical, expected_states=8)


def test_all_static_cache_prefixes_preflight_before_any_timing(timing, monkeypatch, tmp_path):
    seeds, runtime, advantage = passing_evidence()
    dense = SimpleNamespace(gate_proj=SimpleNamespace(weight=SimpleNamespace(device=torch.device("cuda"))))
    layer = SimpleNamespace(mlp=dense)
    model = SimpleNamespace(model=SimpleNamespace(layers=[layer]))
    routed = SimpleNamespace(weight_accounting=lambda: {"physical_dram_bytes": None})
    records = [{"id": str(index), "position": 1} for index in range(8)]
    seen = []

    def prepare(_model, _dense, _routed, record):
        seen.append(record["id"])
        return None, None, None, {
            "dense_matches_stored_token": True, "dense_graph_exact": True,
            "routed_reconstruction_exact": True, "greedy_agreement": record["id"] != "7",
            "kl": 0.001, "route": 8,
        }

    monkeypatch.setattr(timing.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(timing, "_prepare_timing_prefix", prepare)
    monkeypatch.setattr(timing, "measure_sequence", lambda *_args, **_kwargs: pytest.fail("timing before failed preflight"))
    result = timing.run_cached_decode_timing(
        model, dense, routed, records, tmp_path, seed_metrics=seeds, runtime_metrics=runtime,
        policy_family="portfolio", portfolio_advantage_ci_low=advantage,
        budget=SimpleNamespace(check=lambda: None), config={"fullmodel_states": 8},
    )
    assert seen == list(map(str, range(8)))
    assert result["status"] == "skipped_static_cache_subset_quality_failed"
    assert result["rows"] == []
    assert result["static_cache_subset_gate"]["point_pass"] is False
    assert result["all_prefixes_preflighted_before_timed_calls"] is True
    assert layer.mlp is dense
