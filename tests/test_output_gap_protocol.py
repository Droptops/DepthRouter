"""CPU checks of the statistics and read-only protocol used by the Qwen run."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from depth_router.fused_swiglu import PrepackedSwiGLU


@pytest.fixture
def protocol(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments"))
    return importlib.import_module("output_gap_completion")


@pytest.fixture
def config():
    return {
        "seed": 81,
        "seeds": [17, 29, 43],
        "ranks": [8, 16],
        "bootstrap_draws": 500,
        "quality_gate_top1": 0.99,
        "quality_gate_kl": 0.01,
    }


def _row(method, index, agreement=1, split="heldout_matched"):
    return {
        "method": method,
        "id": f"{split}-{index}",
        "split": split,
        "task": "test",
        "top1_agreement": agreement,
        "kl": 0.0,
        "hidden_residual_mse": 0.0,
        "output_gap_mse": 0.0,
        "reference_margin": float(index + 1),
        "dense_winner_outside_cheap8": False,
        "candidate_top_tie_count": 1,
    }


def _paired_rows(config, differences, split="heldout_matched"):
    count = len(next(iter(differences.values())))
    rows = [_row("dense", index, split=split) for index in range(count)]
    for (rank, seed), values in differences.items():
        for index, difference in enumerate(values):
            rows.extend(
                [
                    _row(f"hidden_r{rank}_s{seed}", index, int(difference <= 0), split),
                    _row(f"gap_r{rank}_s{seed}", index, int(difference >= 0), split),
                ]
            )
    return rows


def test_bootstrap_keeps_seeds_paired_within_each_prompt(protocol, config, monkeypatch):
    config["ranks"] = [8]
    differences = {
        (8, 17): [1, 1, 0, -1],
        (8, 29): [-1, 0, 1, 1],
        (8, 43): [0, 1, -1, 1],
    }
    captured = []
    original_ci = protocol.ci

    def traced_ci(values, seed, draws):
        captured.append(np.asarray(values).copy())
        return original_ci(values, seed, draws)

    monkeypatch.setattr(protocol, "ci", traced_ci)
    comparison = protocol.paired_comparisons(_paired_rows(config, differences), config)
    assert len(comparison) == 4
    for index, expected in enumerate(differences.values()):
        np.testing.assert_array_equal(captured[index], expected)
        assert comparison[index]["gap_wins"] == 2
        assert comparison[index]["hidden_wins"] == 1
        assert comparison[index]["gap_minus_hidden_agreement"] == 0.25
    # These are four prompt clusters, not twelve supposedly independent seed rows.
    np.testing.assert_allclose(captured[-1], [0, 2 / 3, 0, 1 / 3])
    aggregate = comparison[-1]
    assert aggregate["seed"] == "paired_seed_mean"
    assert aggregate["gap_minus_hidden_agreement"] == 0.25
    assert aggregate["resampling_unit"] == "prompt, with all seeds kept together"
    assert aggregate["ci95"] == original_ci([0, 2 / 3, 0, 1 / 3], 81, 500)


def test_positive_mean_cannot_override_a_negative_seed(protocol, config):
    differences = {
        (rank, seed): [1 if rank == 8 or seed != 43 else -1] * 8
        for rank in config["ranks"]
        for seed in config["seeds"]
    }
    rows = _paired_rows(config, differences)
    rows += _paired_rows(config, differences, split="heldout_ood")
    aggregate = [
        r for r in protocol.paired_comparisons(rows, config) if r["seed"] == "paired_seed_mean"
    ]
    assert len(aggregate) == 4
    for comparison in aggregate:
        assert comparison["ci95"][0] > 0
        assert comparison["established"] is (comparison["rank"] == 8)
        assert comparison["direction_positive_all_seeds"] is (comparison["rank"] == 8)


def test_primary_comparison_uses_frozen_development_rank(protocol, config):
    differences = {
        (rank, seed): [1 if rank == 8 or seed != 43 else -1] * 8
        for rank in config["ranks"]
        for seed in config["seeds"]
    }
    rows = _paired_rows(config, differences)
    rows += _paired_rows(config, differences, split="heldout_ood")
    comparisons = protocol.paired_comparisons(rows, config)
    development = [
        {"method": "gap_r8_s17", "split": "development", "top1_agreement": 0.9, "mean_kl": 0.001},
        {"method": "gap_r16_s29", "split": "development", "top1_agreement": 0.95, "mean_kl": 0.002},
    ]
    selected = protocol.select_development_arm(development)
    assert selected["method"] == "gap_r16_s29"
    # Rank 8 wins held out, but it cannot rescue the frozen primary rank 16.
    assert protocol.objective_improvement_established(comparisons, 8)
    assert not protocol.objective_improvement_established(comparisons, 16)
    assert not protocol.objective_improvement_established([], 8)
    assert not protocol.objective_improvement_established(
        [r for r in comparisons if r["split"] != "heldout_ood"], 8
    )
    rank8_aggregate = next(
        r for r in comparisons if r["rank"] == 8 and r["seed"] == "paired_seed_mean"
    )
    assert not protocol.objective_improvement_established(comparisons + [rank8_aggregate], 8)


def test_selection_is_development_only_with_preregistered_tiebreaks(protocol):
    def candidate(method, agreement=0.9, kl=0.002, split="development"):
        return {"method": method, "split": split, "top1_agreement": agreement, "mean_kl": kl}

    rows = [
        candidate("gap_r16_s17"),
        candidate("gap_r8_s29"),
        candidate("gap_r8_s17"),
        candidate("hidden_r8_s17", 1.0, 0.0),
        candidate("gap_r8_s43", 1.0, 0.0, "heldout_matched"),
    ]
    assert protocol.select_development_arm(rows)["method"] == "gap_r8_s17"
    # Higher agreement wins even with worse KL; KL wins the next tie before rank.
    rows.append(candidate("gap_r16_s43", 0.95, 0.003))
    assert protocol.select_development_arm(rows)["method"] == "gap_r16_s43"
    rows.append(candidate("gap_r16_s29", 0.95, 0.0025))
    assert protocol.select_development_arm(rows)["method"] == "gap_r16_s29"
    with pytest.raises(ValueError, match="no development"):
        protocol.select_development_arm([candidate("gap_r8_s17", split="heldout_ood")])


def test_summary_reports_reference_margin_quartiles_and_exclusions(protocol, config):
    rows = [_row("dense", index) for index in range(8)]
    pattern = [0, 0, 1, 0, 1, 1, 0, 1]
    for index, agreement in enumerate(pattern):
        row = _row("sparse75", index, agreement)
        row.update(
            {
                "kl": 0.002 * index,
                "hidden_residual_mse": 2.0,
                "output_gap_mse": 3.0,
                "dense_winner_outside_cheap8": index in {0, 2, 4},
                "candidate_top_tie_count": 2 if index == 1 else 1,
            }
        )
        rows.append(row)
    summaries = {r["method"]: r for r in protocol.summarize(rows, config)}
    assert summaries["dense"]["passes_quality_gate"]
    sparse = summaries["sparse75"]
    assert sparse["n"] == 8
    assert sparse["top1_agreement"] == 0.5
    assert sparse["mean_kl"] == pytest.approx(0.007)
    assert not sparse["passes_quality_gate"]
    assert sparse["hidden_residual_mse"] == 2
    assert sparse["output_gap_mse"] == 3
    assert sparse["excluded_winners"] == 3
    assert sparse["excluded_winner_failures"] == 1
    assert sparse["top_ties"] == 1
    assert sparse["margin_quartile_cuts"] == [2.75, 4.5, 6.25]
    assert [q["n"] for q in sparse["margin_quartiles"]] == [2, 2, 2, 2]
    assert [q["top1_agreement"] for q in sparse["margin_quartiles"]] == [0, 0.5, 1, 0.5]


def test_equal_reference_margins_leave_empty_quartiles_explicit(protocol, config):
    rows = [_row("dense", index) for index in range(4)]
    for row in rows:
        row["reference_margin"] = 0.0
    result = protocol.summarize(rows, config)[0]
    assert [q["n"] for q in result["margin_quartiles"]] == [4, 0, 0, 0]
    assert [q["top1_agreement"] for q in result["margin_quartiles"]] == [1, None, None, None]


def test_state_hash_is_read_only_and_detects_parameter_and_buffer_changes(protocol):
    module = nn.Linear(4, 3)
    module.register_buffer("protocol_marker", torch.tensor([7], dtype=torch.int64))
    original = {name: tensor.clone() for name, tensor in module.state_dict().items()}
    digest = protocol.state_hash(module)
    assert protocol.state_hash(module) == digest
    for name, tensor in module.state_dict().items():
        assert torch.equal(tensor, original[name])
    assert module.training
    assert all(parameter.requires_grad for parameter in module.parameters())
    with torch.no_grad():
        module.weight[0, 0] += 1
    assert protocol.state_hash(module) != digest
    module.load_state_dict(original)
    assert protocol.state_hash(module) == digest
    module.protocol_marker.add_(1)
    assert protocol.state_hash(module) != digest


class _TinySwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4, 8, bias=False)
        self.up_proj = nn.Linear(4, 8, bias=False)
        self.down_proj = nn.Linear(8, 4, bias=False)

    def forward(self, value):
        return self.down_proj(F.silu(self.gate_proj(value)) * self.up_proj(value))


class _TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _TinySwiGLU()
        self.post_attention_layernorm = nn.Identity()

    def forward(self, value):
        return value + self.mlp(self.post_attention_layernorm(value))


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_TinyLayer()])
        self.model.norm = nn.LayerNorm(4)
        self.lm_head = nn.Linear(4, 13, bias=False)

    def forward(self, input_ids, **_kwargs):
        hidden = self.model.layers[0](input_ids)
        return SimpleNamespace(logits=self.lm_head(self.model.norm(hidden)))


@pytest.mark.parametrize("tied_head", [False, True])
def test_full_width_evaluation_is_exact_and_restores_modules(protocol, monkeypatch, tied_head):
    torch.manual_seed(2)
    model = _TinyModel().eval()
    if tied_head:
        with torch.no_grad():
            model.lm_head.weight.zero_()
    layer = model.model.layers[0]
    dense = layer.mlp
    full = PrepackedSwiGLU(dense.gate_proj, dense.up_proj, dense.down_proj, torch.arange(8))
    sparse = PrepackedSwiGLU(dense.gate_proj, dense.up_proj, dense.down_proj, torch.arange(6))
    examples = [{"id": f"row-{i}", "task": "test", "split": "development"} for i in range(4)]
    inputs = torch.randn(4, 3, 4)
    monkeypatch.setattr(protocol, "encoded_batches", lambda *_: [(examples, {"input_ids": inputs})])
    initial_hash = protocol.state_hash(model)
    logger = SimpleNamespace(write=lambda *_args, **_kwargs: None)
    rows = protocol.evaluate(
        model,
        None,
        layer,
        {"dense": dense, "full_width_control": full, "sparse75": sparse},
        examples,
        {},
        logger,
    )
    controls = [row for row in rows if row["method"] == "full_width_control"]
    assert len(controls) == 4
    assert all(row["top1_agreement"] == 1 for row in controls)
    assert all(row["max_abs_logit_error"] == 0 for row in controls)
    assert all(row["hidden_residual_mse"] == 0 for row in controls)
    assert all(row["output_gap_mse"] == 0 for row in controls)
    assert all(row["actual_mlp_output_mse"] == 0 for row in controls)
    assert all(row["candidate_top_tie_count"] == (13 if tied_head else 1) for row in controls)
    assert layer.mlp is dense
    assert protocol.state_hash(model) == initial_hash
    assert all(
        not module._forward_hooks and not module._forward_pre_hooks
        for module in (dense, sparse, full, layer.post_attention_layernorm)
    )
