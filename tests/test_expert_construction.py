"""Independent CPU numerical and protocol checks for the offline Qwen audit."""

import importlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from depth_router.expert_partitions import random_partition


@pytest.fixture
def experiment(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments"))
    return importlib.import_module("expert_construction")


class TinyRMSNorm(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))

    def forward(self, state):
        normalized = state.float() * torch.rsqrt(state.float().square().mean(-1, True) + 1e-6)
        return self.weight * normalized.to(state.dtype)


class TinySwiGLU(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.gate_proj = nn.Linear(4, width, bias=False)
        self.up_proj = nn.Linear(4, width, bias=False)
        self.down_proj = nn.Linear(width, 4, bias=False)

    def forward(self, state):
        return self.down_proj(F.silu(self.gate_proj(state)) * self.up_proj(state))


class TinyLayer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.post_attention_layernorm = TinyRMSNorm(4)
        self.mlp = TinySwiGLU(width)

    def forward(self, state):
        return state + self.mlp(self.post_attention_layernorm(state))


class TinyModel(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TinyLayer(width)])
        self.model.norm = TinyRMSNorm(4)
        self.lm_head = nn.Linear(4, 19, bias=False)

    def forward(self, input_ids, logits_to_keep=1, **_kwargs):
        output = self.model.layers[-1](input_ids)
        logits = self.lm_head(self.model.norm(output))
        return SimpleNamespace(logits=logits[:, -logits_to_keep:])


def test_packed_runtime_consumes_only_input_and_keeps_gate_up_down_alignment(experiment):
    torch.manual_seed(31)
    dense = TinySwiGLU(8).to(torch.bfloat16)
    state = torch.randn(3, 1, 4, dtype=torch.bfloat16)
    ids = torch.tensor([0, 2, 3, 5, 6, 7])
    packed = experiment.PackedMLP(dense, ids)
    expected = F.linear(
        F.silu(F.linear(state, dense.gate_proj.weight[ids]))
        * F.linear(state, dense.up_proj.weight[ids]),
        dense.down_proj.weight[:, ids],
    )
    torch.testing.assert_close(packed(state), expected, atol=0, rtol=0)
    assert list(inspect.signature(packed.forward).parameters) == ["x"]
    assert set(dict(packed.named_buffers())) == {"gate", "up", "down"}
    assert not list(packed.parameters())
    assert all(not value.requires_grad for value in packed.buffers())
    full = experiment.PackedMLP(dense, torch.arange(8))
    torch.testing.assert_close(full(state), dense(state), atol=0, rtol=0)


def test_capture_preserves_actual_residual_rmsnorm_and_full_head(experiment):
    torch.manual_seed(41)
    model = TinyModel().to(torch.bfloat16).eval().requires_grad_(False)
    dense = model.model.layers[-1].mlp
    state = torch.randn(3, 1, 4, dtype=torch.bfloat16)
    result, captured = experiment.capture_decode(model, dense, {"input_ids": state})
    assert torch.equal(captured["base"], state)
    expected = model.lm_head(model.model.norm(captured["base"] + captured["output"])).float()
    assert expected.shape == (3, 1, 19)
    assert torch.equal(result.logits[:, -1].float(), expected[:, -1])
    assert torch.equal(
        experiment.final_logits(model, captured["base"], captured["output"]), expected[:, -1]
    )
    for module in (dense, dense.down_proj, model.model.layers[-1].post_attention_layernorm):
        assert not module._forward_hooks and not module._forward_pre_hooks


def test_all_candidates_match_complete_model_and_model_is_restored(experiment):
    torch.manual_seed(43)
    model = TinyModel().to(torch.bfloat16).eval().requires_grad_(False)
    dense = model.model.layers[-1].mlp
    before = {name: value.clone() for name, value in model.state_dict().items()}
    replay = {"input_ids": torch.randn(2, 1, 4, dtype=torch.bfloat16)}
    records = experiment.validate_cached_candidates(
        model, dense, random_partition(8, seed=17), replay, experiment.Budget(60)
    )
    assert len(records) == 29
    assert all(record["exact_logits"] and record["max_abs_logit_error"] == 0 for record in records)
    assert {len(record["experts"]) for record in records} == {6, 8}
    assert model.model.layers[-1].mlp is dense
    for name, value in model.state_dict().items():
        assert torch.equal(before[name], value)


def test_failed_complete_model_validation_restores_mlp_and_hooks(experiment, monkeypatch):
    model = TinyModel().to(torch.bfloat16).eval()
    dense = model.model.layers[-1].mlp
    original_forward = model.forward

    def fail_only_replacement(**kwargs):
        if model.model.layers[-1].mlp is not dense:
            raise RuntimeError("injected complete model failure")
        return original_forward(**kwargs)

    monkeypatch.setattr(model, "forward", fail_only_replacement)
    with pytest.raises(RuntimeError, match="injected complete"):
        experiment.validate_cached_candidates(
            model,
            dense,
            random_partition(8, seed=17),
            {"input_ids": torch.randn(2, 1, 4, dtype=torch.bfloat16)},
            experiment.Budget(60),
        )
    assert model.model.layers[-1].mlp is dense
    assert not dense._forward_hooks and not dense._forward_pre_hooks
    assert not dense.down_proj._forward_pre_hooks


@pytest.mark.parametrize("objective", ["hidden", "output"])
def test_assignment_loss_has_exact_packed_forward_with_finite_surrogate_gradients(
    experiment, objective
):
    torch.manual_seed(53)
    model = TinyModel(width=4864).to(torch.bfloat16).eval().requires_grad_(False)
    dense = model.model.layers[-1].mlp
    state = torch.randn(3, 1, 4, dtype=torch.bfloat16)
    _, captured = experiment.capture_decode(model, dense, {"input_ids": state})
    captured["valid"] = torch.tensor([True, False, True])
    callback, count, activations = experiment.construction_callback(
        model, dense, [captured], experiment.Budget(60), lambda *_args, **_kwargs: None
    )
    assert count == 2
    assert torch.equal(activations, captured["activation"][[0, 2], 0])
    mask = torch.zeros(4864)
    mask[torch.randperm(4864)[:3648]] = 1
    masks = mask[None].clone().requires_grad_()
    losses = callback(objective, torch.tensor([1, 0]), masks)
    x = captured["x"][[2, 0]]
    base = captured["base"][[2, 0]]
    teacher_output = captured["output"][[2, 0]]
    actual = experiment.PackedMLP(dense, mask.nonzero().flatten())(x)
    if objective == "hidden":
        expected = (actual.float() - teacher_output.float()).square().mean((-1, -2))
        expected /= teacher_output.float().square().mean((-1, -2)).clamp_min(1e-8)
    else:
        logp = F.log_softmax(experiment.final_logits(model, base, teacher_output), dim=-1)
        logq = F.log_softmax(experiment.final_logits(model, base, actual), dim=-1)
        assert logp.shape[-1] == model.lm_head.out_features
        torch.testing.assert_close(logp.exp().sum(-1), torch.ones(2))
        torch.testing.assert_close(logq.exp().sum(-1), torch.ones(2))
        expected = (logp.exp() * (logp - logq)).sum(-1)
    torch.testing.assert_close(losses[0], expected, atol=0, rtol=0)
    losses.sum().backward()
    assert masks.grad is not None and bool(torch.isfinite(masks.grad).all())
    assert masks.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.parameters())


def test_budget_stops_at_frozen_boundary_and_reports_elapsed(experiment, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(experiment.time, "monotonic", lambda: clock[0])
    budget = experiment.Budget(20.0)
    clock[0] = 119.999
    budget.check()
    assert budget.elapsed() == pytest.approx(19.999)
    clock[0] = 120.0
    with pytest.raises(TimeoutError, match="ceiling"):
        budget.check()


def test_preheldout_seal_detects_changed_partition_and_fixed_choice(experiment, tmp_path):
    partition = tmp_path / "partition-A-17.json"
    fixed = tmp_path / "fixed_selection.json"
    partition.write_text("[0,1,2,3,4,5,6,7]", encoding="utf-8")
    fixed.write_text('{"A-17": {"index": 0}}', encoding="utf-8")
    seal = {path.name: experiment.sha_file(path) for path in (partition, fixed)}
    experiment.verify_seal(seal, tmp_path)
    partition.write_text("[1,0,2,3,4,5,6,7]", encoding="utf-8")
    with pytest.raises(RuntimeError, match="partition-A-17"):
        experiment.verify_seal(seal, tmp_path)
    partition.write_text("[0,1,2,3,4,5,6,7]", encoding="utf-8")
    fixed.write_text('{"A-17": {"index": 27}}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="fixed_selection"):
        experiment.verify_seal(seal, tmp_path)
    with pytest.raises(RuntimeError, match="nonempty"):
        experiment.verify_seal({}, tmp_path)


def test_interrupted_matrix_retains_only_complete_rows_and_candidate_schema(experiment, tmp_path):
    torch.manual_seed(59)
    model = TinyModel().to(torch.bfloat16).eval().requires_grad_(False)
    dense = model.model.layers[-1].mlp
    groups, records = [], []
    for group in range(2):
        state = torch.randn(2, 1, 4, dtype=torch.bfloat16)
        result, capture = experiment.capture_decode(model, dense, {"input_ids": state})
        capture["valid"] = torch.ones(2, dtype=torch.bool)
        groups.append(capture)
        top = result.logits[:, -1].float().topk(2, dim=-1).values
        records.extend({"id": f"context-{group}-{slot}", "position": 1,
                        "dense_token": int(result.logits[slot, -1].argmax()),
                        "dense_margin": float(top[slot, 0] - top[slot, 1])}
                       for slot in range(2))

    class LimitedBudget:
        calls = 0

        def check(self):
            self.calls += 1
            # Group1: one group check +30 candidates. Interrupt inside group2.
            if self.calls == 34:
                raise TimeoutError("injected matrix budget exhaustion")

    path = tmp_path / "matrix-heldout_matched-A-17.npz"
    with pytest.raises(TimeoutError, match="matrix budget"):
        experiment.evaluate_partition(
            model, dense, random_partition(8, seed=17), groups, records, path,
            LimitedBudget(), lambda *_args, **_kwargs: None,
        )
    assert not path.exists()
    with np.load(tmp_path / "matrix-heldout_matched-A-17.incomplete.npz") as partial:
        assert not partial["complete"].item()
        assert partial["states_completed"].item() == 2
        assert partial["states_requested"].item() == 4
        assert partial["kl"].shape == (2, 28)
        assert np.isfinite(partial["kl"]).all()
        assert partial["selected_experts"].shape == (28, 6)
        assert partial["context_ids"].tolist() == ["context-0-0", "context-0-1"]
        assert np.array_equal(partial["control_token"], partial["dense_token"])
        assert partial["error_type"].item() == "TimeoutError"


def test_preregistration_preserves_exact_sources_before_any_model_call(
    experiment, monkeypatch, tmp_path
):
    root = tmp_path / "repository"
    files = {
        "data/expert_audit/evaluation_manifest.jsonl": '{"prompt":"fresh text"}\n',
        "data/expert_audit/data_audit.json": "{}",
        "data/expert_audit/source_provenance.json": "{}",
        "data/expert_audit/historical_sources.json": "[]",
        "data/expert_audit/historical_prompts.jsonl": '{"prompt":"historical text"}\n',
        "experiments/expert_construction.py": "# frozen experiment source\n",
        "EXPERT_CONSTRUCTION_PREREGISTRATION.md": "Frozen protocol\n",
        "src/depth_router/expert_partitions.py": "# frozen partition source\n",
        "src/depth_router/output_gap.py": "# frozen shared tensor hash implementation\n",
        "scripts/run_expert_construction.ps1": "# frozen launcher\n",
        "requirements-photonic.txt": "torch>=2.2\n",
        "Dockerfile.photonic": "FROM python:3.11\n",
    }
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    config = {"manifest": "data/expert_audit/evaluation_manifest.jsonl", "assignment": {}}
    config_path = root / "configs/expert_construction.json"
    config_path.parent.mkdir()
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(experiment, "ROOT", root)
    monkeypatch.setattr(experiment, "__file__", str(root / "experiments/expert_construction.py"))
    output = tmp_path / "registered"
    assert experiment.register(config_path, output) == config
    record = json.loads((output / "preregistration.json").read_text())
    assert record["before_model_load_fit_and_heldout"]
    assert record["frozen_assignment_recipe"]["steps"] == 100
    for name, digest in record["source_sha256"].items():
        snapshot = output / "registered_sources" / name
        assert snapshot.read_bytes() == (root / name).read_bytes()
        assert experiment.sha_file(snapshot) == digest
    with pytest.raises(FileExistsError):
        experiment.register(config_path, output)
