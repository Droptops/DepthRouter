# ruff: noqa: I001
import sys
from pathlib import Path

import torch


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from hf_block_structured_causal_selector import (
    blocks_to_neurons,
    passes_strict_gate,
    selected_block_count,
    straight_through_topk_mask,
)


def test_block_indices_are_contiguous_and_canonical():
    selected = blocks_to_neurons(
        torch.tensor([2, 0]),
        block_size=4,
        width=10,
    )
    assert selected.tolist() == [0, 1, 2, 3, 8, 9]


def test_straight_through_mask_has_exact_hard_budget_and_gradient():
    scores = torch.tensor([[0.1, 0.4, -0.2, 0.3]], requires_grad=True)
    mask, hard, soft = straight_through_topk_mask(
        scores,
        k=2,
        temperature=0.5,
    )
    assert hard.sum().item() == 2
    assert mask.detach().equal(hard)
    assert 0.0 < soft.mean().item() < 1.0
    mask.sum().backward()
    assert scores.grad is not None
    assert scores.grad.abs().sum().item() > 0


def test_strict_gate_enforces_quality_width_and_metadata():
    assert passes_strict_gate(
        top1_agreement=0.99,
        mean_kl_nats=0.009,
        execution_fraction=0.90,
        metadata_fraction=0.01,
    )
    assert not passes_strict_gate(
        top1_agreement=0.989,
        mean_kl_nats=0.009,
        execution_fraction=0.90,
        metadata_fraction=0.01,
    )
    assert not passes_strict_gate(
        top1_agreement=1.0,
        mean_kl_nats=0.01,
        execution_fraction=0.90,
        metadata_fraction=0.01,
    )
    assert not passes_strict_gate(
        top1_agreement=1.0,
        mean_kl_nats=0.0,
        execution_fraction=0.91,
        metadata_fraction=0.0,
    )
    assert not passes_strict_gate(
        top1_agreement=1.0,
        mean_kl_nats=0.0,
        execution_fraction=0.90,
        metadata_fraction=0.10,
    )


def test_block_budget_never_rounds_above_requested_fraction():
    k = selected_block_count(
        width=4864,
        block_size=32,
        blocks=152,
        fraction=0.90,
    )
    assert k == 136
    assert k * 32 / 4864 <= 0.90


def test_claim_boundary_keeps_physical_metrics_unclaimed():
    source = (EXPERIMENTS / "hf_block_structured_causal_selector.py").read_text(
        encoding="utf-8"
    )
    assert 'parser.add_argument("--revision")' in source
    assert '"passes_strict_gate"' in source
    assert "full-vocabulary logits" in source
    assert "physical bytes and latency are null" in source
