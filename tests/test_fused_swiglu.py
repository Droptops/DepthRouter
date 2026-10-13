import pytest
import torch
from torch import nn

from depth_router.fused_sparse_evaluation import (
    SPLIT_SIZES,
    answer_success,
    make_fused_sparse_manifest,
    validate_fused_sparse_manifest,
)
from depth_router.fused_swiglu import DynamicGatheredSwiGLU, PrepackedSwiGLU


def test_full_width_prepacking_matches_dense() -> None:
    torch.manual_seed(4)
    gate = nn.Linear(7, 13, bias=False)
    up = nn.Linear(7, 13, bias=False)
    down = nn.Linear(13, 7, bias=False)
    x = torch.randn(5, 7)
    expected = down(torch.nn.functional.silu(gate(x)) * up(x))
    packed = PrepackedSwiGLU(gate, up, down, torch.arange(13))
    torch.testing.assert_close(packed(x), expected)


def test_prepacking_validates_indices() -> None:
    gate = nn.Linear(4, 8, bias=False)
    up = nn.Linear(4, 8, bias=False)
    down = nn.Linear(8, 4, bias=False)
    with pytest.raises(ValueError, match="unique"):
        PrepackedSwiGLU(gate, up, down, torch.tensor([1, 1]))


def test_dynamic_route_matches_explicit_predict_select_and_gather() -> None:
    torch.manual_seed(9)
    gate = nn.Linear(4, 8, bias=False)
    up = nn.Linear(4, 8, bias=False)
    down = nn.Linear(8, 4, bias=False)
    score = nn.Linear(4, 8, bias=False)
    x = torch.randn(3, 4)
    module = DynamicGatheredSwiGLU(gate, up, down, score, retained=5)

    index = score(x.mean(dim=0, keepdim=True)).topk(5, sorted=False).indices[0]
    index = index.sort().values
    hidden = torch.nn.functional.silu(
        torch.nn.functional.linear(x, gate.weight.index_select(0, index))
    ) * torch.nn.functional.linear(x, up.weight.index_select(0, index))
    expected = torch.nn.functional.linear(hidden, down.weight.index_select(1, index))

    torch.testing.assert_close(module(x), expected)


def test_new_manifest_is_large_balanced_and_historical_isolated() -> None:
    rows = make_fused_sparse_manifest()
    validate_fused_sparse_manifest(rows, historical_prompts=["an unrelated old prompt"])
    assert len(rows) == sum(SPLIT_SIZES.values()) == 1536
    assert sum(row["split"].startswith("heldout") for row in rows) == 1024
    with pytest.raises(ValueError, match="historical"):
        validate_fused_sparse_manifest(rows, historical_prompts=[rows[0]["prompt"]])


def test_task_success_judges_exact_and_constrained_answers() -> None:
    assert answer_success("YES.", {"answer": "YES", "judge": "exact"})
    assert answer_success(
        "Kernel fusion can reduce launch overhead.",
        {"answer": "kernel", "judge": "contains"},
    )
