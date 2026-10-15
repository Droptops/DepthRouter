import pytest
import torch
from torch import nn

from depth_router.output_gap import (
    CorrectedSwiGLU,
    ResidualCorrection,
    cheap_candidates,
    contrast_rows,
    objective_loss,
    tensor_sha256,
    train_pair,
)


def test_contrast_projection_matches_score_differences() -> None:
    torch.manual_seed(11)
    head = torch.randn(13, 5)
    gamma = torch.randn(5)
    residual = torch.randn(4, 5)
    ids = cheap_candidates(torch.randn(4, 13), k=8)
    contrast = contrast_rows(head, gamma, ids)
    projected = torch.einsum("nkd,nd->nk", contrast, residual)
    scores = residual @ (head * gamma).T
    selected = scores.gather(1, ids)
    torch.testing.assert_close(projected, selected[:, 1:] - selected[:, :1])
    assert contrast.shape == (4, 7, 5)
    assert contrast.dtype == torch.float32


def test_candidates_anchor_argmax_and_do_not_add_excluded_teacher() -> None:
    cheap = torch.tensor([[3.0, 3.0, 2.0, 1.0, -10.0]])
    ids = cheap_candidates(cheap, k=3)
    assert ids[0, 0].item() == cheap.argmax().item() == 0
    assert set(ids[0].tolist()) == {0, 1, 2}
    assert len(set(ids[0].tolist())) == 3
    # An omitted teacher winner stays outside the candidate set. The helper
    # has no teacher input that could silently improve candidate containment.
    teacher = cheap + torch.tensor([[0.0, 0.0, 0.0, 0.0, 30.0]])
    assert teacher.argmax().item() not in ids[0].tolist()
    with pytest.raises(ValueError, match="finite"):
        cheap_candidates(torch.tensor([[1.0, float("nan")]]), k=2)


def test_losses_use_mean_squared_scale_and_gap_geometry() -> None:
    target = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
    pred = torch.zeros_like(target)
    contrast = torch.tensor([[[1.0, -1.0]], [[1.0, -1.0]]])
    hidden_scale = target.square().mean()
    projected = torch.einsum("nkd,nd->nk", contrast, target)
    gap_scale = projected.square().mean()
    torch.testing.assert_close(
        objective_loss(pred, target, contrast, "hidden", hidden_scale), torch.tensor(1.0)
    )
    torch.testing.assert_close(
        objective_loss(pred, target, contrast, "gap", gap_scale), torch.tensor(1.0)
    )
    # Moving in the contrast nullspace changes hidden MSE but not the gap loss.
    shifted = pred + 5.0
    torch.testing.assert_close(
        objective_loss(shifted, target, contrast, "gap", gap_scale), torch.tensor(1.0)
    )
    with pytest.raises(ValueError, match="scale"):
        objective_loss(pred, target, contrast, "hidden", 0.0)


def test_train_pair_handles_inference_tensors_and_audits_matching_budget() -> None:
    with torch.inference_mode():
        torch.manual_seed(5)
        state = torch.randn(12, 4)
        target = state * torch.tensor([1.0, 2.0, 0.0, 0.5])
        contrast = torch.eye(4).expand(12, -1, -1)
        modules, audit = train_pair(state, target, contrast, rank=2, seed=17, steps=4)
    assert audit["initial_state_equal"]
    assert audit["minibatch_order_equal"]
    assert len(set(audit["initial_state_sha256"].values())) == 1
    assert len(set(audit["minibatch_indices_sha256"].values())) == 1
    assert audit["parameter_count"] == 2 * 4 * 2 + 4
    assert audit["steps"] == 4
    assert audit["batch_size"] == 12
    # Identity contrasts make the two objectives identical, so matching
    # initialization, minibatches and normalization must give identical fits.
    for key, value in modules["hidden"].state_dict().items():
        torch.testing.assert_close(value, modules["gap"].state_dict()[key], rtol=0, atol=0)
    rerun, rerun_audit = train_pair(state, target, contrast, rank=2, seed=17, steps=4)
    assert audit["final_state_sha256"] == rerun_audit["final_state_sha256"]
    assert not rerun["gap"].training


def test_deployed_correction_has_only_sparse_module_and_small_predictor() -> None:
    torch.manual_seed(3)
    correction = ResidualCorrection(4, 2)
    sparse = nn.Linear(4, 4, bias=False).to(torch.bfloat16)
    deployed = CorrectedSwiGLU(sparse, correction)
    state = torch.randn(2, 3, 4, dtype=torch.bfloat16)
    actual = deployed(state)
    expected = (sparse(state).float() + correction(state)).to(torch.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == state.dtype
    assert set(deployed._modules) == {"sparse", "residual"}
    assert not dict(correction.named_buffers())
    assert set(correction.state_dict()) == {"in_proj.weight", "out_proj.weight", "out_proj.bias"}
    float_deployed = CorrectedSwiGLU(nn.Identity(), correction)
    assert float_deployed(state.float()).dtype == torch.float32


def test_tensor_hash_records_bfloat_payload_shape_and_dtype() -> None:
    value = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    assert tensor_sha256(value) == tensor_sha256(value.clone())
    assert tensor_sha256(value) != tensor_sha256(value.float())
    assert tensor_sha256(value) != tensor_sha256(value.reshape(1, 2))


def test_nonfinite_training_data_and_unknown_objective_rejected() -> None:
    state = torch.ones(3, 2)
    target = state.clone()
    target[0, 0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        train_pair(state, target, torch.ones(3, 1, 2), rank=1, seed=1, steps=1)
    with pytest.raises(ValueError, match="objective"):
        objective_loss(state, state, torch.ones(3, 1, 2), "other", 1.0)
