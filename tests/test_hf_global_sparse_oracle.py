"""No-model tests for the global sparse behavioral oracle."""

import importlib.util
from pathlib import Path

import torch
from torch import nn

SCRIPT = (Path(__file__).resolve().parents[1] / "experiments"
          / "hf_global_sparse_oracle.py")
spec = importlib.util.spec_from_file_location("global_sparse_oracle", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_layer_groups_cover_expected_regions():
    groups = module.layer_groups(24)
    assert len(groups["all"]) == 24
    assert groups["early_half"] == list(range(12))
    assert groups["late_half"] == list(range(12, 24))
    assert groups["middle_half"] == list(range(6, 18))
    assert groups["alternating"] == list(range(0, 24, 2))


def test_full_fraction_hook_preserves_down_projection_input():
    down = nn.Linear(4, 2, bias=False)
    activation = torch.tensor([[1.0, -3.0, 2.0, 4.0]])
    captured = []

    def capture(_module, args):
        captured.append(args[0].clone())

    oracle = module.install_oracle_topk_hook(down, fraction=1.0)
    observer = down.register_forward_pre_hook(capture)
    try:
        down(activation)
    finally:
        oracle.remove()
        observer.remove()
    assert torch.equal(captured[0], activation)


def test_invalid_fraction_is_rejected():
    with torch.no_grad():
        down = nn.Linear(2, 1, bias=False)
        try:
            module.install_oracle_topk_hook(down, fraction=0.0)
        except ValueError as error:
            assert "fraction" in str(error)
        else:
            raise AssertionError("invalid fraction was accepted")


def test_distribution_tensors_identify_exact_top1_agreement():
    reference = torch.tensor([[3.0, 1.0], [1.0, 3.0]])
    candidate = torch.tensor([[2.0, 0.0], [4.0, 1.0]])
    kl, agreement, drop = module.distribution_tensors(reference, candidate)
    assert kl.shape == (2,)
    assert agreement.tolist() == [True, False]
    assert drop.shape == (2,)
