"""No-model tests for calibration-only layout locality analysis."""

import importlib.util
from pathlib import Path

import torch

SCRIPT = (Path(__file__).resolve().parents[1] / "experiments"
          / "hf_temporal_locality.py")
spec = importlib.util.spec_from_file_location("hf_temporal_locality", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_split_indices_are_prompt_disjoint():
    calibration, test = module.split_indices(20)
    assert set(calibration.tolist()).isdisjoint(test.tolist())
    assert sorted(calibration.tolist() + test.tolist()) == list(range(20))
    assert all(index % 10 < 7 for index in calibration.tolist())
    assert all(index % 10 >= 7 for index in test.tolist())


def test_page_demand_metrics_only_include_valid_tokens():
    requests = torch.tensor(
        [[[True, False], [True, True]], [[False, False], [True, False]]]
    )
    valid = torch.tensor([[True, False], [False, True]])
    result = module.page_demand_metrics(requests, valid)
    assert result["mean_requested_pages_per_token"] == 1.0
    assert result["mean_requested_page_fraction"] == 0.5


def test_page_layout_changes_contiguous_demand():
    neurons = torch.tensor([[[True, False, True, False]]])
    identity = module.page_request_mask(neurons, torch.arange(4), units_per_page=2)
    grouped = module.page_request_mask(
        neurons, torch.tensor([0, 2, 1, 3]), units_per_page=2
    )
    assert identity.sum().item() == 2
    assert grouped.sum().item() == 1
