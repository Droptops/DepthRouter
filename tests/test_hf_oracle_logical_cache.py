"""No-model tests for the oracle logical-neuron cache harness."""

import importlib.util
from pathlib import Path

import torch
from torch import nn

SCRIPT = (Path(__file__).resolve().parents[1] / "experiments"
          / "hf_oracle_physical_cache_frontier.py")
spec = importlib.util.spec_from_file_location("oracle_cache", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class MinimalMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = nn.Linear(4, 2, bias=False)


def test_full_width_selection_preserves_neuron_order():
    mlp = MinimalMlp()
    activation = torch.tensor([[4.0, 1.0, 3.0, 2.0]])
    selected = module.selected_indices(activation, mlp, 1.0)
    assert selected.tolist() == [[0, 1, 2, 3]]


def test_cache_faults_are_logical_neuron_insertions():
    cache, faults = module.update_cache([1, 2], torch.tensor([2, 3]), budget=2)
    assert faults == 1
    assert cache == [2, 3]
