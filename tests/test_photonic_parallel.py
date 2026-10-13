from dataclasses import replace

import pytest
import torch
from torch import nn

from depth_router.photonic_parallel import (
    PhotonicAssumptions,
    balanced_partitions,
    break_even_margin,
    photonic_latency_model,
    swiglu_dense,
    swiglu_grouped_sequential,
    union_fraction,
    validate_partition,
)


@pytest.mark.parametrize("groups", [2, 4, 8, 16])
def test_grouped_swiglu_reconstructs_dense(groups: int) -> None:
    generator = torch.Generator().manual_seed(17)
    gate = nn.Linear(12, 32, bias=False, dtype=torch.float64)
    up = nn.Linear(12, 32, bias=False, dtype=torch.float64)
    down = nn.Linear(32, 12, bias=False, dtype=torch.float64)
    for module in (gate, up, down):
        module.weight.data.normal_(generator=generator)
    x = torch.randn(3, 5, 12, generator=generator, dtype=torch.float64)

    dense = swiglu_dense(x, gate, up, down)
    grouped = swiglu_grouped_sequential(x, gate, up, down, balanced_partitions(32, groups))

    torch.testing.assert_close(grouped, dense, rtol=1e-12, atol=1e-12)


def test_partition_rejects_overlap() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        validate_partition([torch.tensor([0, 1]), torch.tensor([1, 2])], width=4)


def test_union_fraction_is_not_fraction_per_branch() -> None:
    selections = [torch.tensor([0, 1, 2]), torch.tensor([2, 3])]
    assert union_fraction(selections, width=8) == 0.5


def test_photonic_overlap_and_lane_count_reduce_elapsed_time() -> None:
    base = PhotonicAssumptions(branches=8, lanes=2)
    slow = photonic_latency_model(base)
    overlap = photonic_latency_model(replace(base, readout_overlaps_feedback=True))
    more_lanes = photonic_latency_model(replace(base, lanes=8))
    assert overlap["elapsed_latency_ns"] < slow["elapsed_latency_ns"]
    assert more_lanes["elapsed_latency_ns"] < slow["elapsed_latency_ns"]
    assert slow["device_occupancy_ns"] > slow["elapsed_latency_ns"]


def test_break_even_is_labeled_modeled() -> None:
    result = break_even_margin(20_000.0, PhotonicAssumptions())
    assert result["claim_boundary"].startswith("Simulated")
    assert result["modeled_break_even"] == (
        result["elapsed_latency_ns"] < result["measured_gpu_latency_ns"]
    )
