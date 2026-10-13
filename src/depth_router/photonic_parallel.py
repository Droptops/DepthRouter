"""Primitives for the photonic parallel minimal-inference experiment.

The CUDA helpers in this module execute ordinary PyTorch operations.  The
photonic functions are a cost model only; they never report simulated values as
hardware measurements.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def balanced_partitions(width: int, groups: int) -> list[Tensor]:
    """Return disjoint, exhaustive contiguous index groups."""

    if width <= 0:
        raise ValueError("width must be positive")
    if groups <= 0 or groups > width:
        raise ValueError("groups must be in [1, width]")
    return [part for part in torch.tensor_split(torch.arange(width), groups) if part.numel()]


def validate_partition(parts: Sequence[Tensor], width: int) -> None:
    """Raise if *parts* are not an exact partition of ``range(width)``."""

    if not parts:
        raise ValueError("partition is empty")
    flat = torch.cat([part.detach().cpu().long().reshape(-1) for part in parts])
    if flat.numel() != width:
        raise ValueError("partition does not have full coverage")
    if not torch.equal(flat.sort().values, torch.arange(width)):
        raise ValueError("partition must be disjoint and exhaustive")


def swiglu_dense(
    x: Tensor,
    gate: nn.Linear,
    up: nn.Linear,
    down: nn.Linear,
) -> Tensor:
    """Reference Qwen/Llama-style SwiGLU MLP."""

    return down(F.silu(gate(x)) * up(x))


def swiglu_subset(
    x: Tensor,
    gate: nn.Linear,
    up: nn.Linear,
    down: nn.Linear,
    indices: Tensor,
    *,
    include_down_bias: bool = True,
) -> Tensor:
    """Execute only the selected intermediate neurons with packed GEMMs."""

    index = indices.to(device=gate.weight.device, dtype=torch.long)
    gate_bias = None if gate.bias is None else gate.bias.index_select(0, index)
    up_bias = None if up.bias is None else up.bias.index_select(0, index)
    hidden = F.silu(F.linear(x, gate.weight.index_select(0, index), gate_bias))
    hidden = hidden * F.linear(x, up.weight.index_select(0, index), up_bias)
    down_bias = down.bias if include_down_bias else None
    return F.linear(hidden, down.weight.index_select(1, index), down_bias)


def swiglu_grouped_sequential(
    x: Tensor,
    gate: nn.Linear,
    up: nn.Linear,
    down: nn.Linear,
    parts: Sequence[Tensor],
) -> Tensor:
    """Sum independent neuron-group contributions in a fixed order."""

    validate_partition(parts, gate.out_features)
    output: Tensor | None = None
    for group_index, part in enumerate(parts):
        contribution = swiglu_subset(
            x,
            gate,
            up,
            down,
            part,
            include_down_bias=group_index == 0,
        )
        output = contribution if output is None else output + contribution
    if output is None:  # pragma: no cover - validate_partition already prevents this
        raise RuntimeError("no groups were executed")
    return output


def contribution_scores(activation: Tensor, down_weight: Tensor) -> Tensor:
    """Per-neuron contribution-norm proxy for each leading activation row."""

    return activation.detach().float().abs() * down_weight.detach().float().norm(dim=0)


def top_fraction_indices(score: Tensor, fraction: float) -> Tensor:
    """Select a deterministic top fraction from a one-dimensional score."""

    if score.ndim != 1:
        raise ValueError("score must be one-dimensional")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    k = max(1, min(score.numel(), round(score.numel() * fraction)))
    return torch.topk(score, k=k, sorted=True).indices.sort().values


def union_fraction(selections: Sequence[Tensor], width: int) -> float:
    """Fraction of unique neurons retained across branches."""

    if width <= 0:
        raise ValueError("width must be positive")
    if not selections:
        return 0.0
    union = torch.unique(torch.cat([selection.detach().cpu().long() for selection in selections]))
    return float(union.numel() / width)


@dataclass(frozen=True)
class PhotonicAssumptions:
    """Explicit, independently sweepable optical hardware assumptions."""

    feedback_ns: float = 20.0
    iterations: int = 9
    lanes: int = 4
    branches: int = 4
    matrix_programming_ns: float = 25_000.0
    residency_lifetime_inferences: int = 256
    input_conversion_ns: float = 250.0
    readout_ns: float = 6_400.0
    cooldown_ns: float = 2_500.0
    aggregation_ns: float = 300.0
    controller_ns: float = 250.0
    readout_overlaps_feedback: bool = False
    cooldown_overlaps_next_programming: bool = True
    optical_power_w: float = 8.0
    controller_power_w: float = 4.0
    lane_area_mm2: float = 2.0

    def validate(self) -> None:
        if self.feedback_ns <= 0 or self.iterations <= 0:
            raise ValueError("feedback time and iterations must be positive")
        if self.lanes <= 0 or self.branches <= 0:
            raise ValueError("lanes and branches must be positive")
        if self.residency_lifetime_inferences <= 0:
            raise ValueError("residency lifetime must be positive")
        nonnegative = (
            self.matrix_programming_ns,
            self.input_conversion_ns,
            self.readout_ns,
            self.cooldown_ns,
            self.aggregation_ns,
            self.controller_ns,
            self.optical_power_w,
            self.controller_power_w,
            self.lane_area_mm2,
        )
        if any(value < 0 for value in nonnegative):
            raise ValueError("latency, power, and area assumptions cannot be negative")


def photonic_latency_model(assumptions: PhotonicAssumptions) -> dict[str, Any]:
    """Return latency/occupancy/energy for a hypothetical photonic operator.

    Programming is amortized across the declared residency lifetime.  Feedback
    and readout overlap only when explicitly requested.  Cooldown is device
    occupancy; for multiple waves it may overlap the next wave's programming.
    """

    assumptions.validate()
    waves = math.ceil(assumptions.branches / assumptions.lanes)
    feedback_loop_ns = assumptions.feedback_ns * assumptions.iterations
    programming_ns = (
        assumptions.matrix_programming_ns / assumptions.residency_lifetime_inferences
    )
    if assumptions.readout_overlaps_feedback:
        compute_readout_ns = max(feedback_loop_ns, assumptions.readout_ns)
    else:
        compute_readout_ns = feedback_loop_ns + assumptions.readout_ns

    first_wave_ns = programming_ns + assumptions.input_conversion_ns + compute_readout_ns
    if assumptions.cooldown_overlaps_next_programming:
        later_wave_prefix_ns = max(
            assumptions.cooldown_ns,
            programming_ns + assumptions.input_conversion_ns,
        )
    else:
        later_wave_prefix_ns = (
            assumptions.cooldown_ns + programming_ns + assumptions.input_conversion_ns
        )
    elapsed_ns = (
        assumptions.controller_ns
        + assumptions.aggregation_ns
        + first_wave_ns
        + max(0, waves - 1) * (later_wave_prefix_ns + compute_readout_ns)
    )
    occupancy_ns = waves * (
        programming_ns
        + assumptions.input_conversion_ns
        + feedback_loop_ns
        + assumptions.readout_ns
        + assumptions.cooldown_ns
    )
    optical_energy_j = assumptions.optical_power_w * occupancy_ns * 1e-9
    controller_energy_j = assumptions.controller_power_w * elapsed_ns * 1e-9
    return {
        "assumptions": asdict(assumptions),
        "waves": waves,
        "feedback_loop_ns": feedback_loop_ns,
        "amortized_programming_ns": programming_ns,
        "elapsed_latency_ns": elapsed_ns,
        "device_occupancy_ns": occupancy_ns,
        "throughput_bound_per_second": 1e9 / max(occupancy_ns / waves, 1e-12) * waves,
        "energy_j_per_inference": optical_energy_j + controller_energy_j,
        "area_mm2": assumptions.lanes * assumptions.lane_area_mm2,
        "lane_utilization": assumptions.branches / (waves * assumptions.lanes),
        "claim_boundary": "Simulated optical cost; no photonic hardware was measured.",
    }


def break_even_margin(
    gpu_latency_ns: float,
    assumptions: PhotonicAssumptions,
) -> dict[str, Any]:
    """Compare modeled optical elapsed latency with a measured GPU latency."""

    if gpu_latency_ns <= 0:
        raise ValueError("gpu latency must be positive")
    modeled = photonic_latency_model(assumptions)
    optical = float(modeled["elapsed_latency_ns"])
    return {
        **modeled,
        "measured_gpu_latency_ns": gpu_latency_ns,
        "modeled_speedup_vs_gpu": gpu_latency_ns / optical,
        "modeled_break_even": optical < gpu_latency_ns,
        "margin_ns": gpu_latency_ns - optical,
    }
