from __future__ import annotations

from dataclasses import dataclass

from torch import nn


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def parameter_bytes(module: nn.Module) -> int:
    return sum(
        parameter.numel() * parameter.element_size()
        for parameter in module.parameters()
    )


@dataclass(frozen=True)
class WeightTrafficEstimate:
    """Analytic parameter-traffic bounds, not profiler measurements."""

    parameter_storage_bytes: int
    naive_stream_bytes: int
    ideal_resident_bytes: int


def estimate_recurrent_weight_traffic(
    shared_block: nn.Module,
    *,
    passes: int,
    adapters: nn.Module | None = None,
) -> WeightTrafficEstimate:
    if passes < 1:
        raise ValueError("passes must be >= 1")

    block_bytes = parameter_bytes(shared_block)
    adapter_bytes = parameter_bytes(adapters) if adapters is not None else 0

    naive = block_bytes * passes + adapter_bytes
    ideal = block_bytes + adapter_bytes

    return WeightTrafficEstimate(
        parameter_storage_bytes=block_bytes + adapter_bytes,
        naive_stream_bytes=naive,
        ideal_resident_bytes=ideal,
    )
