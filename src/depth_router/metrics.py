from __future__ import annotations

from collections.abc import Iterable


def incremental_nats_per_byte(
    *,
    ce_after_pass1: float,
    ce_schedule: float,
    bytes_after_pass1: int,
    bytes_schedule: int,
    epsilon: float = 1e-12,
) -> float:
    """Return predictive gain per incremental byte after the first pass.

    Cross-entropy values are expected in nats/token. Positive values mean the
    schedule improved cross-entropy relative to pass 1.
    """

    if bytes_after_pass1 < 0 or bytes_schedule < 0:
        raise ValueError("byte counts must be non-negative")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    delta_bytes = bytes_schedule - bytes_after_pass1
    if delta_bytes <= 0:
        raise ValueError(
            "bytes_schedule must exceed bytes_after_pass1 for incremental efficiency"
        )

    return (ce_after_pass1 - ce_schedule) / max(float(delta_bytes), epsilon)


def accounting_residual(
    *,
    hardware_bytes: int,
    transition_bytes: Iterable[int],
) -> int:
    """Return profiler bytes minus the sum of locally modeled transition bytes."""

    if hardware_bytes < 0:
        raise ValueError("hardware_bytes must be non-negative")

    local = 0
    for value in transition_bytes:
        if value < 0:
            raise ValueError("transition byte counts must be non-negative")
        local += int(value)

    return int(hardware_bytes) - local


def coalesced_fault_ratio(
    *,
    requested_faults: int,
    unique_transfers: int,
) -> float:
    """Return unique cold transfers per requested fault.

    Lower is better. A value of 1 means no coalescing. A value of 1/N means N
    requested faults were served by one transfer on average.
    """

    if requested_faults < 0 or unique_transfers < 0:
        raise ValueError("fault counts must be non-negative")
    if unique_transfers > requested_faults:
        raise ValueError("unique_transfers cannot exceed requested_faults")
    if requested_faults == 0:
        return 0.0

    return unique_transfers / requested_faults
