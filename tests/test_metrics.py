import math

import pytest

from depth_router.metrics import (
    accounting_residual,
    coalesced_fault_ratio,
    incremental_nats_per_byte,
)


def test_incremental_nats_per_byte() -> None:
    value = incremental_nats_per_byte(
        ce_after_pass1=2.0,
        ce_schedule=1.5,
        bytes_after_pass1=100,
        bytes_schedule=200,
    )
    assert math.isclose(value, 0.005)


def test_incremental_nats_per_byte_rejects_nonincremental_bytes() -> None:
    with pytest.raises(ValueError):
        incremental_nats_per_byte(
            ce_after_pass1=2.0,
            ce_schedule=1.5,
            bytes_after_pass1=100,
            bytes_schedule=100,
        )


def test_accounting_residual_detects_unmodeled_traffic() -> None:
    assert accounting_residual(
        hardware_bytes=140,
        transition_bytes=[50, 40, 30],
    ) == 20


def test_coalesced_fault_ratio() -> None:
    assert coalesced_fault_ratio(requested_faults=30, unique_transfers=1) == 1 / 30
    assert coalesced_fault_ratio(requested_faults=0, unique_transfers=0) == 0.0
