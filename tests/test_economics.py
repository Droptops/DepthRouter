import math

import pytest

from depth_router.economics import (
    address_information_yield,
    memory_arbitrage_ratio,
    working_set_from_smooth_address_entropy,
)


def test_memory_arbitrage_break_even() -> None:
    ratio = memory_arbitrage_ratio(
        working_set_bytes_before=8_000_000,
        working_set_bytes_after=2_000_000,
        spin_bytes=1_000_000,
    )
    assert ratio == 6.0


def test_memory_arbitrage_rejects_growing_working_set() -> None:
    with pytest.raises(ValueError):
        memory_arbitrage_ratio(
            working_set_bytes_before=1.0,
            working_set_bytes_after=2.0,
            spin_bytes=1.0,
        )


def test_one_nat_retires_working_set_by_e_factor() -> None:
    before = working_set_from_smooth_address_entropy(
        entropy_nats=2.0,
        page_bytes=4096,
    )
    after = working_set_from_smooth_address_entropy(
        entropy_nats=1.0,
        page_bytes=4096,
    )
    assert math.isclose(before / after, math.e, rel_tol=1e-12)


def test_address_information_yield() -> None:
    got = address_information_yield(
        smooth_address_entropy_before_nats=2.5,
        smooth_address_entropy_after_nats=1.5,
        spin_bytes=2_000_000,
    )
    assert math.isclose(got, 5e-7)
