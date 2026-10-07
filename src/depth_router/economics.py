from __future__ import annotations

import math


def memory_arbitrage_ratio(
    *,
    working_set_bytes_before: float,
    working_set_bytes_after: float,
    spin_bytes: float,
) -> float:
    """Future bytes retired per byte spent on one resident refinement spin.

    A value above 1 is the first-order break-even condition: the refinement
    moved fewer bytes than the cold working-set bytes it made unnecessary.
    """

    if working_set_bytes_before < 0 or working_set_bytes_after < 0:
        raise ValueError("working-set bytes must be non-negative")
    if working_set_bytes_after > working_set_bytes_before:
        raise ValueError("working set grew; this is not a retirement event")
    if spin_bytes <= 0:
        raise ValueError("spin_bytes must be positive")

    avoided = working_set_bytes_before - working_set_bytes_after
    return avoided / spin_bytes


def address_information_yield(
    *,
    smooth_address_entropy_before_nats: float,
    smooth_address_entropy_after_nats: float,
    spin_bytes: float,
) -> float:
    """Smooth address entropy retired per byte spent, in nats/byte."""

    if spin_bytes <= 0:
        raise ValueError("spin_bytes must be positive")
    return (
        smooth_address_entropy_before_nats
        - smooth_address_entropy_after_nats
    ) / spin_bytes


def working_set_from_smooth_address_entropy(
    *,
    entropy_nats: float,
    page_bytes: float,
) -> float:
    """Equal-page working set implied by smooth Renyi-0 address entropy."""

    if page_bytes <= 0:
        raise ValueError("page_bytes must be positive")
    if entropy_nats < 0:
        raise ValueError("entropy_nats must be non-negative")
    return page_bytes * math.exp(entropy_nats)
