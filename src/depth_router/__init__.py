"""DepthRouter research primitives."""

from .config import DepthRouterConfig
from .model import DepthRouterModel, ForwardStats, StackedTransformerBaseline
from .placement import (
    Move,
    SolverState,
    coalesce_faults,
    kl_nats,
    nats_per_byte,
    schedule_efficiency,
)

__all__ = [
    "DepthRouterConfig",
    "DepthRouterModel",
    "ForwardStats",
    "Move",
    "SolverState",
    "StackedTransformerBaseline",
    "coalesce_faults",
    "kl_nats",
    "nats_per_byte",
    "schedule_efficiency",
]
