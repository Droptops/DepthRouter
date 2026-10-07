"""DepthRouter research primitives."""

from .config import DepthRouterConfig
from .model import DepthRouterModel, ForwardStats, StackedTransformerBaseline
from .solver import (
    KLFaultPolicy,
    Move,
    SolverState,
    SwitchableColdMLP,
    coalesce_faults,
    kl_nats,
    unique_fault_bytes,
    value_per_byte,
)

__all__ = [
    "DepthRouterConfig",
    "DepthRouterModel",
    "ForwardStats",
    "KLFaultPolicy",
    "Move",
    "SolverState",
    "StackedTransformerBaseline",
    "SwitchableColdMLP",
    "coalesce_faults",
    "kl_nats",
    "unique_fault_bytes",
    "value_per_byte",
]
