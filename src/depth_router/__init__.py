"""DepthRouter research primitives."""

from .causal_memory import (
    PageLayout,
    coalesced_transfer_bytes,
    greedy_affinity_layout,
    page_masses,
    posterior_affinity,
    posterior_entropy,
    predictive_working_set_bytes,
    random_layout,
    requested_pages,
    sequential_layout,
    working_set_pages,
)
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
    "PageLayout",
    "SolverState",
    "StackedTransformerBaseline",
    "SwitchableColdMLP",
    "coalesce_faults",
    "coalesced_transfer_bytes",
    "greedy_affinity_layout",
    "kl_nats",
    "page_masses",
    "posterior_affinity",
    "posterior_entropy",
    "predictive_working_set_bytes",
    "random_layout",
    "requested_pages",
    "sequential_layout",
    "unique_fault_bytes",
    "value_per_byte",
    "working_set_pages",
]
