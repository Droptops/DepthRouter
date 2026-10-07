from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch
from torch import Tensor


class Move(str, Enum):
    """Legal transitions for the bandwidth-constrained solver."""

    SPIN = "spin"
    FAULT = "fault"
    WRITE = "write"
    HALT = "halt"


@dataclass
class SolverState:
    """Explicit solver state s_k = (h_k, c_k, p_k).

    h: residual/scratch state.
    resident_slices: cold-table slices currently treated as hot.
    logp: previous prediction distribution.
    pc: per-token program counter.
    """

    h: Tensor
    resident_slices: set[int] = field(default_factory=set)
    logp: Tensor | None = None
    pc: Tensor | None = None
    kv_written: bool = False
    trace: list[Move] = field(default_factory=list)

    def record(self, move: Move) -> None:
        self.trace.append(move)


def kl_nats(new_logp: Tensor, old_logp: Tensor) -> Tensor:
    """KL(new || old), in nats."""

    if new_logp.shape != old_logp.shape:
        raise ValueError("log-probability tensors must have the same shape")
    p = new_logp.exp()
    return (p * (new_logp - old_logp)).sum(dim=-1)


def nats_per_byte(nats: Tensor | float, nbytes: int) -> Tensor:
    if nbytes <= 0:
        raise ValueError("nbytes must be positive")
    return torch.as_tensor(nats) / float(nbytes)


def coalesce_faults(
    slice_ids: Tensor,
    request_mask: Tensor,
) -> list[tuple[int, Tensor]]:
    """Batch tokens by predicted cold-slice miss.

    A slice is returned once with every requesting token index so an executor can
    fetch it once and serve the whole group.
    """

    if slice_ids.ndim != 1 or request_mask.ndim != 1:
        raise ValueError("slice_ids and request_mask must be 1-D")
    if slice_ids.shape != request_mask.shape:
        raise ValueError("slice_ids and request_mask must have the same shape")

    groups: list[tuple[int, Tensor]] = []
    if not bool(request_mask.any()):
        return groups

    for slice_id in torch.unique(slice_ids[request_mask], sorted=True).tolist():
        idx = torch.nonzero(
            request_mask & (slice_ids == slice_id),
            as_tuple=False,
        ).flatten()
        groups.append((int(slice_id), idx))
    return groups


def schedule_efficiency(
    reference_cross_entropy: float,
    schedule_cross_entropy: float,
    measured_bytes: float,
) -> float:
    """Nats of cross-entropy reduction per measured byte moved."""

    if measured_bytes <= 0:
        raise ValueError("measured_bytes must be positive")
    return (reference_cross_entropy - schedule_cross_entropy) / measured_bytes
