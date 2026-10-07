from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Hashable, Iterable

import torch
from torch import Tensor, nn


class Move(str, Enum):
    SPIN = "spin"
    FAULT = "fault"
    WRITE = "write"
    HALT = "halt"


@dataclass(frozen=True)
class SolverState:
    """Bandwidth-constrained recurrent state s_k = (h_k, c_k, p_k)."""

    hidden: Tensor
    resident: frozenset[Hashable] = field(default_factory=frozenset)
    log_probs: Tensor | None = None
    kv_written: bool = False
    spins: int = 0
    faults: int = 0
    bytes_moved: int = 0

    def after_write(self, write_bytes: int = 0) -> "SolverState":
        if self.kv_written:
            raise RuntimeError("WRITE is legal only once")
        return replace(self, kv_written=True, bytes_moved=self.bytes_moved + int(write_bytes))

    def after_fault(self, slice_id: Hashable, fault_bytes: int = 0) -> "SolverState":
        return replace(
            self,
            resident=self.resident | frozenset((slice_id,)),
            faults=self.faults + 1,
            bytes_moved=self.bytes_moved + int(fault_bytes),
        )

    def after_spin(self, hidden: Tensor, log_probs: Tensor | None, spin_bytes: int = 0) -> "SolverState":
        return replace(
            self,
            hidden=hidden,
            log_probs=log_probs,
            spins=self.spins + 1,
            bytes_moved=self.bytes_moved + int(spin_bytes),
        )


def kl_nats(new_log_probs: Tensor, old_log_probs: Tensor) -> Tensor:
    """KL(new || old) in nats."""

    if new_log_probs.shape != old_log_probs.shape:
        raise ValueError("log-probability tensors must have the same shape")
    p = new_log_probs.float().exp()
    return (p * (new_log_probs.float() - old_log_probs.float())).sum(dim=-1)


def value_per_byte(nats: Tensor | float, nbytes: int) -> Tensor:
    return torch.as_tensor(nats, dtype=torch.float32) / max(float(nbytes), 1.0)


def coalesce_faults(slice_ids: Tensor, active: Tensor | None = None) -> dict[int, Tensor]:
    """Group tokens by predicted cold-slice fault so each slice is fetched once."""

    if slice_ids.ndim != 1:
        raise ValueError("slice_ids must be flat [tokens]")
    if active is None:
        active = torch.ones_like(slice_ids, dtype=torch.bool)
    if active.shape != slice_ids.shape:
        raise ValueError("active must match slice_ids")

    plan: dict[int, Tensor] = {}
    for sid in torch.unique(slice_ids[active]).tolist():
        plan[int(sid)] = torch.nonzero(active & (slice_ids == int(sid)), as_tuple=False).flatten()
    return plan


def unique_fault_bytes(requested_slices: Iterable[Hashable], slice_bytes: dict[Hashable, int]) -> int:
    return sum(int(slice_bytes[s]) for s in set(requested_slices))


class SwitchableColdMLP(nn.Module):
    """Treat an existing MLP as one cold slice and pack faulting tokens into one call."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base
        self.fault_mask: Tensor | None = None
        self.calls = 0
        self.tokens_faulted = 0

    def set_fault_mask(self, mask: Tensor | None) -> None:
        self.fault_mask = mask

    def reset_stats(self) -> None:
        self.calls = 0
        self.tokens_faulted = 0

    def weight_bytes(self) -> int:
        return sum(p.numel() * p.element_size() for p in self.base.parameters())

    def forward(self, x: Tensor) -> Tensor:
        if self.fault_mask is None:
            self.calls += 1
            self.tokens_faulted += x.numel() // x.shape[-1]
            return self.base(x)

        if self.fault_mask.shape != x.shape[:-1]:
            raise ValueError("fault mask must match token dimensions")

        flat_mask = self.fault_mask.reshape(-1)
        if not bool(flat_mask.any()):
            return torch.zeros_like(x)

        flat = x.reshape(-1, x.shape[-1])
        idx = torch.nonzero(flat_mask, as_tuple=False).flatten()
        y = self.base(flat.index_select(0, idx))
        out = torch.zeros_like(flat)
        out.index_copy_(0, idx, y)
        self.calls += 1
        self.tokens_faulted += int(idx.numel())
        return out.reshape_as(x)


class KLFaultPolicy:
    """Minimal pre-training policy for the falsification experiment."""

    def __init__(self, threshold: float) -> None:
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        self.threshold = float(threshold)

    def fault_mask(self, pass_idx: int, previous_kl: Tensor | float | None, *, like: Tensor | None = None) -> Tensor:
        if pass_idx == 0 or previous_kl is None:
            return torch.tensor(True) if like is None else torch.ones_like(like, dtype=torch.bool)
        return torch.as_tensor(previous_kl) > self.threshold
