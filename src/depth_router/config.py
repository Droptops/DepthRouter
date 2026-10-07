from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DepthRouterConfig:
    """Configuration shared by the recurrent and stacked baselines."""

    vocab_size: int
    max_seq_len: int
    num_classes: int
    d_model: int = 64
    nhead: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.0
    max_steps: int = 4
    adapter_rank: int = 8

    def __post_init__(self) -> None:
        if self.d_model % self.nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.adapter_rank < 1:
            raise ValueError("adapter_rank must be >= 1")
        if self.max_seq_len < 1:
            raise ValueError("max_seq_len must be >= 1")
