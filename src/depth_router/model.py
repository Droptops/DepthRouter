from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .config import DepthRouterConfig


@dataclass
class ForwardStats:
    """Routing telemetry from one forward pass.

    mean_sample_passes is logical work, not a wall-clock compute claim. The
    current prototype still evaluates a dense batch until every sample halts.
    """

    physical_block_evaluations: int
    sample_passes: Tensor
    residual_trace: list[Tensor]

    @property
    def mean_sample_passes(self) -> float:
        return float(self.sample_passes.float().mean().item())


class LowRankResidualAdapter(nn.Module):
    """Small pass-specific residual adapter."""

    def __init__(self, d_model: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Linear(d_model, rank, bias=False)
        self.up = nn.Linear(rank, d_model, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.up(self.down(hidden))


class _TokenBackbone(nn.Module):
    def __init__(self, config: DepthRouterConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.final_norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.num_classes)

    def embed(self, input_ids: Tensor) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("sequence length exceeds max_seq_len")

        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        return self.token_embedding(input_ids) + self.position_embedding(positions)

    def classify(self, hidden: Tensor) -> Tensor:
        return self.head(self.final_norm(hidden[:, -1]))


def _make_block(config: DepthRouterConfig) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        d_model=config.d_model,
        nhead=config.nhead,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )


class DepthRouterModel(_TokenBackbone):
    """One transformer block reused across recurrent depth.

    Adaptive halting is sample-level and residual-based in v0. This is a
    research primitive, not yet a sparse execution kernel.
    """

    def __init__(
        self,
        config: DepthRouterConfig,
        *,
        use_pass_adapters: bool = False,
    ) -> None:
        super().__init__(config)
        self.shared_block = _make_block(config)
        self.use_pass_adapters = use_pass_adapters
        if use_pass_adapters:
            self.pass_adapters = nn.ModuleList(
                [
                    LowRankResidualAdapter(config.d_model, config.adapter_rank)
                    for _ in range(config.max_steps)
                ]
            )
        else:
            self.pass_adapters = nn.ModuleList()

    def forward(
        self,
        input_ids: Tensor,
        *,
        adaptive: bool = False,
        halt_threshold: float = 0.01,
        min_steps: int = 1,
        return_stats: bool = False,
    ) -> Tensor | tuple[Tensor, ForwardStats]:
        if min_steps < 1 or min_steps > self.config.max_steps:
            raise ValueError("min_steps must be between 1 and max_steps")
        if halt_threshold < 0:
            raise ValueError("halt_threshold must be non-negative")

        hidden = self.embed(input_ids)
        batch_size = hidden.shape[0]
        halted = torch.zeros(batch_size, dtype=torch.bool, device=hidden.device)
        sample_passes = torch.zeros(batch_size, dtype=torch.long, device=hidden.device)
        residual_trace: list[Tensor] = []
        physical_block_evaluations = 0

        for step in range(self.config.max_steps):
            active_before = ~halted
            if not bool(active_before.any()):
                break

            candidate = self.shared_block(hidden)
            physical_block_evaluations += 1

            if self.use_pass_adapters:
                candidate = candidate + self.pass_adapters[step](candidate)

            numerator = (candidate - hidden).pow(2).mean(dim=(1, 2)).sqrt()
            denominator = hidden.pow(2).mean(dim=(1, 2)).sqrt().clamp_min(1e-8)
            residual = numerator / denominator
            residual_trace.append(residual.detach())

            hidden = torch.where(active_before[:, None, None], candidate, hidden)
            sample_passes = sample_passes + active_before.long()

            if adaptive and step + 1 >= min_steps:
                halted = halted | (active_before & (residual <= halt_threshold))

        logits = self.classify(hidden)
        if not return_stats:
            return logits

        return logits, ForwardStats(
            physical_block_evaluations=physical_block_evaluations,
            sample_passes=sample_passes,
            residual_trace=residual_trace,
        )


class StackedTransformerBaseline(_TokenBackbone):
    """Conventional fixed-depth baseline with independent block parameters."""

    def __init__(self, config: DepthRouterConfig) -> None:
        super().__init__(config)
        self.blocks = nn.ModuleList([_make_block(config) for _ in range(config.max_steps)])

    def forward(
        self,
        input_ids: Tensor,
        *,
        return_stats: bool = False,
    ) -> Tensor | tuple[Tensor, ForwardStats]:
        hidden = self.embed(input_ids)
        residual_trace: list[Tensor] = []

        for block in self.blocks:
            candidate = block(hidden)
            numerator = (candidate - hidden).pow(2).mean(dim=(1, 2)).sqrt()
            denominator = hidden.pow(2).mean(dim=(1, 2)).sqrt().clamp_min(1e-8)
            residual_trace.append((numerator / denominator).detach())
            hidden = candidate

        logits = self.classify(hidden)
        if not return_stats:
            return logits

        batch_size = input_ids.shape[0]
        sample_passes = torch.full(
            (batch_size,),
            self.config.max_steps,
            dtype=torch.long,
            device=input_ids.device,
        )
        return logits, ForwardStats(
            physical_block_evaluations=self.config.max_steps,
            sample_passes=sample_passes,
            residual_trace=residual_trace,
        )
