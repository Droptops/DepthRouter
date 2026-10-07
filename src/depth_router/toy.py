from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class PointerChaseSpec:
    nodes: int = 8
    max_hops: int = 4

    @property
    def vocab_size(self) -> int:
        return 2 * self.nodes + self.max_hops

    @property
    def seq_len(self) -> int:
        return self.nodes + 2


def pointer_chase_batch(
    batch_size: int,
    spec: PointerChaseSpec,
    *,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Generate random functional graphs and K-hop lookup targets.

    Layout:
      [ptr(0), ptr(1), ..., ptr(N-1), START(s), HOPS(k)]

    Pointer source identity is represented by sequence position. Pointer token
    value is the destination node.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if spec.nodes < 2:
        raise ValueError("nodes must be >= 2")
    if spec.max_hops < 1:
        raise ValueError("max_hops must be >= 1")

    pointers = torch.randint(
        spec.nodes,
        (batch_size, spec.nodes),
        device=device,
        generator=generator,
    )
    starts = torch.randint(
        spec.nodes,
        (batch_size,),
        device=device,
        generator=generator,
    )
    hops = torch.randint(
        1,
        spec.max_hops + 1,
        (batch_size,),
        device=device,
        generator=generator,
    )

    current = starts.clone()
    row = torch.arange(batch_size, device=device)
    for step in range(spec.max_hops):
        advanced = pointers[row, current]
        current = torch.where(hops > step, advanced, current)

    start_tokens = starts + spec.nodes
    hop_tokens = (hops - 1) + 2 * spec.nodes
    input_ids = torch.cat(
        [pointers, start_tokens[:, None], hop_tokens[:, None]],
        dim=1,
    )
    return input_ids.long(), current.long(), hops.long()
