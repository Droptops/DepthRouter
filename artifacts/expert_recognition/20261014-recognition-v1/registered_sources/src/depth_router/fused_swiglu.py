"""Inference-only packed and partially fused SwiGLU implementations.

The packed module owns physically compact gate/up/down weights.  It therefore
does not perform a weight gather in ``forward``.  The dynamic module is kept
separate because its top-k selection and three gathers are part of its runtime
cost.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:  # Triton is optional for CPU development and unit tests.
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only without a CUDA stack
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _fused_gate_up_kernel(
        x_ptr,
        gate_ptr,
        up_ptr,
        output_ptr,
        rows: tl.constexpr,
        width: tl.constexpr,
        hidden: tl.constexpr,
        stride_xm: tl.constexpr,
        stride_xk: tl.constexpr,
        stride_wn: tl.constexpr,
        stride_wk: tl.constexpr,
        stride_om: tl.constexpr,
        stride_on: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Compute ``silu(x @ gate.T) * (x @ up.T)`` in one kernel."""

        row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        neuron = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for start in range(0, hidden, BLOCK_K):
            inner = start + tl.arange(0, BLOCK_K)
            x = tl.load(
                x_ptr + row[:, None] * stride_xm + inner[None, :] * stride_xk,
                mask=(row[:, None] < rows) & (inner[None, :] < hidden),
                other=0.0,
            )
            weight_offsets = neuron[None, :] * stride_wn + inner[:, None] * stride_wk
            weight_mask = (neuron[None, :] < width) & (inner[:, None] < hidden)
            gate_weight = tl.load(gate_ptr + weight_offsets, mask=weight_mask, other=0.0)
            up_weight = tl.load(up_ptr + weight_offsets, mask=weight_mask, other=0.0)
            gate_acc += tl.dot(x, gate_weight)
            up_acc += tl.dot(x, up_weight)
        activated = gate_acc * tl.sigmoid(gate_acc) * up_acc
        tl.store(
            output_ptr + row[:, None] * stride_om + neuron[None, :] * stride_on,
            activated,
            mask=(row[:, None] < rows) & (neuron[None, :] < width),
        )


def _validate_linear_triplet(gate: nn.Linear, up: nn.Linear, down: nn.Linear) -> None:
    if gate.in_features != up.in_features:
        raise ValueError("gate and up projections must have the same input width")
    if gate.out_features != up.out_features or gate.out_features != down.in_features:
        raise ValueError("gate, up, and down intermediate widths must match")
    if gate.bias is not None or up.bias is not None:
        raise ValueError("the fused path currently requires bias-free gate and up projections")


class PrepackedSwiGLU(nn.Module):
    """SwiGLU with a fixed neuron subset materialized into contiguous weights."""

    def __init__(
        self,
        gate: nn.Linear,
        up: nn.Linear,
        down: nn.Linear,
        indices: Tensor,
    ) -> None:
        super().__init__()
        _validate_linear_triplet(gate, up, down)
        index = indices.detach().to(device=gate.weight.device, dtype=torch.long)
        if index.ndim != 1 or index.numel() == 0:
            raise ValueError("indices must be a non-empty one-dimensional tensor")
        if int(index.min()) < 0 or int(index.max()) >= gate.out_features:
            raise ValueError("indices are outside the intermediate dimension")
        if torch.unique(index).numel() != index.numel():
            raise ValueError("indices must be unique")
        index = index.sort().values
        self.input_width = gate.in_features
        self.intermediate_width = int(index.numel())
        self.output_width = down.out_features
        self.register_buffer("indices", index)
        self.register_buffer(
            "gate_weight", gate.weight.detach().index_select(0, index).contiguous()
        )
        self.register_buffer(
            "up_weight", up.weight.detach().index_select(0, index).contiguous()
        )
        self.register_buffer(
            "down_weight", down.weight.detach().index_select(1, index).contiguous()
        )
        down_bias = None if down.bias is None else down.bias.detach().contiguous()
        self.register_buffer("down_bias", down_bias)

    def forward(self, x: Tensor) -> Tensor:
        hidden = F.silu(F.linear(x, self.gate_weight)) * F.linear(x, self.up_weight)
        return F.linear(hidden, self.down_weight, self.down_bias)

    @property
    def weight_bytes(self) -> int:
        tensors = (self.gate_weight, self.up_weight, self.down_weight)
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


class TritonPrepackedSwiGLU(PrepackedSwiGLU):
    """Prepacked SwiGLU with gate, up, SiLU, and multiply partially fused."""

    def __init__(
        self,
        gate: nn.Linear,
        up: nn.Linear,
        down: nn.Linear,
        indices: Tensor,
        *,
        block_m: int = 16,
        block_n: int = 64,
        block_k: int = 32,
        num_warps: int = 4,
    ) -> None:
        super().__init__(gate, up, down, indices)
        self.block_m = block_m
        self.block_n = block_n
        self.block_k = block_k
        self.num_warps = num_warps

    def forward(self, x: Tensor) -> Tensor:
        if triton is None:
            raise RuntimeError("Triton is not installed")
        if not x.is_cuda:
            raise RuntimeError("the Triton SwiGLU path requires CUDA")
        if x.shape[-1] != self.input_width:
            raise ValueError("input width does not match packed weights")
        original_shape = x.shape[:-1]
        flat = x.reshape(-1, self.input_width).contiguous()
        hidden = torch.empty(
            (flat.shape[0], self.intermediate_width),
            device=flat.device,
            dtype=flat.dtype,
        )
        grid = (
            triton.cdiv(flat.shape[0], self.block_m),
            triton.cdiv(self.intermediate_width, self.block_n),
        )
        _fused_gate_up_kernel[grid](
            flat,
            self.gate_weight,
            self.up_weight,
            hidden,
            flat.shape[0],
            self.intermediate_width,
            self.input_width,
            flat.stride(0),
            flat.stride(1),
            self.gate_weight.stride(0),
            self.gate_weight.stride(1),
            hidden.stride(0),
            hidden.stride(1),
            BLOCK_M=self.block_m,
            BLOCK_N=self.block_n,
            BLOCK_K=self.block_k,
            num_warps=self.num_warps,
        )
        result = F.linear(hidden, self.down_weight, self.down_bias)
        return result.reshape(*original_shape, self.output_width)


class DynamicGatheredSwiGLU(nn.Module):
    """Runtime top-k and weight-gather baseline with all overhead in forward."""

    def __init__(
        self,
        gate: nn.Linear,
        up: nn.Linear,
        down: nn.Linear,
        score: nn.Module,
        retained: int,
    ) -> None:
        super().__init__()
        _validate_linear_triplet(gate, up, down)
        if not 0 < retained <= gate.out_features:
            raise ValueError("retained must be in the intermediate-width range")
        self.gate = gate
        self.up = up
        self.down = down
        self.score = score
        self.retained = retained

    def forward(self, x: Tensor) -> Tensor:
        flat = x.reshape(-1, x.shape[-1])
        # A deployable per-batch route.  The mean deliberately avoids pretending
        # that different masks can share one compact GEMM without regrouping.
        route_state = flat.detach().float().mean(dim=0, keepdim=True)
        index = self.score(route_state).topk(self.retained, dim=-1, sorted=False).indices[0]
        index = index.sort().values
        gate_weight = self.gate.weight.index_select(0, index)
        up_weight = self.up.weight.index_select(0, index)
        down_weight = self.down.weight.index_select(1, index)
        hidden = F.silu(F.linear(x, gate_weight)) * F.linear(x, up_weight)
        return F.linear(hidden, down_weight, self.down.bias)


class CUDAGraphCallable:
    """Capture a fixed-shape inference callable and expose graph replay."""

    def __init__(self, module: nn.Module, example: Tensor) -> None:
        if not example.is_cuda:
            raise ValueError("CUDA graph capture requires a CUDA example")
        self.module = module
        self.input = example.clone()
        side_stream = torch.cuda.Stream(device=example.device)
        side_stream.wait_stream(torch.cuda.current_stream(example.device))
        with torch.cuda.stream(side_stream), torch.inference_mode():
            for _ in range(3):
                module(self.input)
        torch.cuda.current_stream(example.device).wait_stream(side_stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.inference_mode():
            self.output = module(self.input)

    def __call__(self, value: Tensor | None = None) -> Tensor:
        if value is not None:
            self.input.copy_(value)
        self.graph.replay()
        return self.output
