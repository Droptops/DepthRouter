"""Matched hidden-residual and token-gap correction objectives.

Candidate rows and teacher residuals are training data only. Both objectives
produce the same small hidden-output module for the ordinary full-head path.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import torch
from torch import Tensor, nn


class ResidualCorrection(nn.Module):
    """The prior rank-limited correction architecture, with FP32 parameters."""

    def __init__(self, hidden: int, rank: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(hidden, rank, bias=False)
        self.out_proj = nn.Linear(rank, hidden, bias=True)

    def forward(self, state: Tensor) -> Tensor:
        return self.out_proj(torch.tanh(self.in_proj(state.float())))


class CorrectedSwiGLU(nn.Module):
    """Add the correction in FP32 and return the model's activation dtype."""

    def __init__(self, sparse: nn.Module, residual: nn.Module) -> None:
        super().__init__()
        self.sparse = sparse
        self.residual = residual

    def forward(self, state: Tensor) -> Tensor:
        return (self.sparse(state).float() + self.residual(state)).to(state.dtype)


def cheap_candidates(logits: Tensor, k: int = 8) -> Tensor:
    """Anchor at the cheap argmax, then select the remaining top candidates.

    The explicit anchor preserves argmax's first-index tie rule, which topk
    alone does not promise. No teacher logits or labels enter this operation.
    """

    if logits.ndim != 2 or not 2 <= k <= logits.shape[1]:
        raise ValueError("logits must have shape [examples, vocabulary] and 2 <= k <= vocabulary")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("candidate logits must be finite")
    anchor = logits.argmax(dim=-1, keepdim=True)
    remaining = logits.detach().clone().scatter_(1, anchor, -torch.inf)
    others = remaining.topk(k - 1, dim=-1, sorted=True).indices
    return torch.cat((anchor, others), dim=-1)


def contrast_rows(head_weight: Tensor, norm_weight: Tensor, ids: Tensor) -> Tensor:
    """Return FP32 rows (E[k] - E[anchor]) * final_RMSNorm_weight."""

    if head_weight.ndim != 2 or norm_weight.shape != (head_weight.shape[1],):
        raise ValueError("head and norm weights must share their hidden dimension")
    if ids.ndim != 2 or ids.shape[1] < 2:
        raise ValueError("candidate ids must have shape [examples, candidates >= 2]")
    if ids.dtype != torch.long:
        raise ValueError("candidate ids must have dtype torch.long")
    index = ids.to(head_weight.device)
    if index.numel() and (int(index.min()) < 0 or int(index.max()) >= head_weight.shape[0]):
        raise ValueError("candidate ids are outside the head vocabulary")
    rows = head_weight.detach()[index].float()
    return (rows[:, 1:] - rows[:, :1]) * norm_weight.detach().float().to(rows.device)


def objective_loss(
    pred: Tensor,
    target: Tensor,
    contrast: Tensor,
    objective: str,
    scale: Tensor | float,
) -> Tensor:
    """MSE divided by calibration target mean square (not by target RMS)."""

    if objective not in ("hidden", "gap"):
        raise ValueError("objective must be 'hidden' or 'gap'")
    error = pred.float() - target.float()
    if objective == "gap":
        error = torch.einsum("nkd,nd->nk", contrast.float(), error)
    denominator = torch.as_tensor(scale, dtype=torch.float32, device=error.device)
    if denominator.numel() != 1 or not bool(torch.isfinite(denominator)) or denominator <= 0:
        raise ValueError("scale must be one finite positive mean square")
    return error.square().mean() / denominator


def tensor_sha256(tensor: Tensor) -> str:
    """Hash tensor shape, dtype and exact contiguous payload, including BF16."""

    value = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(json.dumps({"shape": list(value.shape), "dtype": str(value.dtype)}).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _module_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor_sha256(value).encode())
    return digest.hexdigest()


def train_pair(
    state: Tensor,
    target: Tensor,
    contrast: Tensor,
    rank: int,
    seed: int,
    steps: int = 300,
    batch_size: int = 64,
    learning_rate: float = 0.003,
) -> tuple[dict[str, ResidualCorrection], dict[str, Any]]:
    """Fit both objectives with exactly paired initialization and minibatches.

    ``target`` must be the difference between actual dense and actual packed
    sparse MLP outputs. This function never reconstructs a sparse operator or
    invokes a teacher model. The caller must provide calibration-only data.
    """

    if state.ndim != 2 or target.shape != state.shape or not state.shape[0]:
        raise ValueError("state and target must have matching nonempty [examples, hidden] shapes")
    if contrast.ndim != 3 or contrast.shape[0] != state.shape[0]:
        raise ValueError("contrast must have shape [examples, contrasts, hidden]")
    if contrast.shape[1] < 1 or contrast.shape[2] != state.shape[1]:
        raise ValueError("contrast must contain at least one row and match the hidden width")
    if rank < 1 or steps < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("rank, steps, batch_size and learning_rate must be positive")

    # Captures are normally inference tensors. Clone with inference mode off:
    # autograd must be able to save the fixed calibration inputs for backward.
    with torch.inference_mode(False), torch.enable_grad():
        x = state.detach().to(dtype=torch.float32).clone()
        residual = target.detach().to(device=x.device, dtype=torch.float32).clone()
        rows = contrast.detach().to(device=x.device, dtype=torch.float32).clone()
        if not all(bool(torch.isfinite(value).all()) for value in (x, residual, rows)):
            raise ValueError("calibration tensors must be finite")
        projected_target = torch.einsum("nkd,nd->nk", rows, residual)
        scales = {
            "hidden": residual.square().mean().clamp_min(1e-16),
            "gap": projected_target.square().mean().clamp_min(1e-16),
        }
        torch.manual_seed(seed)
        base = ResidualCorrection(x.shape[1], rank).to(x.device)
        modules = {name: copy.deepcopy(base) for name in ("hidden", "gap")}
        initial_hashes = {name: _module_sha256(module) for name, module in modules.items()}
        generator = torch.Generator(device="cpu").manual_seed(seed + 1)
        minibatches = torch.stack(
            [
                torch.randint(x.shape[0], (min(batch_size, x.shape[0]),), generator=generator)
                for _ in range(steps)
            ]
        )
        minibatch_hash = tensor_sha256(minibatches)
        batches = minibatches.to(x.device)
        final_losses: dict[str, float] = {}
        for objective, module in modules.items():
            optimizer = torch.optim.AdamW(module.parameters(), lr=learning_rate, weight_decay=1e-4)
            module.train()
            for batch in batches:
                loss = objective_loss(
                    module(x[batch]), residual[batch], rows[batch], objective, scales[objective]
                )
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"nonfinite {objective} training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            module.eval()
            with torch.no_grad():
                final_loss = objective_loss(module(x), residual, rows, objective, scales[objective])
                if not bool(torch.isfinite(final_loss)):
                    raise FloatingPointError(f"nonfinite final {objective} training loss")
                final_losses[objective] = float(final_loss)
        audit: dict[str, Any] = {
            "rank": rank,
            "seed": seed,
            "steps": steps,
            "batch_size": min(batch_size, x.shape[0]),
            "learning_rate": learning_rate,
            "weight_decay": 1e-4,
            "parameter_count": sum(parameter.numel() for parameter in base.parameters()),
            "scales_mean_square": {key: float(value) for key, value in scales.items()},
            "initial_state_sha256": initial_hashes,
            "minibatch_indices_sha256": {key: minibatch_hash for key in modules},
            "initial_state_equal": initial_hashes["hidden"] == initial_hashes["gap"],
            "minibatch_order_equal": True,
            "calibration_state_sha256": tensor_sha256(x),
            "calibration_target_sha256": tensor_sha256(residual),
            "calibration_contrast_sha256": tensor_sha256(rows),
            "final_training_loss": final_losses,
            "final_state_sha256": {key: _module_sha256(value) for key, value in modules.items()},
        }
    return modules, audit
