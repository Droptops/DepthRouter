# ruff: noqa: I001
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import torch
from torch import Tensor, nn

from hf_temporal_tangent_distilled import (
    capture_dense_state,
    load_texts,
    parse_dtype,
    resolve_device,
    resolve_layers,
    sparse_exact_output,
)
from hf_global_lowbit_decode import LowBitAddress


@dataclass
class TrainingStates:
    current: Tensor
    base_hidden: Tensor
    dense_hidden: Tensor
    activation: Tensor
    top_ids: Tensor
    reference_top_logits: Tensor


@dataclass
class EvaluationState:
    input_ids: Tensor
    current: Tensor
    reference_logits: Tensor


class BlockSelectorHead(nn.Module):
    """Causal low-rank address head over contiguous MLP neuron blocks."""

    def __init__(
        self,
        hidden: int,
        blocks: int,
        rank: int,
        *,
        use_prior: bool = False,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(hidden, rank, bias=False)
        self.out_proj = nn.Linear(rank, blocks, bias=True)
        if use_prior:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)
            self.prior_scale = nn.Parameter(torch.ones(()))
        else:
            self.register_parameter("prior_scale", None)

    def forward(self, hidden: Tensor, prior: Tensor | None = None) -> Tensor:
        scores = self.out_proj(torch.tanh(self.in_proj(hidden)))
        if self.prior_scale is not None:
            if prior is None:
                raise ValueError("a prior-enabled selector requires prior scores")
            scores = scores + self.prior_scale * prior
        return scores


def texts_from_json_records(records: object) -> list[str]:
    if not isinstance(records, list):
        raise TypeError("remote text source must contain a JSON list")
    texts: list[str] = []
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("every remote text record must be a JSON object")
        if "text" in record:
            text = str(record["text"]).strip()
        elif "instruction" in record:
            instruction = str(record["instruction"]).strip()
            extra = str(record.get("input", "")).strip()
            text = instruction if not extra else f"{instruction}\n\nInput:\n{extra}"
        else:
            raise ValueError("remote text record needs 'text' or 'instruction'")
        if text:
            texts.append(text)
    if len(texts) < 12:
        raise ValueError("need at least twelve remote texts")
    return texts


def load_texts_from_url(url: str) -> list[str]:
    with urlopen(url, timeout=60) as response:
        records = json.loads(response.read().decode("utf-8"))
    return texts_from_json_records(records)


@torch.inference_mode()
def collect_training_states(
    model: nn.Module,
    layer: nn.Module,
    tokenizer,
    texts: list[str],
    *,
    decode_steps: int,
    max_length: int,
    train_top_logits: int,
    device: torch.device,
) -> TrainingStates:
    current: list[Tensor] = []
    base_hidden: list[Tensor] = []
    dense_hidden: list[Tensor] = []
    activation: list[Tensor] = []
    top_ids: list[Tensor] = []
    reference_top_logits: list[Tensor] = []

    for text in texts:
        prefix = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )["input_ids"].to(device)
        for _ in range(decode_steps):
            reference, hidden, base, dense, active = capture_dense_state(
                model,
                layer,
                prefix,
            )
            ids = torch.topk(
                reference,
                k=min(train_top_logits, reference.shape[-1]),
                dim=-1,
                sorted=True,
            ).indices
            current.append(hidden.detach().clone())
            base_hidden.append(base.detach().clone())
            dense_hidden.append(dense.detach().clone())
            activation.append(active.detach().clone())
            top_ids.append(ids.detach().clone())
            reference_top_logits.append(
                torch.gather(reference, 1, ids).detach().clone()
            )
            token = reference.argmax(dim=-1).to(device)
            prefix = torch.cat([prefix, token[:, None]], dim=1)

    return TrainingStates(
        current=torch.cat(current).detach().clone(),
        base_hidden=torch.cat(base_hidden).detach().clone(),
        dense_hidden=torch.cat(dense_hidden).detach().clone(),
        activation=torch.cat(activation).detach().clone(),
        top_ids=torch.cat(top_ids).detach().clone(),
        reference_top_logits=torch.cat(reference_top_logits).detach().clone(),
    )


@torch.inference_mode()
def collect_evaluation_states(
    model: nn.Module,
    layer: nn.Module,
    tokenizer,
    texts: list[str],
    *,
    decode_steps: int,
    max_length: int,
    device: torch.device,
) -> tuple[list[EvaluationState], list[list[int]]]:
    states: list[EvaluationState] = []
    dense_tokens: list[list[int]] = []

    for text in texts:
        prefix = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )["input_ids"].to(device)
        prompt_tokens: list[int] = []
        for _ in range(decode_steps):
            reference, hidden, _base, _dense, _active = capture_dense_state(
                model,
                layer,
                prefix,
            )
            states.append(
                EvaluationState(
                    input_ids=prefix.detach().cpu().clone(),
                    current=hidden.detach().clone(),
                    reference_logits=reference.detach().clone(),
                )
            )
            token = reference.argmax(dim=-1).to(device)
            prompt_tokens.append(int(token.item()))
            prefix = torch.cat([prefix, token[:, None]], dim=1)
        dense_tokens.append(prompt_tokens)
    return states, dense_tokens


@torch.inference_mode()
def build_block_contributions(
    activation: Tensor,
    mlp: nn.Module,
    *,
    block_size: int,
    device: torch.device,
    chunk_size: int = 32,
) -> tuple[Tensor, Tensor]:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    width = activation.shape[-1]
    hidden = mlp.down_proj.out_features
    blocks = math.ceil(width / block_size)
    padded_width = blocks * block_size
    padding = padded_width - width

    weight = mlp.down_proj.weight.detach().float().T.contiguous()
    if padding:
        weight = torch.nn.functional.pad(weight, (0, 0, 0, padding))
    weight = weight.view(blocks, block_size, hidden)

    rows: list[Tensor] = []
    for start in range(0, activation.shape[0], chunk_size):
        active = activation[start : start + chunk_size].to(device).float()
        if padding:
            active = torch.nn.functional.pad(active, (0, padding))
        active = active.view(active.shape[0], blocks, block_size)
        rows.append(torch.einsum("nbq,bqh->nbh", active, weight).cpu())

    bias = mlp.down_proj.bias
    if bias is None:
        dense_bias = torch.zeros(hidden, dtype=torch.float32)
    else:
        dense_bias = bias.detach().float().cpu().clone()
    return torch.cat(rows).detach().clone(), dense_bias


def straight_through_topk_mask(
    scores: Tensor,
    *,
    k: int,
    temperature: float,
) -> tuple[Tensor, Tensor, Tensor]:
    if not 0 < k <= scores.shape[-1]:
        raise ValueError("k must be in [1, number of blocks]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    hard_index = torch.topk(scores, k=k, dim=-1, sorted=False).indices
    hard = torch.zeros_like(scores)
    hard.scatter_(1, hard_index, 1.0)

    if k == scores.shape[-1]:
        soft = torch.ones_like(scores)
    else:
        centered = scores - scores.mean(dim=-1, keepdim=True)
        with torch.no_grad():
            lower = centered.amin(dim=-1, keepdim=True) - 20.0 * temperature
            upper = centered.amax(dim=-1, keepdim=True) + 20.0 * temperature
            for _ in range(32):
                threshold = (lower + upper) * 0.5
                mass = torch.sigmoid(
                    (centered - threshold) / temperature
                ).sum(dim=-1, keepdim=True)
                lower = torch.where(mass > k, threshold, lower)
                upper = torch.where(mass > k, upper, threshold)
            threshold = ((lower + upper) * 0.5).detach()
        soft = torch.sigmoid((centered - threshold) / temperature)
    mask = hard + soft - soft.detach()
    return mask, hard, soft


def apply_rmsnorm(hidden: Tensor, weight: Tensor, eps: float) -> Tensor:
    variance = hidden.float().square().mean(dim=-1, keepdim=True)
    return hidden.float() * torch.rsqrt(variance + eps) * weight.float()


def norm_epsilon(norm: nn.Module) -> float:
    return float(
        getattr(
            norm,
            "variance_epsilon",
            getattr(norm, "eps", 1e-6),
        )
    )


def top_logit_reconstruction(
    hidden: Tensor,
    top_ids: Tensor,
    norm: nn.Module,
    lm_head: nn.Linear,
) -> tuple[Tensor, Tensor]:
    normalized = apply_rmsnorm(
        hidden,
        norm.weight.detach().to(hidden.device),
        norm_epsilon(norm),
    )
    selected_head = lm_head.weight.detach()[top_ids].float()
    logits = torch.einsum("bd,bmd->bm", normalized, selected_head)
    return normalized, logits


def selector_metadata_fraction(head: nn.Module, mlp: nn.Module) -> float:
    metadata = sum(parameter.numel() for parameter in head.parameters())
    cold = sum(parameter.numel() for parameter in mlp.parameters())
    return metadata / cold


def selected_block_count(
    *,
    width: int,
    block_size: int,
    blocks: int,
    fraction: float,
) -> int:
    """Largest whole-block budget that never exceeds the requested neuron fraction."""
    if width <= 0 or block_size <= 0 or blocks <= 0:
        raise ValueError("width, block_size, and blocks must be positive")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    return max(1, min(blocks, math.floor(width * fraction / block_size)))


def train_selector(
    states: TrainingStates,
    block_contributions: Tensor,
    dense_bias: Tensor,
    mlp: nn.Module,
    norm: nn.Module,
    lm_head: nn.Linear,
    *,
    block_size: int,
    fraction: float,
    rank: int,
    steps: int,
    batch_size: int,
    lr: float,
    temperature: float,
    hidden_loss_weight: float,
    margin_loss_weight: float,
    budget_loss_weight: float,
    seed: int,
    device: torch.device,
    prior_scores: Tensor | None = None,
) -> tuple[BlockSelectorHead, dict[str, float]]:
    blocks = block_contributions.shape[1]
    width = mlp.down_proj.in_features
    k = selected_block_count(
        width=width,
        block_size=block_size,
        blocks=blocks,
        fraction=fraction,
    )
    actual_fraction = min(1.0, (k * block_size) / width)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    head = BlockSelectorHead(
        states.current.shape[-1],
        blocks,
        rank,
        use_prior=prior_scores is not None,
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed + 1)
    final: dict[str, float] = {}

    for _ in range(steps):
        index = torch.randint(
            states.current.shape[0],
            (min(batch_size, states.current.shape[0]),),
            generator=generator,
        )
        current = states.current[index].to(device)
        base = states.base_hidden[index].to(device)
        dense = states.dense_hidden[index].to(device)
        contributions = block_contributions[index].to(device)
        top_ids = states.top_ids[index].to(device)
        reference = states.reference_top_logits[index].to(device)

        prior = None if prior_scores is None else prior_scores[index].to(device)
        scores = head(current, prior)
        mask, _hard, soft = straight_through_topk_mask(
            scores,
            k=k,
            temperature=temperature,
        )
        candidate = base + torch.einsum("nb,nbh->nh", mask, contributions)
        candidate = candidate + dense_bias.to(device)
        dense_normalized, _dense_recomputed = top_logit_reconstruction(
            dense,
            top_ids,
            norm,
            lm_head,
        )
        candidate_normalized, candidate_logits = top_logit_reconstruction(
            candidate,
            top_ids,
            norm,
            lm_head,
        )

        reference_logp = torch.log_softmax(reference, dim=-1)
        candidate_logp = torch.log_softmax(candidate_logits, dim=-1)
        probability = reference_logp.exp()
        kl_loss = (
            probability * (reference_logp - candidate_logp)
        ).sum(dim=-1).mean()
        hidden_scale = dense_normalized.square().mean().clamp_min(1e-8)
        hidden_loss = torch.nn.functional.mse_loss(
            candidate_normalized,
            dense_normalized,
        ) / hidden_scale
        dense_margin = (
            reference[:, 0] - reference[:, 1:].max(dim=-1).values
        ).detach()
        candidate_margin = (
            candidate_logits[:, 0]
            - candidate_logits[:, 1:].max(dim=-1).values
        )
        margin_loss = torch.relu(dense_margin - candidate_margin).mean()
        budget_loss = (soft.mean() - actual_fraction) ** 2
        loss = (
            kl_loss
            + hidden_loss_weight * hidden_loss
            + margin_loss_weight * margin_loss
            + budget_loss_weight * budget_loss
        )
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("non-finite block-selector loss")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final = {
            "loss": float(loss.detach().cpu().item()),
            "top_logit_kl_loss": float(kl_loss.detach().cpu().item()),
            "normalized_hidden_loss": float(hidden_loss.detach().cpu().item()),
            "top1_margin_loss": float(margin_loss.detach().cpu().item()),
            "soft_budget_fraction": float(soft.mean().detach().cpu().item()),
        }
    return head.eval(), final


def blocks_to_neurons(
    block_ids: Tensor,
    *,
    block_size: int,
    width: int,
) -> Tensor:
    offsets = torch.arange(block_size, device=block_ids.device)
    neurons = block_ids[:, None] * block_size + offsets[None, :]
    return neurons.flatten()[neurons.flatten() < width].sort().values


@torch.inference_mode()
def run_sparse_logits(
    model: nn.Module,
    mlp: nn.Module,
    input_ids: Tensor,
    selected_neurons: Tensor,
) -> Tensor:
    def hook(
        _module: nn.Module,
        hook_args: tuple[Tensor, ...],
        output: Tensor,
    ) -> Tensor:
        hidden = hook_args[0][:, -1, :]
        sparse = sparse_exact_output(
            hidden,
            mlp,
            selected_neurons[None, :],
        )
        replaced = output.clone()
        replaced[:, -1, :] = sparse.to(replaced.dtype)
        return replaced

    handle = mlp.register_forward_hook(hook)
    try:
        return model(input_ids=input_ids).logits[:, -1, :].detach().float().cpu()
    finally:
        handle.remove()


def make_trained_selector(
    head: BlockSelectorHead,
    *,
    k: int,
    block_size: int,
    width: int,
    device: torch.device,
    prior_scorer: Callable[[Tensor], Tensor] | None = None,
) -> Callable[[Tensor], Tensor]:
    @torch.inference_mode()
    def select(hidden: Tensor) -> Tensor:
        prior = None if prior_scorer is None else prior_scorer(hidden)
        scores = head(hidden.to(device), prior)
        blocks = torch.topk(scores, k=k, dim=-1, sorted=False).indices[0]
        return blocks_to_neurons(
            blocks,
            block_size=block_size,
            width=width,
        )

    return select


def make_fixed_selector(
    blocks: Tensor,
    *,
    block_size: int,
    width: int,
) -> Callable[[Tensor], Tensor]:
    selected = blocks_to_neurons(
        blocks,
        block_size=block_size,
        width=width,
    )

    def select(_hidden: Tensor) -> Tensor:
        return selected

    return select


def lowbit_metadata_fraction(
    mlp: nn.Module,
    *,
    bits: int,
) -> float:
    address_bytes, cold_bytes = lowbit_metadata_bytes(mlp, bits=bits)
    return address_bytes / cold_bytes


def lowbit_metadata_bytes(
    mlp: nn.Module,
    *,
    bits: int,
) -> tuple[int, int]:
    if bits not in (1, 2, 3, 4):
        raise ValueError("bits must be one of {1, 2, 3, 4}")
    gate = mlp.gate_proj
    up = mlp.up_proj
    down = mlp.down_proj
    cold_parameters = (
        gate.weight.numel()
        + up.weight.numel()
        + down.weight.numel()
    )
    cold_bytes = cold_parameters * 2
    packed_bits = bits * (gate.weight.numel() + up.weight.numel())
    address_bytes = (packed_bits + 7) // 8
    # Per-neuron FP16 gate scale, up scale, and down-column norm.
    address_bytes += down.in_features * 3 * 2
    return address_bytes, cold_bytes


class LowBitBlockScorer:
    """Causal block scores from packed low-bit gate/up projection proxies."""

    def __init__(
        self,
        address: LowBitAddress,
        mlp: nn.Module,
        *,
        block_size: int,
        width: int,
        device: torch.device,
    ) -> None:
        self.gate_hat = address.gate_hat.to(device)
        self.up_hat = address.up_hat.to(device)
        self.down_norm = address.down_norm.to(device)
        self.fn = getattr(mlp, "act_fn", torch.nn.functional.silu)
        self.block_size = block_size
        self.width = width
        self.blocks = math.ceil(width / block_size)
        self.padding = self.blocks * block_size - width
        self.device = device

    @torch.inference_mode()
    def __call__(self, hidden: Tensor) -> Tensor:
        current = hidden.to(self.device).float()
        gate = torch.nn.functional.linear(current, self.gate_hat)
        up = torch.nn.functional.linear(current, self.up_hat)
        neuron_score = (self.fn(gate) * up).abs() * self.down_norm[None, :]
        if self.padding:
            neuron_score = torch.nn.functional.pad(
                neuron_score,
                (0, self.padding),
            )
        block_score = neuron_score.view(
            neuron_score.shape[0],
            self.blocks,
            self.block_size,
        ).sum(dim=-1)
        log_score = block_score.clamp_min(1e-12).log()
        centered = log_score - log_score.mean(dim=-1, keepdim=True)
        return centered / centered.std(dim=-1, keepdim=True).clamp_min(1e-6)


@torch.inference_mode()
def score_hidden_states(
    scorer: Callable[[Tensor], Tensor],
    hidden: Tensor,
    *,
    chunk_size: int = 64,
) -> Tensor:
    rows = [
        scorer(hidden[start : start + chunk_size]).cpu()
        for start in range(0, hidden.shape[0], chunk_size)
    ]
    return torch.cat(rows).detach().clone()


def make_lowbit_block_selector(
    address: LowBitAddress,
    mlp: nn.Module,
    *,
    k: int,
    block_size: int,
    width: int,
    device: torch.device,
) -> Callable[[Tensor], Tensor]:
    scorer = LowBitBlockScorer(
        address,
        mlp,
        block_size=block_size,
        width=width,
        device=device,
    )

    @torch.inference_mode()
    def select(hidden: Tensor) -> Tensor:
        block_score = scorer(hidden)
        chosen = torch.topk(block_score, k=k, dim=-1, sorted=False).indices[0]
        return blocks_to_neurons(
            chosen,
            block_size=block_size,
            width=width,
        )

    return select


@torch.inference_mode()
def evaluate_dense_states(
    model: nn.Module,
    mlp: nn.Module,
    states: list[EvaluationState],
    selector: Callable[[Tensor], Tensor],
    *,
    device: torch.device,
) -> dict[str, float]:
    all_kl: list[Tensor] = []
    all_agreement: list[Tensor] = []
    for state in states:
        selected = selector(state.current)
        candidate = run_sparse_logits(
            model,
            mlp,
            state.input_ids.to(device),
            selected,
        )
        reference_logp = torch.log_softmax(state.reference_logits, dim=-1)
        candidate_logp = torch.log_softmax(candidate, dim=-1)
        probability = reference_logp.exp()
        all_kl.append(
            (probability * (reference_logp - candidate_logp)).sum(dim=-1)
        )
        all_agreement.append(
            state.reference_logits.argmax(dim=-1) == candidate.argmax(dim=-1)
        )

    kl = torch.cat(all_kl)
    agreement = torch.cat(all_agreement).float()
    return {
        "states": int(kl.numel()),
        "mean_kl_nats": float(kl.mean().item()),
        "p95_kl_nats": float(torch.quantile(kl, 0.95).item()),
        "max_kl_nats": float(kl.max().item()),
        "top1_agreement": float(agreement.mean().item()),
    }


@torch.inference_mode()
def evaluate_free_running(
    model: nn.Module,
    layer: nn.Module,
    tokenizer,
    texts: list[str],
    dense_tokens: list[list[int]],
    selector: Callable[[Tensor], Tensor],
    *,
    max_length: int,
    device: torch.device,
) -> float:
    exact = 0
    mlp = layer.mlp
    for text, reference_tokens in zip(texts, dense_tokens, strict=True):
        prefix = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )["input_ids"].to(device)
        prompt_exact = True
        for reference_token in reference_tokens:
            _reference, hidden, _base, _dense, _active = capture_dense_state(
                model,
                layer,
                prefix,
            )
            selected = selector(hidden)
            candidate = run_sparse_logits(model, mlp, prefix, selected)
            token = candidate.argmax(dim=-1).to(device)
            prompt_exact = prompt_exact and int(token.item()) == reference_token
            prefix = torch.cat([prefix, token[:, None]], dim=1)
        exact += int(prompt_exact)
    return exact / max(len(texts), 1)


def passes_strict_gate(
    *,
    top1_agreement: float,
    mean_kl_nats: float,
    execution_fraction: float,
    metadata_fraction: float,
) -> bool:
    return (
        top1_agreement >= 0.99
        and mean_kl_nats < 0.01
        and execution_fraction <= 0.90
        and execution_fraction + metadata_fraction < 1.0
    )


@torch.inference_mode()
def identity_diagnostics(
    states: TrainingStates,
    block_contributions: Tensor,
    dense_bias: Tensor,
    norm: nn.Module,
    lm_head: nn.Linear,
    *,
    device: torch.device,
) -> dict[str, float]:
    candidate = (
        states.base_hidden.to(device)
        + block_contributions.sum(dim=1).to(device)
        + dense_bias.to(device)
    )
    dense = states.dense_hidden.to(device)
    hidden_error = candidate - dense
    _normalized, logits = top_logit_reconstruction(
        candidate,
        states.top_ids.to(device),
        norm,
        lm_head,
    )
    logit_error = logits.cpu() - states.reference_top_logits
    return {
        "full_block_hidden_max_abs_error": float(hidden_error.abs().max().cpu().item()),
        "full_block_hidden_rmse": float(hidden_error.square().mean().sqrt().cpu().item()),
        "full_block_top_logit_max_abs_error": float(logit_error.abs().max().item()),
        "full_block_top_logit_rmse": float(logit_error.square().mean().sqrt().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--revision")
    parser.add_argument("--text-file")
    parser.add_argument("--text-url")
    parser.add_argument(
        "--deduplicate-texts",
        action="store_true",
        help="Keep the first occurrence of each prompt before the train/test split.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        help="Deterministically shuffle prompts before taking disjoint train/test slices.",
    )
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--train-decode-steps", type=int, default=8)
    parser.add_argument("--test-decode-steps", type=int, default=10)
    parser.add_argument("--max-train-prompts", type=int, default=20)
    parser.add_argument("--max-test-prompts", type=int, default=20)
    parser.add_argument("--train-top-logits", type=int, default=128)
    parser.add_argument("--block-sizes", nargs="+", type=int, default=[64, 128])
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.80, 0.90])
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--hidden-loss-weight", type=float, default=0.25)
    parser.add_argument("--margin-loss-weight", type=float, default=0.1)
    parser.add_argument("--budget-loss-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1301)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument("--skip-controls", action="store_true")
    parser.add_argument(
        "--lowbit-controls",
        nargs="*",
        type=int,
        default=[],
        help="Evaluate causal block scores from packed low-bit gate/up proxies.",
    )
    parser.add_argument(
        "--train-lowbit-prior-bits",
        type=int,
        choices=[1, 2, 3, 4],
        help="Train the low-rank selector as a residual over causal low-bit scores.",
    )
    parser.add_argument("--skip-free-running", action="store_true")
    parser.add_argument("--json-out")
    args = parser.parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit('Install with: pip install -e ".[hf]"') from exc

    if any(not 0.0 < fraction <= 0.90 for fraction in args.fractions):
        raise ValueError("every execution fraction must be in (0, 0.90]")
    if any(block_size <= 0 for block_size in args.block_sizes):
        raise ValueError("block sizes must be positive")
    if any(bits not in (1, 2, 3, 4) for bits in args.lowbit_controls):
        raise ValueError("low-bit controls must use 1, 2, 3, or 4 bits")

    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype)
    if device.type == "cpu" and dtype == torch.float16:
        dtype = torch.float32
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.text_file and args.text_url:
        raise ValueError("choose only one of --text-file and --text-url")
    texts = (
        load_texts_from_url(args.text_url)
        if args.text_url
        else load_texts(args.text_file)
    )
    if args.deduplicate_texts:
        texts = list(dict.fromkeys(texts))
    if args.split_seed is not None:
        split_generator = torch.Generator().manual_seed(args.split_seed)
        order = torch.randperm(len(texts), generator=split_generator).tolist()
        texts = [texts[index] for index in order]
    if args.max_train_prompts + args.max_test_prompts > len(texts):
        raise ValueError(
            "requested train/test prompts exceed the available disjoint texts: "
            f"{args.max_train_prompts} + {args.max_test_prompts} > {len(texts)}"
        )
    train_texts = texts[: args.max_train_prompts]
    test_texts = texts[-args.max_test_prompts :]
    if set(train_texts) & set(test_texts):
        raise ValueError("train and test prompts must be disjoint")

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=dtype,
    ).to(device)
    model.eval().requires_grad_(False)
    layers = resolve_layers(model)
    layer = layers[-1]
    mlp = layer.mlp
    norm = model.model.norm
    lm_head = model.lm_head
    width = mlp.down_proj.in_features
    requested_lowbit = set(args.lowbit_controls)
    if args.train_lowbit_prior_bits is not None:
        requested_lowbit.add(args.train_lowbit_prior_bits)
    lowbit_addresses = {
        bits: LowBitAddress(mlp, bits)
        for bits in sorted(requested_lowbit)
    }

    training = collect_training_states(
        model,
        layer,
        tokenizer,
        train_texts,
        decode_steps=args.train_decode_steps,
        max_length=args.max_length,
        train_top_logits=args.train_top_logits,
        device=device,
    )
    evaluation, dense_tokens = collect_evaluation_states(
        model,
        layer,
        tokenizer,
        test_texts,
        decode_steps=args.test_decode_steps,
        max_length=args.max_length,
        device=device,
    )

    rows = []
    diagnostics = []
    for block_size in args.block_sizes:
        contributions, dense_bias = build_block_contributions(
            training.activation,
            mlp,
            block_size=block_size,
            device=device,
        )
        blocks = contributions.shape[1]
        prior_scorer = None
        training_prior = None
        if args.train_lowbit_prior_bits is not None:
            prior_scorer = LowBitBlockScorer(
                lowbit_addresses[args.train_lowbit_prior_bits],
                mlp,
                block_size=block_size,
                width=width,
                device=device,
            )
            training_prior = score_hidden_states(
                prior_scorer,
                training.current,
            )
        block_energy = contributions.square().mean(dim=(0, 2))
        identity = identity_diagnostics(
            training,
            contributions,
            dense_bias,
            norm,
            lm_head,
            device=device,
        )
        diagnostics.append(
            {
                "block_size": block_size,
                "block_count": blocks,
                **identity,
            }
        )

        for fraction in args.fractions:
            k = selected_block_count(
                width=width,
                block_size=block_size,
                blocks=blocks,
                fraction=fraction,
            )
            actual_fraction = min(1.0, (k * block_size) / width)
            head, train_metrics = train_selector(
                training,
                contributions,
                dense_bias,
                mlp,
                norm,
                lm_head,
                block_size=block_size,
                fraction=fraction,
                rank=args.rank,
                steps=args.steps,
                batch_size=args.batch_size,
                lr=args.lr,
                temperature=args.temperature,
                hidden_loss_weight=args.hidden_loss_weight,
                margin_loss_weight=args.margin_loss_weight,
                budget_loss_weight=args.budget_loss_weight,
                seed=args.seed + block_size + round(fraction * 1000),
                device=device,
                prior_scores=training_prior,
            )
            selector = make_trained_selector(
                head,
                k=k,
                block_size=block_size,
                width=width,
                device=device,
                prior_scorer=prior_scorer,
            )
            metrics = evaluate_dense_states(
                model,
                mlp,
                evaluation,
                selector,
                device=device,
            )
            metadata = selector_metadata_fraction(head, mlp)
            if args.train_lowbit_prior_bits is not None:
                metadata += lowbit_metadata_fraction(
                    mlp,
                    bits=args.train_lowbit_prior_bits,
                )
            row = {
                "policy": (
                    "causal_trained"
                    if args.train_lowbit_prior_bits is None
                    else f"causal_trained_lowbit_{args.train_lowbit_prior_bits}bit"
                ),
                "block_size": block_size,
                "block_count": blocks,
                "selected_blocks": k,
                "requested_fraction": fraction,
                "execution_fraction": actual_fraction,
                "selector_rank": args.rank,
                "resident_selector_fraction": metadata,
                "metadata_plus_execution_fraction": actual_fraction + metadata,
                "train_final": train_metrics,
                **metrics,
            }
            row["passes_strict_gate"] = passes_strict_gate(
                top1_agreement=row["top1_agreement"],
                mean_kl_nats=row["mean_kl_nats"],
                execution_fraction=actual_fraction,
                metadata_fraction=metadata,
            )
            if not args.skip_free_running:
                row["free_running_exact_prompt_fraction"] = evaluate_free_running(
                    model,
                    layer,
                    tokenizer,
                    test_texts,
                    dense_tokens,
                    selector,
                    max_length=args.max_length,
                    device=device,
                )
            else:
                row["free_running_exact_prompt_fraction"] = None
            rows.append(row)

            for bits, address in lowbit_addresses.items():
                lowbit_selector = make_lowbit_block_selector(
                    address,
                    mlp,
                    k=k,
                    block_size=block_size,
                    width=width,
                    device=device,
                )
                lowbit_metrics = evaluate_dense_states(
                    model,
                    mlp,
                    evaluation,
                    lowbit_selector,
                    device=device,
                )
                lowbit_metadata = lowbit_metadata_fraction(mlp, bits=bits)
                lowbit_row = {
                    "policy": f"causal_lowbit_{bits}bit",
                    "block_size": block_size,
                    "block_count": blocks,
                    "selected_blocks": k,
                    "requested_fraction": fraction,
                    "execution_fraction": actual_fraction,
                    "selector_rank": None,
                    "resident_selector_fraction": lowbit_metadata,
                    "metadata_plus_execution_fraction": (
                        actual_fraction + lowbit_metadata
                    ),
                    "train_final": None,
                    **lowbit_metrics,
                }
                lowbit_row["passes_strict_gate"] = passes_strict_gate(
                    top1_agreement=lowbit_row["top1_agreement"],
                    mean_kl_nats=lowbit_row["mean_kl_nats"],
                    execution_fraction=actual_fraction,
                    metadata_fraction=lowbit_metadata,
                )
                if not args.skip_free_running:
                    lowbit_row["free_running_exact_prompt_fraction"] = (
                        evaluate_free_running(
                            model,
                            layer,
                            tokenizer,
                            test_texts,
                            dense_tokens,
                            lowbit_selector,
                            max_length=args.max_length,
                            device=device,
                        )
                    )
                else:
                    lowbit_row["free_running_exact_prompt_fraction"] = None
                rows.append(lowbit_row)

            if not args.skip_controls:
                static_blocks = torch.topk(
                    block_energy,
                    k=k,
                    sorted=False,
                ).indices
                random_generator = torch.Generator().manual_seed(
                    args.seed + block_size + k
                )
                random_blocks = torch.randperm(
                    blocks,
                    generator=random_generator,
                )[:k]
                for policy, chosen in [
                    ("static_train_energy", static_blocks),
                    ("random_fixed", random_blocks),
                ]:
                    control = evaluate_dense_states(
                        model,
                        mlp,
                        evaluation,
                        make_fixed_selector(
                            chosen,
                            block_size=block_size,
                            width=width,
                        ),
                        device=device,
                    )
                    rows.append(
                        {
                            "policy": policy,
                            "block_size": block_size,
                            "block_count": blocks,
                            "selected_blocks": k,
                            "requested_fraction": fraction,
                            "execution_fraction": actual_fraction,
                            "selector_rank": None,
                            "resident_selector_fraction": 0.0,
                            "metadata_plus_execution_fraction": actual_fraction,
                            "train_final": None,
                            "free_running_exact_prompt_fraction": None,
                            "passes_strict_gate": passes_strict_gate(
                                top1_agreement=control["top1_agreement"],
                                mean_kl_nats=control["mean_kl_nats"],
                                execution_fraction=actual_fraction,
                                metadata_fraction=0.0,
                            ),
                            **control,
                        }
                    )

    payload = {
        "experiment": "hf_block_structured_causal_selector_v0",
        "model": args.model,
        "model_revision": getattr(model.config, "_commit_hash", None)
        or args.revision,
        "tokenizer_revision": tokenizer.init_kwargs.get("_commit_hash")
        or args.revision,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "deduplicated_texts": args.deduplicate_texts,
        "text_source": args.text_url or args.text_file or "built-in defaults",
        "device": str(device),
        "dtype": str(dtype),
        "layer": len(layers) - 1,
        "train_prompts": len(train_texts),
        "test_prompts": len(test_texts),
        "train_states": int(training.current.shape[0]),
        "test_states": len(evaluation),
        "train_decode_steps": args.train_decode_steps,
        "test_decode_steps": args.test_decode_steps,
        "train_top_logits": args.train_top_logits,
        "selector_rank": args.rank,
        "training_steps": args.steps,
        "quality_gate": {
            "minimum_top1_agreement": 0.99,
            "maximum_mean_kl_nats_exclusive": 0.01,
            "maximum_execution_fraction": 0.90,
            "metadata_plus_execution_must_be_below_dense": True,
        },
        "modeled_scope": "final-layer MLP parameter fractions only",
        "physical_bytes_per_token": None,
        "latency_ms_per_token": None,
        "identity_diagnostics": diagnostics,
        "rows": rows,
        "claim_boundary": (
            "The causal selector sees only the incoming final-layer MLP state and "
            "chooses contiguous neuron blocks before gate/up/down execution. Training "
            "uses dense block contributions but optimizes exact normalized-hidden "
            "reconstruction plus top-logit KL and margin losses, not oracle-set labels. "
            "Optional low-bit controls use only a packed causal approximation of the "
            "current gate/up projections and include its resident metadata; a trained "
            "hybrid learns a direct behavioral residual over those causal scores. "
            "Held-out quality uses full-vocabulary logits; free-running rows recursively "
            "feed sparse-selected tokens. PyTorch hooks still execute the dense MLP for "
            "instrumentation before replacement, so physical bytes and latency are null."
        ),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        temporary.replace(output)


if __name__ == "__main__":
    main()
