"""Balanced final-MLP partitions for an offline capability audit.

Assignment fitting uses the log-Sinkhorn and hard-forward/soft-gradient idea
from DOT-MoE, but learns no router and is not a reproduction of that method.
The caller supplies calibration-only losses and the exact hard candidate
forward. None of the routines below is a deployment selector or timing kernel.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from .output_gap import tensor_sha256


def _check_dimensions(width: int, experts: int) -> None:
    if experts < 2 or width < experts or width % experts:
        raise ValueError("width must be positive and divisible by experts >= 2")


def validate_partition(partition: Tensor, *, experts: int = 8) -> None:
    """Require one integer expert label per neuron and equal exact capacities."""
    if partition.ndim != 1 or partition.dtype != torch.long:
        raise ValueError("partition must be a one-dimensional torch.long tensor")
    _check_dimensions(partition.numel(), experts)
    if bool(((partition < 0) | (partition >= experts)).any()):
        raise ValueError("partition labels must lie within the expert range")
    counts = torch.bincount(partition, minlength=experts)
    if not bool((counts == partition.numel() // experts).all()):
        raise ValueError("partition must have exactly balanced expert capacities")


def random_partition(width: int, *, experts: int = 8, seed: int) -> Tensor:
    """Seeded balanced random labels; returns a CPU tensor indexed by neuron."""
    _check_dimensions(width, experts)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    labels = torch.arange(experts).repeat_interleave(width // experts)
    return labels[torch.randperm(width, generator=generator)]


def candidate_combinations(*, experts: int = 8, active: int = 6) -> Tensor:
    """Enumerate every unique subset in lexicographic order, once each."""
    if not 1 <= active <= experts:
        raise ValueError("active must lie between one and experts")
    return torch.tensor(list(combinations(range(experts), active)), dtype=torch.long)


def partition_indices(
    partition: Tensor,
    selected_experts: Tensor | list[int] | None = None,
    *,
    experts: int = 8,
) -> Tensor:
    """Pack by selected expert order, then original neuron order within expert.

    The same returned indices must index gate/up rows and down columns. Full
    width is a mathematical permutation, not a claim of BF16 bitwise identity.
    """
    validate_partition(partition, experts=experts)
    if selected_experts is None:
        chosen = torch.arange(experts, dtype=torch.long)
    else:
        chosen = torch.as_tensor(selected_experts)
    if chosen.ndim != 1 or chosen.dtype != torch.long or chosen.numel() == 0:
        raise ValueError("selected_experts must be a nonempty integer vector")
    if bool(((chosen < 0) | (chosen >= experts)).any()) or chosen.unique().numel() != len(
        chosen
    ):
        raise ValueError("selected_experts must be unique expert indices")
    return torch.cat([(partition == expert).nonzero().flatten() for expert in chosen.tolist()])


@torch.no_grad()
def balanced_round(scores: Tensor) -> Tensor:
    """Global descending greedy assignment with exact capacities.

    Ties are resolved by flattened neuron-major index. This is the bounded
    greedy projection used in DOT-MoE, not an optimal discrete transport solve.
    """
    if scores.ndim != 2:
        raise ValueError("scores must have shape [neurons, experts]")
    width, experts = scores.shape
    _check_dimensions(width, experts)
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("scores must be finite")
    ranking = torch.argsort(scores.detach().float().cpu().flatten(), descending=True, stable=True)
    labels = [-1] * width
    capacity = [width // experts] * experts
    remaining = width
    for entry in ranking.tolist():
        neuron, expert = divmod(entry, experts)
        if labels[neuron] < 0 and capacity[expert] > 0:
            labels[neuron] = expert
            capacity[expert] -= 1
            remaining -= 1
            if remaining == 0:
                break
    result = torch.tensor(labels, dtype=torch.long, device=scores.device)
    validate_partition(result, experts=experts)
    return result


def sinkhorn_assignment(logits: Tensor, *, temperature: float, iterations: int) -> Tensor:
    """FP32 log-domain transport with row mass 1 and column mass width/experts.

    A finite iteration count approximates the soft row marginal. Exact integer
    capacities are enforced separately by ``balanced_round``.
    """
    if logits.ndim != 2:
        raise ValueError("logits must have shape [neurons, experts]")
    width, experts = logits.shape
    _check_dimensions(width, experts)
    if temperature <= 0 or not math.isfinite(temperature) or iterations < 1:
        raise ValueError("temperature and iterations must be positive")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must be finite")
    log_plan = logits.float() / temperature
    target_log_mass = math.log(width / experts)
    for _ in range(iterations):
        log_plan = log_plan - torch.logsumexp(log_plan, dim=1, keepdim=True)
        log_plan = log_plan - torch.logsumexp(log_plan, dim=0, keepdim=True) + target_log_mass
    return log_plan.exp()


def straight_through_assignment(
    logits: Tensor, *, temperature: float, iterations: int
) -> tuple[Tensor, Tensor]:
    """Return exactly binary forward assignments and soft Sinkhorn gradients."""
    soft = sinkhorn_assignment(logits, temperature=temperature, iterations=iterations)
    labels = balanced_round(soft)
    hard = F.one_hot(labels, num_classes=logits.shape[1]).to(soft.dtype)
    return hard + (soft - soft.detach()), labels


@torch.no_grad()
def coactivation_partition(
    calibration_activations: Tensor,
    *,
    experts: int = 8,
    seed: int,
    sketch_dim: int = 32,
    iterations: int = 8,
) -> Tensor:
    """Balanced cosine clustering of absolute calibration activation traces.

    A seeded Rademacher projection sketches neuron traces over calibration
    states, avoiding the quadratic neuron-by-neuron affinity matrix. Seeded
    initial neuron centers and a fixed iteration count make this a bounded
    construction recipe. Returns CPU labels; no development data are needed.
    """
    if calibration_activations.ndim != 2 or calibration_activations.shape[0] < 1:
        raise ValueError("calibration_activations must have shape [states, neurons]")
    count, width = calibration_activations.shape
    _check_dimensions(width, experts)
    if sketch_dim < 1 or iterations < 1:
        raise ValueError("sketch_dim and iterations must be positive")
    if not bool(torch.isfinite(calibration_activations).all()):
        raise ValueError("calibration activations must be finite")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    projection = torch.randint(2, (count, sketch_dim), generator=generator).float()
    projection = (2 * projection - 1).to(calibration_activations.device) / math.sqrt(sketch_dim)
    traces = F.normalize(calibration_activations.detach().float().abs(), dim=0)
    features = F.normalize(traces.T @ projection, dim=1)
    centers = features[torch.randperm(width, generator=generator)[:experts].to(features.device)]
    for _ in range(iterations):
        labels = balanced_round(features @ centers.T)
        centers = torch.stack([features[labels == expert].mean(dim=0) for expert in range(experts)])
        centers = F.normalize(centers, dim=1)
    return labels.cpu()


@dataclass(frozen=True)
class AssignmentRecipe:
    """Frozen paired construction budget; no validation-driven early stopping."""

    steps: int = 100
    batch_size: int = 16
    candidates_per_step: int = 4
    learning_rate: float = 0.03
    sinkhorn_iterations: int = 10
    temperature_start: float = 1.0
    temperature_end: float = 0.2
    initial_assignment_strength: float = 0.1
    initial_noise_std: float = 0.01


LossCallback = Callable[[str, Tensor, Tensor], Tensor]


def fit_assignment_pair(
    initial_partition: Tensor,
    num_examples: int,
    loss_callback: LossCallback,
    *,
    seed: int,
    device: torch.device | str,
    experts: int = 8,
    active: int = 6,
    recipe: AssignmentRecipe | None = None,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """Fit hidden and output objectives with exactly paired construction inputs.

    ``loss_callback(objective, row_ids, masks)`` receives objective ``hidden``
    or ``output``, CPU row indices, and device-resident exactly binary forward
    masks of shape [candidates, neurons] with STE gradients. It must return a
    differentiable [candidates, batch] matrix. The caller must restrict row IDs
    to calibration data, freeze all teacher/model weights, preserve the actual
    hard candidate forward, and check any wall-time budget inside the callback.

    For every token, backpropagation uses the lowest-loss sampled candidate
    (first candidate breaks ties). The final iterate is frozen without looking
    at development or held-out results. Returned partitions live on CPU.
    Callback exceptions propagate unchanged with an ``assignment_partial``
    attribute containing completed and current tensors plus partial audit
    metadata, so the launcher can preserve interrupted fits honestly.
    """
    recipe = recipe or AssignmentRecipe()
    validate_partition(initial_partition, experts=experts)
    combos = candidate_combinations(experts=experts, active=active)
    if num_examples < 1 or recipe.steps < 1 or recipe.batch_size < 1:
        raise ValueError("num_examples, steps and batch_size must be positive")
    if not 1 <= recipe.candidates_per_step <= len(combos):
        raise ValueError("candidates_per_step must lie within the candidate count")
    positive_values = (
        recipe.learning_rate,
        recipe.temperature_start,
        recipe.temperature_end,
    )
    if any(value <= 0 or not math.isfinite(value) for value in positive_values):
        raise ValueError("learning rate and temperatures must be finite and positive")
    if recipe.sinkhorn_iterations < 1 or recipe.initial_noise_std < 0:
        raise ValueError("invalid Sinkhorn iterations or initialization noise")
    if not math.isfinite(recipe.initial_assignment_strength) or not math.isfinite(
        recipe.initial_noise_std
    ):
        raise ValueError("initialization values must be finite")

    # A private CPU generator prevents either objective from altering the next
    # one's initialization, data order, or sampled candidate schedule.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    width = initial_partition.numel()
    initial = F.one_hot(initial_partition.cpu(), num_classes=experts).float()
    initial = recipe.initial_assignment_strength * initial + recipe.initial_noise_std * torch.randn(
        width, experts, generator=generator
    )
    batch_size = min(recipe.batch_size, num_examples)
    batches = torch.randint(num_examples, (recipe.steps, batch_size), generator=generator)
    candidate_ids = torch.stack(
        [
            torch.randperm(len(combos), generator=generator)[: recipe.candidates_per_step]
            for _ in range(recipe.steps)
        ]
    )
    expert_masks = F.one_hot(combos, num_classes=experts).sum(dim=1).float().to(device)
    partitions: dict[str, Tensor] = {}
    records: dict[str, Any] = {}
    with torch.inference_mode(False), torch.enable_grad():
        expert_masks = expert_masks.clone()
        batches = batches.clone()
        candidate_ids = candidate_ids.clone()
        for objective in ("hidden", "output"):
            logits = torch.nn.Parameter(initial.to(device).clone())
            optimizer = torch.optim.Adam([logits], lr=recipe.learning_rate, weight_decay=0.0)
            losses: list[float] = []
            gradient_norms: list[float] = []
            for step in range(recipe.steps):
                fraction = step / max(recipe.steps - 1, 1)
                temperature = recipe.temperature_start + fraction * (
                    recipe.temperature_end - recipe.temperature_start
                )
                assignment, current_labels = straight_through_assignment(
                    logits, temperature=temperature, iterations=recipe.sinkhorn_iterations
                )
                masks = expert_masks[candidate_ids[step].to(device)] @ assignment.T
                try:
                    matrix = loss_callback(objective, batches[step], masks)
                except Exception as error:
                    error.assignment_partial = {
                        "completed_partitions": partitions.copy(),
                        "current_partition": current_labels.detach().cpu(),
                        "current_logits": logits.detach().cpu(),
                        "audit": {
                            "status": "incomplete",
                            "seed": seed,
                            "recipe": asdict(recipe),
                            "effective_batch_size": batch_size,
                            "current_objective": objective,
                            "current_completed_steps": step,
                            "current_loss_by_step": losses,
                            "current_gradient_norm_by_step": gradient_norms,
                            "completed_objectives": records.copy(),
                            "initial_logits_sha256": tensor_sha256(initial),
                            "minibatch_indices_sha256": tensor_sha256(batches),
                            "candidate_indices_sha256": tensor_sha256(candidate_ids),
                            "exception_type": type(error).__name__,
                            "exception_message": str(error),
                            "router_trained": False,
                        },
                    }
                    raise
                if matrix.shape != (recipe.candidates_per_step, batch_size):
                    raise ValueError("callback must return a [candidates, batch] loss matrix")
                if not bool(torch.isfinite(matrix).all()):
                    raise FloatingPointError(f"nonfinite {objective} construction loss")
                loss = matrix.min(dim=0).values.mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if logits.grad is None or not bool(torch.isfinite(logits.grad).all()):
                    raise FloatingPointError(f"missing or nonfinite {objective} assignment gradient")
                losses.append(float(loss.detach()))
                gradient_norms.append(float(logits.grad.norm()))
                optimizer.step()
            with torch.no_grad():
                final_soft = sinkhorn_assignment(
                    logits,
                    temperature=recipe.temperature_end,
                    iterations=recipe.sinkhorn_iterations,
                )
                final_partition = balanced_round(final_soft).cpu()
            partitions[objective] = final_partition
            records[objective] = {
                "loss_by_step": losses,
                "gradient_norm_by_step": gradient_norms,
                "initial_logits_sha256": tensor_sha256(initial),
                "minibatch_indices_sha256": tensor_sha256(batches),
                "candidate_indices_sha256": tensor_sha256(candidate_ids),
                "final_logits_sha256": tensor_sha256(logits),
                "final_partition_sha256": tensor_sha256(final_partition),
                "neurons_changed_from_initial": int(
                    (final_partition != initial_partition.cpu()).sum()
                ),
                "max_soft_row_mass_error": float((final_soft.sum(dim=1) - 1).abs().max()),
                "max_soft_column_mass_error": float(
                    (final_soft.sum(dim=0) - width / experts).abs().max()
                ),
            }
    audit = {
        "seed": seed,
        "recipe": asdict(recipe),
        "effective_batch_size": batch_size,
        "parameter_count": width * experts,
        "initial_partition_sha256": tensor_sha256(initial_partition),
        "initialization_equal": True,
        "minibatch_order_equal": True,
        "candidate_order_equal": True,
        "selection": "per-token hard minimum over shared sampled candidate set",
        "checkpoint_rule": "final iterate",
        "optimizer": {"name": "Adam", "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0},
        "router_trained": False,
        "objectives": records,
    }
    return partitions, audit
