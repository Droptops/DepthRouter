from __future__ import annotations

import torch
from torch import Tensor


def _validate(
    page_probabilities: Tensor,
    page_bytes: Tensor,
    resident_probability: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor]:
    if page_probabilities.ndim != 2:
        raise ValueError("page_probabilities must have shape [tokens, pages]")
    if page_bytes.ndim != 1 or page_bytes.shape[0] != page_probabilities.shape[1]:
        raise ValueError("page_bytes must have shape [pages]")
    if bool((page_probabilities < 0).any()) or bool((page_probabilities > 1).any()):
        raise ValueError("page probabilities must be in [0, 1]")
    if bool((page_bytes < 0).any()):
        raise ValueError("page bytes must be non-negative")

    probabilities = page_probabilities.float()
    nbytes = page_bytes.to(
        device=probabilities.device,
        dtype=probabilities.dtype,
    )

    if resident_probability is None:
        resident = torch.zeros_like(nbytes)
    else:
        if resident_probability.ndim != 1 or resident_probability.shape != nbytes.shape:
            raise ValueError("resident_probability must have shape [pages]")
        if bool((resident_probability < 0).any()) or bool(
            (resident_probability > 1).any()
        ):
            raise ValueError("resident probabilities must be in [0, 1]")
        resident = resident_probability.to(
            device=probabilities.device,
            dtype=probabilities.dtype,
        )

    return probabilities, nbytes, resident


def page_fault_probability(page_probabilities: Tensor) -> Tensor:
    """Probability each page is requested by at least one token in the batch.

    The formula assumes token page requests are conditionally independent given
    the router probabilities. It is differentiable and exact under that model.
    """

    if page_probabilities.ndim != 2:
        raise ValueError("page_probabilities must have shape [tokens, pages]")
    if bool((page_probabilities < 0).any()) or bool((page_probabilities > 1).any()):
        raise ValueError("page probabilities must be in [0, 1]")

    probabilities = page_probabilities.float()
    return 1.0 - torch.prod(1.0 - probabilities, dim=0)


def expected_union_bytes(
    page_probabilities: Tensor,
    page_bytes: Tensor,
    *,
    resident_probability: Tensor | None = None,
) -> Tensor:
    """Expected *new unique bytes* loaded for a coalesced batch.

    For token-page request probabilities q_ij:

        P(page j is needed) = 1 - prod_i (1 - q_ij)

    A resident page contributes no new transfer. Soft residency in [0, 1] is
    supported so the expression can also be used inside a differentiable cache
    policy.
    """

    probabilities, nbytes, resident = _validate(
        page_probabilities,
        page_bytes,
        resident_probability,
    )
    fault_probability = page_fault_probability(probabilities)
    return (fault_probability * (1.0 - resident) * nbytes).sum()


def expected_independent_bytes(
    page_probabilities: Tensor,
    page_bytes: Tensor,
    *,
    resident_probability: Tensor | None = None,
) -> Tensor:
    """Expected bytes if every token request paid for its own page transfer."""

    probabilities, nbytes, resident = _validate(
        page_probabilities,
        page_bytes,
        resident_probability,
    )
    return (
        probabilities
        * (1.0 - resident)[None, :]
        * nbytes[None, :]
    ).sum()


def coalescing_gain(
    page_probabilities: Tensor,
    page_bytes: Tensor,
    *,
    resident_probability: Tensor | None = None,
    epsilon: float = 1e-12,
) -> Tensor:
    """Independent expected traffic divided by coalesced expected traffic."""

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    independent = expected_independent_bytes(
        page_probabilities,
        page_bytes,
        resident_probability=resident_probability,
    )
    union = expected_union_bytes(
        page_probabilities,
        page_bytes,
        resident_probability=resident_probability,
    )
    return independent / union.clamp_min(epsilon)


def normalized_union_cost(
    page_probabilities: Tensor,
    page_bytes: Tensor,
    *,
    resident_probability: Tensor | None = None,
) -> Tensor:
    """Expected coalesced bytes as a fraction of the non-resident page bank."""

    _, nbytes, resident = _validate(
        page_probabilities,
        page_bytes,
        resident_probability,
    )
    denominator = ((1.0 - resident) * nbytes).sum().clamp_min(1.0)
    return expected_union_bytes(
        page_probabilities,
        page_bytes,
        resident_probability=resident_probability,
    ) / denominator


def normalized_independent_cost(
    page_probabilities: Tensor,
    page_bytes: Tensor,
    *,
    resident_probability: Tensor | None = None,
) -> Tensor:
    """Mean per-token traffic as a fraction of the non-resident page bank."""

    probabilities, nbytes, resident = _validate(
        page_probabilities,
        page_bytes,
        resident_probability,
    )
    denominator = ((1.0 - resident) * nbytes).sum().clamp_min(1.0)
    total = expected_independent_bytes(
        probabilities,
        nbytes,
        resident_probability=resident,
    )
    return total / (max(probabilities.shape[0], 1) * denominator)
