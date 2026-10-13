"""Causal working-set predictors and structured operator-page helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

SPLIT_SIZES = {"calibration": 256, "development": 128, "heldout": 512}
TASKS = ("classification", "extraction", "numerical", "generation")


def _classification_example(split: str, index: int) -> tuple[str, str, str]:
    if split == "calibration":
        value = 17 + index * 7
        label = "EVEN" if value % 2 == 0 else "ODD"
        return (
            f"Classify the integer {value} as EVEN or ODD. Reply with one label only.",
            label,
            "parity",
        )
    if split == "development":
        value = index * 11 - 173
        label = "NEGATIVE" if value < 0 else "NONNEGATIVE"
        return (
            f"Is {value} NEGATIVE or NONNEGATIVE? Reply with one label only.",
            label,
            "sign",
        )
    value = 101 + index * 13
    label = "DIVISIBLE" if value % 3 == 0 else "NOT_DIVISIBLE"
    return (
        f"Decide whether {value} is divisible by three. Reply DIVISIBLE or NOT_DIVISIBLE.",
        label,
        "divisibility",
    )


def _extraction_example(split: str, index: int) -> tuple[str, str, str]:
    code = f"{(index * 37 + 113) % 10000:04d}"
    if split == "calibration":
        return (
            f"Record: owner=Iris-{index}; access_code={code}; zone=west. Return the access code only.",
            code,
            "key_value",
        )
    if split == "development":
        return (
            f"Ticket D{index:03d} says [region:north] [pin:{code}] [status:open]. Extract the pin.",
            code,
            "bracket_field",
        )
    return (
        f"Audit line {index}: checksum<{code}> follows source node-{index % 17}. Give only the checksum.",
        code,
        "tagged_field",
    )


def _numerical_example(split: str, index: int) -> tuple[str, str, str]:
    left = 9 + (index * 5) % 89
    right = 3 + (index * 7) % 47
    if split == "calibration":
        answer = left + right
        return (f"Compute {left} + {right}. Return only the number.", str(answer), "addition")
    if split == "development":
        answer = left - right
        return (f"Compute {left} - {right}. Return only the number.", str(answer), "subtraction")
    factor = 2 + index % 8
    answer = left * factor
    return (f"Compute {left} times {factor}. Return only the number.", str(answer), "multiplication")


def _generation_example(split: str, index: int) -> tuple[str, str, str]:
    nouns = (
        "cache eviction",
        "memory bandwidth",
        "batch scheduling",
        "operator fusion",
        "numerical stability",
        "load balancing",
        "data locality",
        "latency variance",
    )
    topic = nouns[index % len(nouns)]
    if split == "calibration":
        prompt = f"Explain {topic} in one short sentence for a new systems student. Variant {index}."
        family = "explanation"
    elif split == "development":
        prompt = f"Give one practical consequence of {topic}, using at most twenty words. Case {index}."
        family = "consequence"
    else:
        prompt = f"State one falsifiable claim about {topic} in a single concise sentence. Trial {index}."
        family = "falsifiable_claim"
    return prompt, f"A concise statement about {topic}.", family


def make_evaluation_manifest() -> list[dict[str, str]]:
    """Create deterministic, balanced, template-disjoint experiment splits."""

    builders = {
        "classification": _classification_example,
        "extraction": _extraction_example,
        "numerical": _numerical_example,
        "generation": _generation_example,
    }
    rows: list[dict[str, str]] = []
    for split, split_size in SPLIT_SIZES.items():
        per_task = split_size // len(TASKS)
        for task in TASKS:
            for index in range(per_task):
                prompt, answer, family = builders[task](split, index)
                rows.append(
                    {
                        "id": f"{split[:3]}-{task[:3]}-{index:04d}",
                        "split": split,
                        "task": task,
                        "template_family": family,
                        "prompt": prompt,
                        "answer": answer,
                    }
                )
    return rows


def validate_manifest(rows: Sequence[dict[str, str]]) -> None:
    """Validate size, balance, identity, and template-family isolation."""

    ids = [row["id"] for row in rows]
    prompts = [row["prompt"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("manifest IDs must be unique")
    if len(prompts) != len(set(prompts)):
        raise ValueError("manifest prompts must be unique")
    for split, expected_size in SPLIT_SIZES.items():
        split_rows = [row for row in rows if row["split"] == split]
        if len(split_rows) != expected_size:
            raise ValueError(f"{split} must contain {expected_size} rows")
        counts = {task: sum(row["task"] == task for row in split_rows) for task in TASKS}
        if len(set(counts.values())) != 1:
            raise ValueError(f"{split} is not task balanced")
    families = {
        split: {row["template_family"] for row in rows if row["split"] == split}
        for split in SPLIT_SIZES
    }
    for left_index, left in enumerate(SPLIT_SIZES):
        for right in list(SPLIT_SIZES)[left_index + 1 :]:
            if families[left] & families[right]:
                raise ValueError("template families must not cross splits")


class LowRankWorkingSetPredictor(nn.Module):
    """A small causal MLP that predicts per-neuron or per-page utility."""

    def __init__(self, input_width: int, rank: int, output_width: int) -> None:
        super().__init__()
        if min(input_width, rank, output_width) <= 0:
            raise ValueError("predictor widths must be positive")
        self.input_width = input_width
        self.rank = rank
        self.output_width = output_width
        self.down = nn.Linear(input_width, rank)
        self.up = nn.Linear(rank, output_width)
        self.register_buffer("feature_mean", torch.zeros(input_width))
        self.register_buffer("feature_scale", torch.ones(input_width))

    def set_normalization(self, features: Tensor) -> None:
        if features.ndim != 2 or features.shape[1] != self.input_width:
            raise ValueError("feature shape does not match predictor input")
        self.feature_mean.copy_(features.mean(dim=0))
        self.feature_scale.copy_(features.std(dim=0).clamp_min(1e-4))

    def forward(self, features: Tensor) -> Tensor:
        normalized = (features - self.feature_mean) / self.feature_scale
        return self.up(F.gelu(self.down(normalized)))

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def topk_rows(scores: Tensor, retained_fraction: float) -> Tensor:
    """Return sorted top-k indices independently for each score row."""

    if scores.ndim != 2:
        raise ValueError("scores must be two-dimensional")
    if not 0 < retained_fraction <= 1:
        raise ValueError("retained_fraction must be in (0, 1]")
    retained = max(1, min(scores.shape[1], round(scores.shape[1] * retained_fraction)))
    return scores.topk(retained, dim=1, sorted=False).indices.sort(dim=1).values


def row_set_metrics(predicted: Tensor, oracle: Tensor) -> dict[str, Tensor]:
    """Compute per-row recall and Jaccard for equal-width integer selections."""

    if predicted.ndim != 2 or oracle.ndim != 2 or predicted.shape != oracle.shape:
        raise ValueError("predicted and oracle selections must have matching 2-D shapes")
    matches = (predicted.unsqueeze(2) == oracle.unsqueeze(1)).any(dim=2).sum(dim=1).float()
    width = float(predicted.shape[1])
    return {
        "recall": matches / width,
        "jaccard": matches / (2.0 * width - matches),
    }


def contiguous_operator_pages(width: int, page_count: int) -> list[Tensor]:
    """Partition an intermediate dimension into contiguous operator pages."""

    if width <= 0 or page_count <= 0 or page_count > width:
        raise ValueError("page_count must be in [1, width]")
    return [page for page in torch.tensor_split(torch.arange(width), page_count) if page.numel()]


def aggregate_page_scores(neuron_scores: Tensor, pages: Sequence[Tensor]) -> Tensor:
    """Sum neuron utility into contiguous operator pages."""

    if neuron_scores.ndim != 2:
        raise ValueError("neuron_scores must be two-dimensional")
    return torch.stack(
        [neuron_scores.index_select(1, page.to(neuron_scores.device)).sum(dim=1) for page in pages],
        dim=1,
    )


def expand_page_selections(page_selections: Tensor, pages: Sequence[Tensor]) -> Tensor:
    """Expand equal-size selected pages to sorted neuron indices."""

    if page_selections.ndim != 2:
        raise ValueError("page_selections must be two-dimensional")
    page_widths = {page.numel() for page in pages}
    if len(page_widths) != 1:
        raise ValueError("equal-width pages are required for batched expansion")
    rows = []
    for selection in page_selections:
        selected = [pages[int(page_index)] for page_index in selection]
        rows.append(torch.cat(selected).sort().values)
    return torch.stack(rows)


@dataclass(frozen=True)
class PredictorFootprint:
    parameters: int
    bytes_bfloat16: int
    macs_per_token: int
    fraction_of_swiglu_macs: float


def predictor_footprint(
    predictor: LowRankWorkingSetPredictor,
    *,
    hidden_width: int,
    intermediate_width: int,
) -> PredictorFootprint:
    """Return analytic parameter and multiply-accumulate overhead."""

    macs = predictor.input_width * predictor.rank + predictor.rank * predictor.output_width
    swiglu_macs = 3 * hidden_width * intermediate_width
    return PredictorFootprint(
        parameters=predictor.parameter_count,
        bytes_bfloat16=2 * predictor.parameter_count,
        macs_per_token=macs,
        fraction_of_swiglu_macs=macs / swiglu_macs,
    )


def retained_count(width: int, fraction: float) -> int:
    """Expose the experiment's deterministic fraction-to-width rule."""

    if width <= 0 or not 0 < fraction <= 1:
        raise ValueError("invalid width or fraction")
    return max(1, min(width, math.floor(width * fraction + 0.5)))
