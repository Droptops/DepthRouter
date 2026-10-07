# ruff: noqa: I001
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


DEFAULT_TEXTS = [
    "Explain why a cache miss can dominate inference latency.",
    "Write a short proof that repeated shared weights permit projection reuse from state deltas.",
    "Compare bandwidth-bound and compute-bound recurrent inference.",
    "A user asks the same question repeatedly while moving along one latent trajectory.",
    "Describe a fixed-point iteration and explain what happens to successive residuals.",
    "Explain why a shared projection W gives W(h+d)=Wh+Wd.",
    "What would make a hidden-state delta cheap to project?",
    "Explain the difference between low-rank state and low-rank state change.",
    "Describe how an incremental database view differs from recomputing the whole query.",
    "Why might recurrence create a compressible update even when the state itself is dense?",
    "Explain what stable rank measures.",
    "What does it mean for a recurrent neural state to converge?",
    "Describe a low-rank matrix update in simple terms.",
    "Why is weight sharing essential for exact projection reuse?",
    "How could a recurrent transformer cache linear projections across loop iterations?",
    "Explain when a delta update would cost less than a full matrix multiplication.",
]


class RepeatedLayer(nn.Module):
    def __init__(self, inner: nn.Module, loops: int):
        super().__init__()
        if loops < 2:
            raise ValueError("loops must be >= 2")
        self.inner = inner
        self.loops = loops
        self.inputs: list[Tensor] = []
        self.outputs: list[Tensor] = []

    def forward(self, hidden_states: Tensor, *args, **kwargs):
        x = hidden_states
        self.inputs = []
        self.outputs = []
        output = None
        for _ in range(self.loops):
            self.inputs.append(x.detach().cpu())
            output = self.inner(x, *args, **kwargs)
            if isinstance(output, tuple):
                x = output[0]
            else:
                x = output
            self.outputs.append(x.detach().cpu())

        if isinstance(output, tuple):
            return (x, *output[1:])
        return x


def resolve_layers(model: nn.Module) -> nn.ModuleList:
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    if not isinstance(layers, nn.ModuleList):
        raise ValueError("expected model.model.layers ModuleList")
    return layers


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def load_texts(path: str | None) -> list[str]:
    if path is None:
        return list(DEFAULT_TEXTS)
    rows = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("{"):
            rows.append(str(json.loads(line)["text"]))
        else:
            rows.append(line)
    if not rows:
        raise ValueError("text corpus is empty")
    return rows


def energy_rank(singular_values: Tensor, target: float) -> int:
    energy = singular_values.square()
    cumulative = energy.cumsum(0) / energy.sum().clamp_min(1e-20)
    return int((cumulative < target).sum().item() + 1)


def row_feature_fraction_for_energy(delta: Tensor, target: float) -> dict[str, float]:
    energy = delta.square()
    sorted_energy = energy.sort(dim=-1, descending=True).values
    cumulative = sorted_energy.cumsum(dim=-1)
    cumulative = cumulative / energy.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    counts = (cumulative < target).sum(dim=-1) + 1
    fractions = counts.float() / delta.shape[-1]
    return {
        "mean": float(fractions.mean().item()),
        "p50": float(torch.quantile(fractions, 0.50).item()),
        "p95": float(torch.quantile(fractions, 0.95).item()),
    }


def low_rank_projection_errors(
    delta: Tensor,
    linear: nn.Linear,
    ranks: list[int],
) -> dict[str, dict[str, float]]:
    """Approximate W*delta using a low-rank factorization of the batch delta."""

    d = delta.float()
    u, s, vh = torch.linalg.svd(d, full_matrices=False)
    weight = linear.weight.detach().float().cpu()
    exact = d @ weight.T
    denominator = exact.norm().clamp_min(1e-20)

    rows: dict[str, dict[str, float]] = {}
    n, hidden = d.shape
    out = weight.shape[0]
    baseline_flops = float(n * hidden * out)
    for rank in ranks:
        r = min(rank, s.numel())
        left = u[:, :r] * s[:r]
        compressed_projection = vh[:r] @ weight.T
        approx = left @ compressed_projection
        relative = (exact - approx).norm() / denominator

        # If delta factors are already available, projecting the factorized
        # update costs B@W^T plus A@(BW^T). SVD/factor-discovery cost is excluded
        # and must be paid/avoided by a practical representation.
        factor_flops = float(r * hidden * out + n * r * out)
        rows[str(rank)] = {
            "relative_projection_error": float(relative.item()),
            "ideal_factorized_matmul_flop_ratio": factor_flops / baseline_flops,
        }
    return rows


def top_feature_projection_errors(
    delta: Tensor,
    linear: nn.Linear,
    fractions: list[float],
) -> dict[str, dict[str, float]]:
    d = delta.float()
    weight = linear.weight.detach().float().cpu()
    exact = d @ weight.T
    denominator = exact.norm().clamp_min(1e-20)
    width = d.shape[-1]
    rows: dict[str, dict[str, float]] = {}

    for fraction in fractions:
        k = max(1, min(width, round(width * fraction)))
        selected = torch.topk(d.abs(), k=k, dim=-1, sorted=False).indices
        sparse = torch.zeros_like(d)
        sparse.scatter_(1, selected, torch.gather(d, 1, selected))
        approx = sparse @ weight.T
        rows[str(fraction)] = {
            "features": k,
            "relative_projection_error": float(
                ((exact - approx).norm() / denominator).item()
            ),
            "ideal_sparse_matmul_flop_ratio": k / width,
        }
    return rows


def projection_modules(layer: nn.Module) -> dict[str, nn.Linear]:
    """Linear maps whose inputs recur with the shared layer."""

    result: dict[str, nn.Linear] = {}
    attention = getattr(layer, "self_attn", None)
    mlp = getattr(layer, "mlp", None)

    for name in ("q_proj", "k_proj", "v_proj"):
        module = getattr(attention, name, None) if attention is not None else None
        if isinstance(module, nn.Linear):
            result[f"attn_{name}"] = module

    for name in ("gate_proj", "up_proj"):
        module = getattr(mlp, name, None) if mlp is not None else None
        if isinstance(module, nn.Linear):
            result[f"mlp_{name}"] = module

    return result


def capture_projection_inputs(
    modules: dict[str, nn.Linear],
) -> tuple[dict[str, list[Tensor]], list[torch.utils.hooks.RemovableHandle]]:
    traces: dict[str, list[Tensor]] = {name: [] for name in modules}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    for name, module in modules.items():
        def prehook(
            _module: nn.Module,
            hook_args: tuple[Tensor, ...],
            *,
            trace_name: str = name,
        ) -> None:
            traces[trace_name].append(hook_args[0].detach().cpu())

        handles.append(module.register_forward_pre_hook(prehook))

    return traces, handles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--text-file")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--loops", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--max-rows", type=int, default=256)
    parser.add_argument(
        "--ranks",
        nargs="+",
        type=int,
        default=[4, 8, 16, 32, 64],
    )
    parser.add_argument(
        "--feature-fractions",
        nargs="+",
        type=float,
        default=[0.05, 0.10, 0.25, 0.50],
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument("--json-out")
    args = parser.parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit('Install with: pip install -e ".[hf]"') from exc

    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype)
    if device.type == "cpu" and dtype == torch.float16:
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(
        load_texts(args.text_file),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_length,
    )
    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
        if isinstance(value, Tensor)
    }

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    layers = resolve_layers(model)
    layer_index = args.layer if args.layer >= 0 else len(layers) // 2
    if layer_index < 0 or layer_index >= len(layers):
        raise ValueError("layer index out of range")

    original = layers[layer_index]
    modules = projection_modules(original)
    projection_traces, trace_handles = capture_projection_inputs(modules)

    repeated = RepeatedLayer(original, args.loops)
    layers[layer_index] = repeated

    try:
        with torch.inference_mode():
            output = model(**encoded, use_cache=False)
            final_logits = output.logits[:, -1, :].detach().float().cpu()
    finally:
        for handle in trace_handles:
            handle.remove()

    for name, traces in projection_traces.items():
        if len(traces) != args.loops:
            raise RuntimeError(
                f"{name} expected {args.loops} recurrent calls, got {len(traces)}"
            )

    rows = []
    previous = repeated.inputs[0].reshape(-1, repeated.inputs[0].shape[-1]).float()
    previous = previous[: args.max_rows]

    for loop_idx in range(1, len(repeated.inputs)):
        current = repeated.inputs[loop_idx].reshape(
            -1, repeated.inputs[loop_idx].shape[-1]
        ).float()[: args.max_rows]
        delta = current - previous

        numerator = delta.norm(dim=-1)
        denominator = previous.norm(dim=-1).clamp_min(1e-20)
        relative = numerator / denominator

        singular = torch.linalg.svdvals(delta)
        stable_rank = float(
            (delta.square().sum() / singular[0].square().clamp_min(1e-20)).item()
        )
        delta_energy = delta.square().sum()
        state_energy = previous.square().sum().clamp_min(1e-20)

        projection_rows = {}
        for name, module in modules.items():
            previous_input = projection_traces[name][loop_idx - 1].reshape(
                -1,
                projection_traces[name][loop_idx - 1].shape[-1],
            ).float()[: args.max_rows]
            current_input = projection_traces[name][loop_idx].reshape(
                -1,
                projection_traces[name][loop_idx].shape[-1],
            ).float()[: args.max_rows]
            projection_delta = current_input - previous_input
            input_relative = (
                projection_delta.norm(dim=-1)
                / previous_input.norm(dim=-1).clamp_min(1e-20)
            )
            singular_input = torch.linalg.svdvals(projection_delta)

            projection_rows[name] = {
                "mean_relative_input_delta_norm": float(input_relative.mean().item()),
                "stable_rank_of_input_delta": float(
                    (
                        projection_delta.square().sum()
                        / singular_input[0].square().clamp_min(1e-20)
                    ).item()
                ),
                "rank_for_95pct_input_delta_energy": energy_rank(
                    singular_input,
                    0.95,
                ),
                "low_rank": low_rank_projection_errors(
                    projection_delta,
                    module.cpu(),
                    args.ranks,
                ),
                "top_features": top_feature_projection_errors(
                    projection_delta,
                    module.cpu(),
                    args.feature_fractions,
                ),
            }
            module.to(device)

        rows.append(
            {
                "from_loop": loop_idx,
                "to_loop": loop_idx + 1,
                "rows": int(delta.shape[0]),
                "hidden_width": int(delta.shape[1]),
                "mean_relative_delta_norm": float(relative.mean().item()),
                "p50_relative_delta_norm": float(torch.quantile(relative, 0.50).item()),
                "p95_relative_delta_norm": float(torch.quantile(relative, 0.95).item()),
                "delta_to_state_energy": float((delta_energy / state_energy).item()),
                "stable_rank": stable_rank,
                "rank_for_90pct_delta_energy": energy_rank(singular, 0.90),
                "rank_for_95pct_delta_energy": energy_rank(singular, 0.95),
                "rank_for_99pct_delta_energy": energy_rank(singular, 0.99),
                "feature_fraction_for_90pct_delta_energy": row_feature_fraction_for_energy(
                    delta,
                    0.90,
                ),
                "feature_fraction_for_95pct_delta_energy": row_feature_fraction_for_energy(
                    delta,
                    0.95,
                ),
                "feature_fraction_for_99pct_delta_energy": row_feature_fraction_for_energy(
                    delta,
                    0.99,
                ),
                "projections": projection_rows,
            }
        )
        previous = current

    payload = {
        "experiment": "hf_loop_delta_scan_v0",
        "model": args.model,
        "layer": layer_index,
        "loops": args.loops,
        "final_logit_l2": float(final_logits.norm().item()),
        "rows": rows,
        "claim_boundary": (
            "This repeats one frozen pretrained decoder layer without loop-specific "
            "training. Low-rank factorized FLOP ratios exclude the cost of discovering "
            "the factorization. A useful result here is evidence that recurrent state "
            "updates are intrinsically more compressible than the dense state; it is "
            "not yet an inference speedup."
        ),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        Path(args.json_out).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
