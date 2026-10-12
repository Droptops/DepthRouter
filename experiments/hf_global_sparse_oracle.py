from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


def atomic_write(path: str, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(destination)


DEFAULT_TEXTS = [
    "Explain why a cache miss can dominate inference latency.",
    "Write a short proof that a function-preserving neuron permutation keeps a dense MLP unchanged.",
    "Compare bandwidth-bound and compute-bound inference.",
    "A user asks the same question in several different ways. What latent structure could be shared?",
    "Summarize the difference between posterior uncertainty and model confidence.",
    "Describe how a query planner coalesces repeated I/O.",
    "What information should remain resident when the future computation is uncertain?",
    "Explain why parameter count is not the same thing as bytes moved.",
    "Give an example of a compiler optimization that changes layout but not semantics.",
    "What is the difference between semantic similarity and computational co-demand?",
    "Explain a rate-distortion tradeoff in simple terms.",
    "Why can adaptive routing fail if routing metadata is too expensive?",
    "Describe a memory hierarchy from registers through device memory.",
    "Why should calibration target behavior rather than an arbitrary oracle label?",
    "What does it mean for two histories to induce the same predictive future?",
    "Describe how batch scheduling can turn many logical misses into one physical transfer.",
]


def load_texts(path: str | None) -> list[str]:
    if path is None:
        return list(DEFAULT_TEXTS)
    rows: list[str] = []
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


def resolve_layers(model: nn.Module) -> list[nn.Module]:
    candidates: list[Any] = [
        getattr(getattr(model, "model", None), "layers", None),
        getattr(model, "layers", None),
        getattr(getattr(model, "transformer", None), "h", None),
    ]
    for candidate in candidates:
        if candidate is not None and len(candidate) > 0:
            return list(candidate)
    raise ValueError("could not locate decoder layers")


def resolve_down(layer: nn.Module) -> nn.Linear:
    mlp = getattr(layer, "mlp", None)
    down = getattr(mlp, "down_proj", None) if mlp is not None else None
    if not isinstance(down, nn.Linear):
        raise TypeError("layer does not expose a SwiGLU down projection")
    return down


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


@torch.inference_mode()
def last_logits(model: nn.Module, encoded: dict[str, Tensor]) -> Tensor:
    return model(**encoded).logits[:, -1, :].detach().float().cpu()


def distribution_tensors(reference: Tensor, candidate: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    ref_logp = torch.log_softmax(reference, dim=-1)
    cand_logp = torch.log_softmax(candidate, dim=-1)
    p = ref_logp.exp()
    kl = (p * (ref_logp - cand_logp)).sum(dim=-1)
    agreement = reference.argmax(dim=-1) == candidate.argmax(dim=-1)

    ref_top = reference.argmax(dim=-1)
    ref_top_logp = torch.gather(ref_logp, 1, ref_top[:, None]).squeeze(1)
    cand_ref_top_logp = torch.gather(cand_logp, 1, ref_top[:, None]).squeeze(1)
    drop = ref_top_logp - cand_ref_top_logp
    return kl, agreement, drop


def trajectory_metrics(
    model: nn.Module,
    dense_states: list[tuple[dict[str, Tensor], Tensor]],
) -> dict[str, float]:
    all_kl = []
    all_agreement = []
    all_drop = []
    for encoded, reference in dense_states:
        candidate = last_logits(model, encoded)
        kl, agreement, drop = distribution_tensors(reference, candidate)
        all_kl.append(kl)
        all_agreement.append(agreement)
        all_drop.append(drop)
    kl = torch.cat(all_kl)
    agreement = torch.cat(all_agreement).float()
    drop = torch.cat(all_drop)
    rollout_agreement = torch.stack(all_agreement).all(dim=0).float()

    return {
        "mean_kl_nats": float(kl.mean().item()),
        "p95_kl_nats": float(torch.quantile(kl, 0.95).item()),
        "max_kl_nats": float(kl.max().item()),
        "top1_agreement": float(agreement.mean().item()),
        "exact_rollout_fraction": float(rollout_agreement.mean().item()),
        "mean_reference_top1_logp_drop": float(drop.mean().item()),
        "p95_reference_top1_logp_drop": float(torch.quantile(drop, 0.95).item()),
    }


def install_oracle_topk_hook(down: nn.Linear, *, fraction: float):
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    column_norm = down.weight.detach().float().norm(dim=0)

    def prehook(_module: nn.Module, args: tuple[Tensor, ...]):
        activation = args[0]
        width = activation.shape[-1]
        k = max(1, min(width, round(width * fraction)))
        score = activation.detach().float().abs() * column_norm.to(activation.device)
        selected = torch.topk(score, k=k, dim=-1, sorted=False).indices
        sparse = torch.zeros_like(activation)
        values = torch.gather(activation, -1, selected)
        sparse.scatter_(-1, selected, values)
        return (sparse,)

    return down.register_forward_pre_hook(prehook)


def layer_groups(num_layers: int) -> dict[str, list[int]]:
    midpoint = num_layers // 2
    quarter = max(num_layers // 4, 1)
    return {
        "all": list(range(num_layers)),
        "early_half": list(range(midpoint)),
        "late_half": list(range(midpoint, num_layers)),
        "middle_half": list(range(quarter, num_layers - quarter)),
        "alternating": list(range(0, num_layers, 2)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--revision")
    parser.add_argument("--text-file")
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=[0.05, 0.10, 0.25, 0.50],
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["all", "early_half", "late_half", "middle_half", "alternating"],
    )
    parser.add_argument("--json-out")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--decode-steps", type=int, default=1)
    args = parser.parse_args()
    if (args.max_length < 2 or (args.max_prompts is not None and args.max_prompts < 1)
            or args.decode_steps < 1
            or any(not 0 < fraction <= 1 for fraction in args.fractions)):
        parser.error("invalid max-length/max-prompts/decode-steps/fractions")

    try:
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit('Install with: pip install -e ".[hf]"') from exc

    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype)
    if device.type == "cpu" and dtype == torch.float16:
        dtype = torch.float32
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    texts = load_texts(args.text_file)
    if args.max_prompts is not None:
        texts = texts[:args.max_prompts]
    encoded = tokenizer(
        texts,
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

    model_kwargs: dict[str, Any] = {"revision": args.revision}
    if int(transformers.__version__.split(".")[0]) >= 5:
        model_kwargs["dtype"] = dtype
    else:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs).to(device)
    model.eval()
    layers = resolve_layers(model)
    groups = layer_groups(len(layers))
    unknown = set(args.groups) - set(groups)
    if unknown:
        raise ValueError(f"unknown groups: {sorted(unknown)}")

    dense_states = []
    dense_prefix = {key: value.clone() for key, value in encoded.items()}
    for _step in range(args.decode_steps):
        reference = last_logits(model, dense_prefix)
        dense_states.append((
            {key: value.clone() for key, value in dense_prefix.items()},
            reference,
        ))
        token = reference.argmax(dim=-1).to(device)
        dense_prefix["input_ids"] = torch.cat(
            [dense_prefix["input_ids"], token[:, None]], dim=1
        )
        dense_prefix["attention_mask"] = torch.cat(
            [
                dense_prefix["attention_mask"],
                torch.ones_like(token[:, None], device=device),
            ],
            dim=1,
        )
    rows = []

    for group_name in args.groups:
        selected_layers = groups[group_name]
        for fraction in args.fractions:
            handles = [
                install_oracle_topk_hook(
                    resolve_down(layers[layer_idx]),
                    fraction=fraction,
                )
                for layer_idx in selected_layers
            ]
            try:
                quality = trajectory_metrics(model, dense_states)
            finally:
                for handle in handles:
                    handle.remove()

            rows.append(
                {
                    "group": group_name,
                    "layers_sparsified": len(selected_layers),
                    "layer_fraction": len(selected_layers) / len(layers),
                    "active_neuron_fraction": fraction,
                    "ideal_selected_payload_fraction_within_sparsified_layers": fraction,
                    "ideal_global_mlp_payload_fraction": (
                        1.0
                        - (len(selected_layers) / len(layers))
                        * (1.0 - fraction)
                    ),
                    **quality,
                }
            )

    payload = {
        "experiment": "hf_global_sparse_oracle_v1",
        "model": args.model,
        "requested_revision": args.revision,
        "model_revision": getattr(model.config, "_commit_hash", None) or args.revision,
        "tokenizer_revision": tokenizer.init_kwargs.get("_commit_hash") or args.revision,
        "seed": args.seed,
        "device": str(device),
        "dtype": str(dtype),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "prompts": len(texts),
        "decode_steps": args.decode_steps,
        "num_layers": len(layers),
        "rows": rows,
        "physical_bytes_per_token": None,
        "claim_boundary": (
            "This is a behavioral oracle. Top-k selection is computed from the "
            "already-materialized exact SwiGLU intermediate, so it does not yet "
            "save gate/up traffic. It asks whether the final predictive "
            "distribution tolerates simultaneous neuron sparsity across many layers."
        ),
    }
    try:
        payload["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        payload["git_commit"] = None
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        atomic_write(args.json_out, rendered + "\n")


if __name__ == "__main__":
    main()
