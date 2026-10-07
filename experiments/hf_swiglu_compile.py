# ruff: noqa: I001\nfrom __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from depth_router import (
    contiguous_swiglu_pages,
    learn_codemand_permutation,
    learn_simhash_codemand_permutation,
    page_importance,
    pages_for_mass,
    permute_swiglu_neurons_,
)


DEFAULT_TEXTS = [
    "Explain why a cache miss can dominate inference latency.",
    "Write a short proof that permuting hidden neurons with matching inverse wiring preserves a dense MLP.",
    "Compare bandwidth-bound and compute-bound inference.",
    "A user asks the same question in several different ways. What latent structure could be shared?",
    "Summarize the difference between posterior uncertainty and model confidence.",
    "Describe how a query planner coalesces repeated I/O.",
    "What information should remain resident when the future computation is uncertain?",
    "Explain why parameter count is not the same thing as bytes moved.",
    "Give an example of a function-preserving compiler optimization.",
    "What is the difference between semantic similarity and computational co-demand?",
    "Explain a rate-distortion tradeoff in simple terms.",
    "Why can adaptive routing fail if the routing metadata is too expensive?",
    "Describe a memory hierarchy from registers through device memory.",
    "Explain why calibration should target behavior rather than an arbitrary oracle label.",
    "What does it mean for two histories to induce the same predictive future?",
    "Describe how batch scheduling can turn many logical misses into one physical transfer.",
]


def parse_dtype(name: str) -> torch.dtype:
    choices = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in choices:
        raise ValueError(f"unsupported dtype: {name}")
    return choices[name]


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_texts(path: str | None) -> list[str]:
    if path is None:
        return list(DEFAULT_TEXTS)

    texts: list[str] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("{"):
            payload = json.loads(line)
            if "text" not in payload:
                raise ValueError("JSONL rows must contain a 'text' field")
            texts.append(str(payload["text"]))
        else:
            texts.append(line)

    if len(texts) < 4:
        raise ValueError("need at least four non-empty texts")
    return texts


def resolve_layers(model: nn.Module) -> list[nn.Module]:
    candidates: list[Any] = [
        getattr(getattr(model, "model", None), "layers", None),
        getattr(model, "layers", None),
        getattr(getattr(model, "transformer", None), "h", None),
    ]
    for candidate in candidates:
        if candidate is not None and len(candidate) > 0:
            return list(candidate)
    raise ValueError("could not locate decoder layers on this model")


def resolve_swiglu(layer: nn.Module) -> tuple[nn.Linear, nn.Linear, nn.Linear]:
    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        raise ValueError("selected layer has no .mlp module")

    gate = getattr(mlp, "gate_proj", None)
    up = getattr(mlp, "up_proj", None)
    down = getattr(mlp, "down_proj", None)
    if not all(isinstance(module, nn.Linear) for module in (gate, up, down)):
        raise ValueError(
            "selected MLP is not a gate_proj/up_proj/down_proj SwiGLU block"
        )
    return gate, up, down


@torch.inference_mode()
def capture_down_inputs(
    model: nn.Module,
    down_proj: nn.Linear,
    encoded: dict[str, Tensor],
    *,
    max_trace_tokens: int,
) -> Tensor:
    rows: list[Tensor] = []

    def prehook(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
        hidden = args[0].detach()
        rows.append(hidden.reshape(-1, hidden.shape[-1]).cpu())

    handle = down_proj.register_forward_pre_hook(prehook)
    try:
        model(**encoded)
    finally:
        handle.remove()

    if not rows:
        raise RuntimeError("failed to capture SwiGLU intermediate activations")
    flat = torch.cat(rows, dim=0)
    return flat[:max_trace_tokens]


@torch.inference_mode()
def model_logits(model: nn.Module, encoded: dict[str, Tensor]) -> Tensor:
    output = model(**encoded)
    logits = getattr(output, "logits", None)
    if logits is None:
        raise ValueError("model output has no logits")
    # Last-token logits are sufficient for the exact-function gate and avoid
    # retaining a full [batch, sequence, vocabulary] tensor on CPU.
    return logits[:, -1, :].detach().float().cpu()


def logit_kl(reference: Tensor, candidate: Tensor) -> Tensor:
    ref_logp = torch.log_softmax(reference, dim=-1)
    cand_logp = torch.log_softmax(candidate, dim=-1)
    p = ref_logp.exp()
    return (p * (ref_logp - cand_logp)).sum(dim=-1)


def encode_texts(
    tokenizer,
    texts: list[str],
    *,
    device: torch.device,
    max_length: int,
) -> dict[str, Tensor]:
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return {
        key: value.to(device)
        for key, value in encoded.items()
        if isinstance(value, Tensor)
    }


def page_payload_bytes(pages: list[Any]) -> int:
    return sum(int(page.payload_bytes) for page in pages)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Hugging Face model id or local model path",
    )
    parser.add_argument("--text-file")
    parser.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="-1 selects the middle decoder layer",
    )
    parser.add_argument("--units-per-page", type=int, default=256)
    parser.add_argument(
        "--compiler",
        choices=["affinity", "simhash"],
        default="simhash",
    )
    parser.add_argument("--simhash-bits", type=int, default=16)
    parser.add_argument("--mass-target", type=float, default=0.95)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-trace-tokens", type=int, default=4096)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--save-compiled")
    parser.add_argument("--json-out")
    args = parser.parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            'Install the real-model extras first: pip install -e ".[hf]"'
        ) from exc

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    dtype = parse_dtype(args.dtype)
    if device.type == "cpu" and dtype == torch.float16:
        dtype = torch.float32

    texts = load_texts(args.text_file)
    midpoint = max(len(texts) // 2, 2)
    compile_texts = texts[:midpoint]
    verify_texts = texts[midpoint:]
    if not verify_texts:
        verify_texts = texts[-2:]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    layers = resolve_layers(model)
    layer_index = args.layer
    if layer_index < 0:
        layer_index = len(layers) // 2
    if not 0 <= layer_index < len(layers):
        raise ValueError("layer index is out of range")

    gate, up, down = resolve_swiglu(layers[layer_index])
    if args.units_per_page > gate.out_features:
        raise ValueError("units-per-page exceeds intermediate width")

    compile_encoded = encode_texts(
        tokenizer,
        compile_texts,
        device=device,
        max_length=args.max_length,
    )
    verify_encoded = encode_texts(
        tokenizer,
        verify_texts,
        device=device,
        max_length=args.max_length,
    )

    before_logits = model_logits(model, verify_encoded)
    intermediate = capture_down_inputs(
        model,
        down,
        compile_encoded,
        max_trace_tokens=args.max_trace_tokens,
    )

    raw_pages = contiguous_swiglu_pages(
        gate,
        up,
        down,
        units_per_page=args.units_per_page,
    )
    raw_scores = page_importance(
        intermediate.to(down.weight.device),
        down,
        raw_pages,
    ).cpu()
    raw_working_set = pages_for_mass(
        raw_scores,
        mass_target=args.mass_target,
    ).float()

    if args.compiler == "affinity":
        permutation = learn_codemand_permutation(
            intermediate,
            down,
            units_per_page=args.units_per_page,
        )
    else:
        permutation = learn_simhash_codemand_permutation(
            intermediate,
            down,
            bits=args.simhash_bits,
            max_trace_tokens=args.max_trace_tokens,
            seed=args.seed,
        )

    model.to(device)
    gate, up, down = resolve_swiglu(layers[layer_index])
    permute_swiglu_neurons_(gate, up, down, permutation)

    after_logits = model_logits(model, verify_encoded)
    max_abs_logit_error = float((before_logits - after_logits).abs().max().item())
    kl = logit_kl(before_logits, after_logits)
    max_kl = float(kl.max().item())
    mean_kl = float(kl.mean().item())

    compiled_intermediate = intermediate[:, permutation]
    compiled_pages = contiguous_swiglu_pages(
        gate,
        up,
        down,
        units_per_page=args.units_per_page,
    )
    compiled_scores = page_importance(
        compiled_intermediate.to(down.weight.device),
        down,
        compiled_pages,
    ).cpu()
    compiled_working_set = pages_for_mass(
        compiled_scores,
        mass_target=args.mass_target,
    ).float()

    raw_mean = float(raw_working_set.mean().item())
    compiled_mean = float(compiled_working_set.mean().item())
    payload = page_payload_bytes(compiled_pages)
    result = {
        "experiment": "hf_swiglu_compile_v0",
        "model": args.model,
        "layer_index": layer_index,
        "intermediate_width": gate.out_features,
        "units_per_page": args.units_per_page,
        "compiler": args.compiler,
        "simhash_bits": args.simhash_bits if args.compiler == "simhash" else None,
        "num_pages": len(compiled_pages),
        "mass_target": args.mass_target,
        "trace_tokens": int(intermediate.shape[0]),
        "dense_function_preserved": bool(max_kl < 1e-6),
        "max_abs_logit_error": max_abs_logit_error,
        "mean_logit_kl_nats": mean_kl,
        "max_logit_kl_nats": max_kl,
        "raw_mean_pages_for_mass": raw_mean,
        "compiled_mean_pages_for_mass": compiled_mean,
        "working_set_page_reduction_fraction": (
            (raw_mean - compiled_mean) / max(raw_mean, 1e-12)
        ),
        "cold_payload_bytes": payload,
        "raw_mean_payload_bytes_for_mass": (
            raw_mean * payload / len(compiled_pages)
        ),
        "compiled_mean_payload_bytes_for_mass": (
            compiled_mean * payload / len(compiled_pages)
        ),
        "claim_boundary": (
            "This validates exact function-preserving physical neuron repacking "
            "and a structural co-demand locality metric on a real pretrained "
            "SwiGLU block. It is not yet a sparse-kernel or HBM-traffic result."
        ),
    }

    if args.save_compiled:
        output_dir = Path(args.save_compiled)
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        (output_dir / "depthrouter_layout.json").write_text(
            json.dumps(
                {
                    "source_model": args.model,
                    "layer_index": layer_index,
                    "permutation": permutation.tolist(),
                    "units_per_page": args.units_per_page,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        Path(args.json_out).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
