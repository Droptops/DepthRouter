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
    "Write a short proof that a function-preserving neuron permutation keeps a dense MLP unchanged.",
    "Compare bandwidth-bound and compute-bound inference.",
    "A user asks the same question in several different ways. What latent structure could be shared?",
    "Summarize the difference between posterior uncertainty and model confidence.",
    "Describe how a query planner coalesces repeated I/O.",
    "What information should remain resident when future computation is uncertain?",
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
    rows = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("{"):
            rows.append(str(json.loads(line)["text"]))
        else:
            rows.append(line)
    if len(rows) < 4:
        raise ValueError("need at least four texts")
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


def resolve_mlp(layer: nn.Module) -> nn.Module:
    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        raise TypeError("layer has no MLP")
    return mlp


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


def encode(tokenizer, texts: list[str], device: torch.device, max_length: int) -> dict[str, Tensor]:
    batch = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return {
        key: value.to(device)
        for key, value in batch.items()
        if isinstance(value, Tensor)
    }


@torch.inference_mode()
def capture_mlp_io(
    model: nn.Module,
    layer: nn.Module,
    encoded: dict[str, Tensor],
    *,
    max_tokens: int,
) -> tuple[Tensor, Tensor]:
    mlp = resolve_mlp(layer)
    xs: list[Tensor] = []
    ys: list[Tensor] = []

    def prehook(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
        xs.append(args[0].detach().reshape(-1, args[0].shape[-1]).cpu())

    def posthook(
        _module: nn.Module,
        _args: tuple[Tensor, ...],
        output: Tensor,
    ) -> None:
        ys.append(output.detach().reshape(-1, output.shape[-1]).cpu())

    h1 = mlp.register_forward_pre_hook(prehook)
    h2 = mlp.register_forward_hook(posthook)
    try:
        model(**encoded)
    finally:
        h1.remove()
        h2.remove()

    return (
        torch.cat(xs)[:max_tokens].float(),
        torch.cat(ys)[:max_tokens].float(),
    )


def kmeans(
    y: Tensor,
    k: int,
    *,
    iterations: int,
    seed: int,
) -> tuple[Tensor, Tensor]:
    k = min(k, y.shape[0])
    generator = torch.Generator().manual_seed(seed)

    # Farthest-ish initialization: random first center, then farthest residual.
    first = int(torch.randint(y.shape[0], (1,), generator=generator).item())
    chosen = [first]
    distance = torch.cdist(y, y[first : first + 1]).squeeze(1)
    for _ in range(1, k):
        idx = int(torch.argmax(distance).item())
        chosen.append(idx)
        candidate = torch.cdist(y, y[idx : idx + 1]).squeeze(1)
        distance = torch.minimum(distance, candidate)

    centers = y[torch.tensor(chosen)].clone()
    labels = torch.zeros(y.shape[0], dtype=torch.long)
    for _ in range(iterations):
        distances = torch.cdist(y, centers)
        labels = distances.argmin(dim=-1)
        updated = []
        for cluster in range(k):
            members = y[labels == cluster]
            if members.numel() == 0:
                updated.append(centers[cluster])
            else:
                updated.append(members.mean(dim=0))
        centers = torch.stack(updated)

    return centers, labels


def train_router(
    x: Tensor,
    labels: Tensor,
    *,
    classes: int,
    steps: int,
    lr: float,
    seed: int,
) -> nn.Linear:
    torch.manual_seed(seed)
    router = nn.Linear(x.shape[-1], classes)
    optimizer = torch.optim.AdamW(router.parameters(), lr=lr)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(router(x), labels)
        loss.backward()
        optimizer.step()
    return router


def relative_error(reference: Tensor, candidate: Tensor) -> dict[str, float]:
    error = (
        (reference - candidate).norm(dim=-1)
        / reference.norm(dim=-1).clamp_min(1e-12)
    )
    return {
        "mean_relative_l2": float(error.mean().item()),
        "p50_relative_l2": float(torch.quantile(error, 0.50).item()),
        "p95_relative_l2": float(torch.quantile(error, 0.95).item()),
        "fraction_below_20pct_error": float((error <= 0.20).float().mean().item()),
        "fraction_below_10pct_error": float((error <= 0.10).float().mean().item()),
        "fraction_below_5pct_error": float((error <= 0.05).float().mean().item()),
    }


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--text-file")
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--max-train-tokens", type=int, default=192)
    parser.add_argument("--max-test-tokens", type=int, default=192)
    parser.add_argument("--layers", nargs="+", type=int, default=[0, 12, 23])
    parser.add_argument("--clusters", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--kmeans-iterations", type=int, default=12)
    parser.add_argument("--router-steps", type=int, default=300)
    parser.add_argument("--router-lr", type=float, default=1e-2)
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

    texts = load_texts(args.text_file)
    split = max(2, len(texts) // 2)
    train_texts = texts[:split]
    test_texts = texts[split:]
    if len(test_texts) < 2:
        raise ValueError("held-out split needs at least two texts")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    layers = resolve_layers(model)
    if any(idx < 0 or idx >= len(layers) for idx in args.layers):
        raise ValueError("layer index out of range")

    train_encoded = encode(tokenizer, train_texts, device, args.max_length)
    test_encoded = encode(tokenizer, test_texts, device, args.max_length)

    rows = []
    for layer_idx in args.layers:
        layer = layers[layer_idx]
        x_train, y_train = capture_mlp_io(
            model,
            layer,
            train_encoded,
            max_tokens=args.max_train_tokens,
        )
        x_test, y_test = capture_mlp_io(
            model,
            layer,
            test_encoded,
            max_tokens=args.max_test_tokens,
        )
        cold_bytes_fp16 = parameter_count(resolve_mlp(layer)) * 2
        hidden = x_train.shape[-1]

        cluster_rows = {}
        for requested_k in args.clusters:
            centers, labels = kmeans(
                y_train,
                requested_k,
                iterations=args.kmeans_iterations,
                seed=layer_idx + requested_k,
            )
            k = centers.shape[0]

            test_distances = torch.cdist(y_test, centers)
            oracle_labels = test_distances.argmin(dim=-1)
            oracle_prediction = centers[oracle_labels]

            router = train_router(
                x_train,
                labels,
                classes=k,
                steps=args.router_steps,
                lr=args.router_lr,
                seed=1000 + layer_idx + requested_k,
            )
            with torch.no_grad():
                predicted_labels = router(x_test).argmax(dim=-1)
            routed_prediction = centers[predicted_labels]

            metadata_parameters = k * hidden + k + centers.numel()
            metadata_bytes_fp16 = metadata_parameters * 2
            cluster_rows[str(requested_k)] = {
                "actual_clusters": k,
                "metadata_bytes_fp16": metadata_bytes_fp16,
                "metadata_fraction_of_cold": metadata_bytes_fp16 / cold_bytes_fp16,
                "router_matches_oracle_cluster": float(
                    (predicted_labels == oracle_labels).float().mean().item()
                ),
                "oracle_vq": relative_error(y_test, oracle_prediction),
                "routed_vq": relative_error(y_test, routed_prediction),
            }

        rows.append(
            {
                "layer": layer_idx,
                "train_tokens": int(x_train.shape[0]),
                "test_tokens": int(x_test.shape[0]),
                "clusters": cluster_rows,
            }
        )

    payload = {
        "experiment": "hf_predictive_state_vq_v0",
        "model": args.model,
        "layers": rows,
        "claim_boundary": (
            "Output-space vector quantization defines empirical predictive-state "
            "classes on the calibration split. Oracle VQ measures compressibility; "
            "the linear router is the deployable proxy. Held-out prompts are used "
            "for all reported distortion metrics."
        ),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        Path(args.json_out).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
