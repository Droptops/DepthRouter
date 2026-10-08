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
    "Explain how branch prediction differs from semantic routing.",
    "Why can a large model have a small active working set?",
    "Describe a compiler pass that improves memory locality.",
    "How can held-out calibration prevent a routing policy from overclaiming?",
    "What does posterior concentration mean for future computation?",
    "Explain why a cache should track expected value per byte.",
    "How can repeated workloads create reusable computational trajectories?",
    "Describe the difference between a hot interpreter and cold program memory.",
    "Why is activation magnitude not necessarily predictive importance?",
    "What would falsify a neural virtual-memory hypothesis?",
    "Explain the role of a page table in an operating system.",
    "How could a neural model learn an address for cold weights?",
    "Why can distributed representations make neuron paging look dense?",
    "Describe prediction-conditioned sparsity.",
    "How can low-bit metadata be useful for routing without being useful for generation?",
    "Explain why final-logit behavior can tolerate local MLP approximation error.",
    "Describe predictive coding as a baseline plus sparse residual corrections.",
    "Why might the residual after dominant neurons be easier to predict?",
    "How can a small hot operator and cold exact pages cooperate?",
    "What is the difference between approximating a full nonlinear map and its residual?",
    "Explain why exact sparse corrections can protect a cheap surrogate from worst cases.",
    "How would you account for resident metadata in a bandwidth budget?",
    "Why can a low-rank residual predictor be useful even when a low-rank full predictor fails?",
    "What would make a sparse correction architecture hardware efficient?",
    "Explain selective prediction and abstention.",
    "Why should an uncertain sparse router fall back to dense execution?",
    "How can two cheap predictors provide a disagreement signal?",
    "What is the difference between average byte savings and worst-case byte savings?",
    "How can a calibrated fallback preserve quality while reducing average traffic?",
    "Why is a conditional page fault safer than unconditional sparsification?",
    "Describe a confidence threshold calibrated on held-out data.",
    "What would make an uncertainty-triggered memory fault useful?",
    "Explain why predictive equivalence can be easier than reconstructing an internal vector exactly.",
    "How can a small hot model approximate a much larger cold operator on a narrow state distribution?",
    "Why should a last-layer approximation be judged in logit space?",
    "What does it mean for compute to refine a probability trajectory?",
    "How can an approximate operator become a cache entry?",
    "Why might the final transformer layer be more compressible for prediction than for representation?",
    "Describe an anytime predictor whose hot path is cheap and whose cold path is exact.",
    "What evidence would justify replacing a dense operator with a resident surrogate?",
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
    if len(rows) < 12:
        raise ValueError("need at least twelve texts")
    return rows


def resolve_layers(model: nn.Module) -> list[nn.Module]:
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None) if inner is not None else None
    if layers is None:
        raise ValueError("could not locate decoder layers")
    return list(layers)


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


def encode(tokenizer, rows: list[str], device: torch.device, max_length: int) -> dict[str, Tensor]:
    batch = tokenizer(
        rows,
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
def capture_last_layer(
    model: nn.Module,
    layer: nn.Module,
    encoded: dict[str, Tensor],
    *,
    max_tokens: int,
) -> tuple[Tensor, Tensor, Tensor]:
    mlp = layer.mlp
    inputs: list[Tensor] = []
    outputs: list[Tensor] = []
    layer_outputs: list[Tensor] = []

    def pre(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
        inputs.append(args[0].detach())

    def post(_module: nn.Module, _args: tuple[Tensor, ...], output: Tensor) -> None:
        outputs.append(output.detach())

    def layer_post(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        layer_outputs.append(hidden.detach())

    handles = [
        mlp.register_forward_pre_hook(pre),
        mlp.register_forward_hook(post),
        layer.register_forward_hook(layer_post),
    ]
    try:
        model(**encoded)
    finally:
        for handle in handles:
            handle.remove()

    x = inputs[0].reshape(-1, inputs[0].shape[-1]).cpu()
    m = outputs[0].reshape(-1, outputs[0].shape[-1]).cpu()
    h = layer_outputs[0].reshape(-1, layer_outputs[0].shape[-1]).cpu()
    mask = encoded.get("attention_mask")
    if mask is not None:
        valid = mask.reshape(-1).to(torch.bool).cpu()
        x, m, h = x[valid], m[valid], h[valid]
    return (
        x[:max_tokens].float(),
        (h - m)[:max_tokens].float(),
        h[:max_tokens].float(),
    )


def kmeans(
    x: Tensor,
    clusters: int,
    *,
    iterations: int,
    seed: int,
) -> tuple[Tensor, Tensor]:
    clusters = min(clusters, x.shape[0])
    generator = torch.Generator().manual_seed(seed)
    normalized = torch.nn.functional.normalize(x, dim=-1)
    first = torch.randint(x.shape[0], (1,), generator=generator).item()
    chosen = [first]
    best_distance = 1.0 - normalized @ normalized[first]
    for _ in range(1, clusters):
        index = torch.argmax(best_distance).item()
        chosen.append(index)
        distance = 1.0 - normalized @ normalized[index]
        best_distance = torch.minimum(best_distance, distance)

    centroids = normalized[chosen].clone()
    assignment = torch.zeros(x.shape[0], dtype=torch.long)
    for _ in range(iterations):
        similarity = normalized @ centroids.T
        assignment = similarity.argmax(dim=-1)
        updated = []
        for cluster in range(clusters):
            members = normalized[assignment == cluster]
            if members.numel() == 0:
                updated.append(centroids[cluster])
            else:
                updated.append(
                    torch.nn.functional.normalize(
                        members.mean(dim=0),
                        dim=0,
                    )
                )
        centroids = torch.stack(updated)
    return centroids, assignment


class LocalOperator(nn.Module):
    def __init__(self, hidden: int, rank: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(hidden, rank, bias=False)
        self.out_proj = nn.Linear(rank, hidden, bias=True)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.out_proj(torch.nn.functional.silu(self.in_proj(hidden)))


def train_local_operators(
    x: Tensor,
    base: Tensor,
    dense: Tensor,
    assignment: Tensor,
    final_norm: nn.Module,
    *,
    clusters: int,
    rank: int,
    steps: int,
    lr: float,
    seed: int,
) -> nn.ModuleList:
    torch.manual_seed(seed)
    operators = nn.ModuleList(
        [LocalOperator(x.shape[-1], rank) for _ in range(clusters)]
    )
    norm = final_norm.cpu()
    with torch.no_grad():
        target = norm(dense).detach()

    for cluster, operator in enumerate(operators):
        members = torch.where(assignment == cluster)[0]
        if members.numel() == 0:
            continue
        optimizer = torch.optim.AdamW(
            operator.parameters(),
            lr=lr,
            weight_decay=1e-4,
        )
        for _ in range(steps):
            candidate = norm(base[members] + operator(x[members]))
            loss = torch.nn.functional.mse_loss(
                candidate,
                target[members],
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return operators.eval()


@torch.inference_mode()
def route(
    x: Tensor,
    centroids: Tensor,
) -> Tensor:
    normalized = torch.nn.functional.normalize(x, dim=-1)
    return (normalized @ centroids.T).argmax(dim=-1)


@torch.inference_mode()
def evaluate(
    x: Tensor,
    base: Tensor,
    dense: Tensor,
    centroids: Tensor,
    operators: nn.ModuleList,
    final_norm: nn.Module,
    lm_head: nn.Linear,
) -> dict[str, float]:
    assignment = route(x, centroids)
    predicted_mlp = torch.zeros_like(x)
    for cluster, operator in enumerate(operators):
        members = torch.where(assignment == cluster)[0]
        if members.numel() == 0:
            continue
        predicted_mlp[members] = operator(x[members])

    norm = final_norm.cpu()
    head = lm_head.cpu()
    dense_norm = norm(dense)
    candidate_norm = norm(base + predicted_mlp)
    dense_logits = head(dense_norm)
    candidate_logits = head(candidate_norm)

    ref_logp = torch.log_softmax(dense_logits, dim=-1)
    cand_logp = torch.log_softmax(candidate_logits, dim=-1)
    probability = ref_logp.exp()
    kl = (probability * (ref_logp - cand_logp)).sum(dim=-1)
    agree = dense_logits.argmax(dim=-1) == candidate_logits.argmax(dim=-1)
    normalized_error = (
        (dense_norm - candidate_norm).norm(dim=-1)
        / dense_norm.norm(dim=-1).clamp_min(1e-12)
    )
    return {
        "mean_kl_nats": float(kl.mean().item()),
        "p95_kl_nats": float(torch.quantile(kl, 0.95).item()),
        "max_kl_nats": float(kl.max().item()),
        "top1_agreement": float(agree.float().mean().item()),
        "mean_normalized_hidden_error": float(normalized_error.mean().item()),
    }


def metadata_fraction(
    centroids: Tensor,
    operators: nn.ModuleList,
    mlp: nn.Module,
) -> float:
    hot = centroids.numel()
    hot += sum(p.numel() for operator in operators for p in operator.parameters())
    cold = sum(p.numel() for p in mlp.parameters())
    return hot / cold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--text-file")
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--clusters", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--ranks", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--kmeans-iterations", type=int, default=8)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=601)
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
    split = len(texts) // 2
    train_texts = texts[:split]
    test_texts = texts[split:]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_encoded = encode(tokenizer, train_texts, device, args.max_length)
    test_encoded = encode(tokenizer, test_texts, device, args.max_length)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    layers = resolve_layers(model)
    layer = layers[-1]
    mlp = layer.mlp
    final_norm = model.model.norm
    lm_head = model.lm_head

    train_x, train_base, train_dense = capture_last_layer(
        model,
        layer,
        train_encoded,
        max_tokens=args.max_tokens,
    )
    test_x, test_base, test_dense = capture_last_layer(
        model,
        layer,
        test_encoded,
        max_tokens=args.max_tokens,
    )
    train_x = train_x.detach().clone()
    train_base = train_base.detach().clone()
    train_dense = train_dense.detach().clone()
    test_x = test_x.detach().clone()
    test_base = test_base.detach().clone()
    test_dense = test_dense.detach().clone()

    rows = []
    for clusters in args.clusters:
        centroids, assignment = kmeans(
            train_x,
            clusters,
            iterations=args.kmeans_iterations,
            seed=args.seed + clusters,
        )
        actual_clusters = centroids.shape[0]
        for rank in args.ranks:
            operators = train_local_operators(
                train_x,
                train_base,
                train_dense,
                assignment,
                final_norm,
                clusters=actual_clusters,
                rank=rank,
                steps=args.steps,
                lr=args.lr,
                seed=args.seed + clusters * 100 + rank,
            )
            fraction = metadata_fraction(centroids, operators, mlp)
            rows.append(
                {
                    "clusters": actual_clusters,
                    "rank": rank,
                    "resident_metadata_fraction": fraction,
                    "analytic_cold_byte_reduction_fraction": 1.0 - fraction,
                    **evaluate(
                        test_x,
                        test_base,
                        test_dense,
                        centroids,
                        operators,
                        final_norm,
                        lm_head,
                    ),
                }
            )

    payload = {
        "experiment": "hf_piecewise_hot_surrogate_v0",
        "model": args.model,
        "layer": len(layers) - 1,
        "train_tokens": int(train_x.shape[0]),
        "test_tokens": int(test_x.shape[0]),
        "rows": rows,
        "claim_boundary": (
            "Hidden states are clustered only on the training prompt split. Each "
            "predictive-state cluster gets a tiny local nonlinear operator that "
            "replaces the final dense MLP. Held-out routing is nearest-centroid "
            "and quality is measured in full-vocabulary logit space. This is a "
            "workload-specific predictive operator cache, not yet a general model "
            "replacement or measured hardware speedup."
        ),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        Path(args.json_out).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
