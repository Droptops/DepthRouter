# ruff: noqa: I001
from __future__ import annotations

import argparse
import json
from pathlib import Path
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
    "How can a repeated workload create only a few recurring computational states?",
    "What is a causal state in a predictive process?",
    "Why can a codebook be cheaper than predicting thousands of independent neuron decisions?",
    "Describe a semantic page table whose address is a small discrete state code.",
    "How could a Markov state transition prefetch the next neural working set?",
    "Why might demand masks cluster even when hidden vectors appear continuous?",
    "What does it mean to compile recurring computation into a physical working set?",
    "How can a tiny classifier select among precompiled exact neuron subsets?",
    "Why should the address space be smaller than the payload space?",
    "What would make a computational codebook useful for hardware caching?",
    "Explain why recurring futures matter more than surface-form similarity.",
    "How can a latent trajectory state determine which exact model weights are needed?",
    "What is the difference between predicting neurons independently and predicting a demand template?",
    "Why might a codebook solve an addressability bottleneck?",
    "How would you test whether active neural computation has low discrete entropy?",
    "Explain how a finite-state cache can represent a much larger dense model.",
    "What would falsify the hypothesis that user workloads revisit recurring computational states?",
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
    if len(rows) < 12:
        raise ValueError("need at least twelve texts")
    return rows


def resolve_layers(model: nn.Module) -> list[nn.Module]:
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None) if inner is not None else None
    if layers is None:
        raise ValueError("could not locate decoder layers")
    return list(layers)


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
def capture_all_token_state(
    model: nn.Module,
    mlp: nn.Module,
    encoded: dict[str, Tensor],
    *,
    max_tokens: int,
) -> tuple[Tensor, Tensor]:
    hidden_sink: list[Tensor] = []
    activation_sink: list[Tensor] = []

    def pre(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
        hidden_sink.append(args[0].detach())

    def down_pre(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
        activation_sink.append(args[0].detach())

    handles = [
        mlp.register_forward_pre_hook(pre),
        mlp.down_proj.register_forward_pre_hook(down_pre),
    ]
    try:
        model(**encoded)
    finally:
        for handle in handles:
            handle.remove()

    hidden = hidden_sink[0].reshape(-1, hidden_sink[0].shape[-1]).cpu()
    activation = activation_sink[0].reshape(-1, activation_sink[0].shape[-1]).cpu()
    mask = encoded.get("attention_mask")
    if mask is not None:
        valid = mask.reshape(-1).to(torch.bool).cpu()
        hidden = hidden[valid]
        activation = activation[valid]
    return hidden[:max_tokens].float(), activation[:max_tokens].float()


@torch.inference_mode()
def capture_last_token(
    model: nn.Module,
    mlp: nn.Module,
    encoded: dict[str, Tensor],
) -> tuple[Tensor, Tensor, Tensor]:
    hidden_sink: list[Tensor] = []
    activation_sink: list[Tensor] = []

    def pre(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
        hidden_sink.append(args[0][:, -1, :].detach())

    def down_pre(_module: nn.Module, args: tuple[Tensor, ...]) -> None:
        activation_sink.append(args[0][:, -1, :].detach())

    handles = [
        mlp.register_forward_pre_hook(pre),
        mlp.down_proj.register_forward_pre_hook(down_pre),
    ]
    try:
        logits = model(**encoded).logits[:, -1, :].detach().float().cpu()
    finally:
        for handle in handles:
            handle.remove()

    return (
        hidden_sink[0].float().cpu(),
        activation_sink[0].float().cpu(),
        logits,
    )


def demand_scores(activation: Tensor, mlp: nn.Module) -> Tensor:
    norm = mlp.down_proj.weight.detach().float().cpu().norm(dim=0)
    score = activation.abs() * norm[None, :]
    return score / score.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def spherical_kmeans(
    score: Tensor,
    clusters: int,
    *,
    iterations: int,
    seed: int,
) -> tuple[Tensor, Tensor]:
    clusters = min(clusters, score.shape[0])
    normalized = torch.nn.functional.normalize(score, dim=-1)
    generator = torch.Generator().manual_seed(seed)
    first = torch.randint(score.shape[0], (1,), generator=generator).item()
    chosen = [first]
    best_distance = 1.0 - normalized @ normalized[first]
    for _ in range(1, clusters):
        index = torch.argmax(best_distance).item()
        chosen.append(index)
        distance = 1.0 - normalized @ normalized[index]
        best_distance = torch.minimum(best_distance, distance)

    centroids = normalized[chosen].clone()
    assignment = torch.zeros(score.shape[0], dtype=torch.long)
    for _ in range(iterations):
        assignment = (normalized @ centroids.T).argmax(dim=-1)
        updated = []
        for cluster in range(clusters):
            members = normalized[assignment == cluster]
            if members.numel() == 0:
                updated.append(centroids[cluster])
            else:
                updated.append(
                    torch.nn.functional.normalize(members.mean(dim=0), dim=0)
                )
        centroids = torch.stack(updated)
    return centroids, assignment


def compile_codebook(
    score: Tensor,
    assignment: Tensor,
    clusters: int,
    *,
    fraction: float,
) -> Tensor:
    width = score.shape[-1]
    k = max(1, min(width, round(width * fraction)))
    rows = []
    global_mean = score.mean(dim=0)
    for cluster in range(clusters):
        members = score[assignment == cluster]
        mean = members.mean(dim=0) if members.numel() else global_mean
        rows.append(torch.topk(mean, k=k, dim=-1, sorted=False).indices)
    return torch.stack(rows)


class StateClassifier(nn.Module):
    def __init__(self, hidden: int, rank: int, clusters: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(hidden, rank, bias=False)
        self.out_proj = nn.Linear(rank, clusters, bias=True)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.out_proj(torch.tanh(self.in_proj(hidden)))


def train_classifier(
    hidden: Tensor,
    labels: Tensor,
    *,
    rank: int,
    clusters: int,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> StateClassifier:
    torch.manual_seed(seed)
    head = StateClassifier(hidden.shape[-1], rank, clusters)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed + 1)
    for _ in range(steps):
        idx = torch.randint(
            hidden.shape[0],
            (min(batch_size, hidden.shape[0]),),
            generator=generator,
        )
        loss = torch.nn.functional.cross_entropy(head(hidden[idx]), labels[idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return head.eval()


def sparse_exact_output(
    hidden: Tensor,
    mlp: nn.Module,
    state: Tensor,
    codebook: Tensor,
) -> Tensor:
    fn = getattr(mlp, "act_fn", torch.nn.functional.silu)
    rows = []
    for row in range(hidden.shape[0]):
        idx = codebook[state[row]].to(hidden.device)
        x = hidden[row]
        gate = torch.nn.functional.linear(
            x,
            mlp.gate_proj.weight[idx],
            mlp.gate_proj.bias[idx] if mlp.gate_proj.bias is not None else None,
        )
        up = torch.nn.functional.linear(
            x,
            mlp.up_proj.weight[idx],
            mlp.up_proj.bias[idx] if mlp.up_proj.bias is not None else None,
        )
        activation = fn(gate) * up
        rows.append(
            torch.nn.functional.linear(
                activation,
                mlp.down_proj.weight[:, idx],
                mlp.down_proj.bias,
            )
        )
    return torch.stack(rows)


@torch.inference_mode()
def run_codebook(
    model: nn.Module,
    mlp: nn.Module,
    encoded: dict[str, Tensor],
    classifier: StateClassifier,
    codebook: Tensor,
) -> tuple[Tensor, Tensor]:
    state_sink: list[Tensor] = []

    def hook(_module: nn.Module, args: tuple[Tensor, ...], output: Tensor) -> Tensor:
        hidden_device = args[0][:, -1, :]
        hidden = hidden_device.detach().float().cpu()
        state = classifier(hidden).argmax(dim=-1)
        state_sink.append(state)
        sparse = sparse_exact_output(
            hidden_device,
            mlp,
            state,
            codebook,
        )
        replaced = output.clone()
        replaced[:, -1, :] = sparse.to(replaced.dtype)
        return replaced

    handle = mlp.register_forward_hook(hook)
    try:
        logits = model(**encoded).logits[:, -1, :].detach().float().cpu()
    finally:
        handle.remove()
    return logits, torch.cat(state_sink)


@torch.inference_mode()
def run_oracle_codebook(
    model: nn.Module,
    mlp: nn.Module,
    encoded: dict[str, Tensor],
    centroids: Tensor,
    codebook: Tensor,
) -> tuple[Tensor, Tensor]:
    state_sink: list[Tensor] = []

    def hook(_module: nn.Module, args: tuple[Tensor, ...], output: Tensor) -> Tensor:
        hidden_device = args[0][:, -1, :]
        # Diagnostic only: compute exact current demand to route to its closest
        # recurring demand template.
        gate = mlp.gate_proj(hidden_device)
        up = mlp.up_proj(hidden_device)
        fn = getattr(mlp, "act_fn", torch.nn.functional.silu)
        activation = fn(gate) * up
        score = demand_scores(activation.detach().float().cpu(), mlp)
        state = (
            torch.nn.functional.normalize(score, dim=-1) @ centroids.T
        ).argmax(dim=-1)
        state_sink.append(state)
        sparse = sparse_exact_output(hidden_device, mlp, state, codebook)
        replaced = output.clone()
        replaced[:, -1, :] = sparse.to(replaced.dtype)
        return replaced

    handle = mlp.register_forward_hook(hook)
    try:
        logits = model(**encoded).logits[:, -1, :].detach().float().cpu()
    finally:
        handle.remove()
    return logits, torch.cat(state_sink)


def quality(reference: Tensor, candidate: Tensor) -> dict[str, float]:
    ref_logp = torch.log_softmax(reference, dim=-1)
    cand_logp = torch.log_softmax(candidate, dim=-1)
    probability = ref_logp.exp()
    kl = (probability * (ref_logp - cand_logp)).sum(dim=-1)
    agreement = reference.argmax(dim=-1) == candidate.argmax(dim=-1)
    return {
        "mean_kl_nats": float(kl.mean().item()),
        "p95_kl_nats": float(torch.quantile(kl, 0.95).item()),
        "max_kl_nats": float(kl.max().item()),
        "top1_agreement": float(agreement.float().mean().item()),
    }


def metadata_fraction(
    classifier: StateClassifier,
    codebook: Tensor,
    mlp: nn.Module,
) -> float:
    classifier_bytes = sum(p.numel() for p in classifier.parameters()) * 2
    # uint16 is enough for Qwen's 4864 intermediate neurons.
    codebook_bytes = codebook.numel() * 2
    cold_bytes = sum(p.numel() for p in mlp.parameters()) * 2
    return (classifier_bytes + codebook_bytes) / cold_bytes


def active_set_recall(
    state: Tensor,
    codebook: Tensor,
    exact_score: Tensor,
    *,
    fraction: float,
) -> float:
    width = exact_score.shape[-1]
    k = max(1, min(width, round(width * fraction)))
    target = torch.topk(exact_score, k=k, dim=-1, sorted=False).indices
    recalls = []
    for row in range(target.shape[0]):
        a = set(target[row].tolist())
        b = set(codebook[state[row]].tolist())
        recalls.append(len(a & b) / k)
    return sum(recalls) / len(recalls)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--text-file")
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--max-train-tokens", type=int, default=768)
    parser.add_argument("--clusters", nargs="+", type=int, default=[4, 8, 16, 32])
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.10, 0.15, 0.20])
    parser.add_argument("--classifier-rank", type=int, default=8)
    parser.add_argument("--kmeans-iterations", type=int, default=8)
    parser.add_argument("--steps", type=int, default=350)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=701)
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
    if args.layer < 0 or args.layer >= len(layers):
        raise ValueError("layer index out of range")
    mlp = resolve_mlp(layers[args.layer])

    train_hidden, train_activation = capture_all_token_state(
        model,
        mlp,
        train_encoded,
        max_tokens=args.max_train_tokens,
    )
    train_hidden = train_hidden.detach().clone()
    train_score = demand_scores(train_activation.detach().clone(), mlp)

    test_hidden, test_activation, baseline = capture_last_token(
        model,
        mlp,
        test_encoded,
    )
    test_hidden = test_hidden.detach().clone()
    test_score = demand_scores(test_activation.detach().clone(), mlp)

    rows = []
    for clusters in args.clusters:
        centroids, labels = spherical_kmeans(
            train_score,
            clusters,
            iterations=args.kmeans_iterations,
            seed=args.seed + clusters,
        )
        actual_clusters = centroids.shape[0]
        classifier = train_classifier(
            train_hidden,
            labels,
            rank=args.classifier_rank,
            clusters=actual_clusters,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed + 100 + clusters,
        )
        predicted_state = classifier(test_hidden).argmax(dim=-1)
        oracle_state = (
            torch.nn.functional.normalize(test_score, dim=-1) @ centroids.T
        ).argmax(dim=-1)
        state_accuracy = float((predicted_state == oracle_state).float().mean().item())

        for fraction in args.fractions:
            codebook = compile_codebook(
                train_score,
                labels,
                actual_clusters,
                fraction=fraction,
            )
            predicted_logits, predicted_runtime_state = run_codebook(
                model,
                mlp,
                test_encoded,
                classifier,
                codebook,
            )
            oracle_logits, oracle_runtime_state = run_oracle_codebook(
                model,
                mlp,
                test_encoded,
                centroids,
                codebook,
            )

            meta = metadata_fraction(classifier, codebook, mlp)
            rows.append(
                {
                    "clusters": actual_clusters,
                    "classifier_rank": args.classifier_rank,
                    "selected_payload_fraction": fraction,
                    "resident_address_plus_codebook_fraction": meta,
                    "total_metadata_plus_payload_fraction": meta + fraction,
                    "state_prediction_accuracy": state_accuracy,
                    "predicted_active_set_recall": active_set_recall(
                        predicted_runtime_state,
                        codebook,
                        test_score,
                        fraction=fraction,
                    ),
                    "oracle_active_set_recall": active_set_recall(
                        oracle_runtime_state,
                        codebook,
                        test_score,
                        fraction=fraction,
                    ),
                    "predicted_route": quality(baseline, predicted_logits),
                    "oracle_state_route": quality(baseline, oracle_logits),
                }
            )

    payload = {
        "experiment": "hf_causal_state_codebook_v0",
        "model": args.model,
        "layer": args.layer,
        "train_tokens": int(train_hidden.shape[0]),
        "test_prompts": len(test_texts),
        "rows": rows,
        "claim_boundary": (
            "Demand templates are learned only from training prompts. The "
            "deployable route predicts a small discrete template ID from the "
            "incoming hidden state and then evaluates exact weights only for the "
            "precompiled neuron subset. The oracle-state route uses current dense "
            "activation only to test whether the codebook itself is expressive "
            "enough. Byte fractions are analytic until a selective kernel is profiled."
        ),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        Path(args.json_out).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
