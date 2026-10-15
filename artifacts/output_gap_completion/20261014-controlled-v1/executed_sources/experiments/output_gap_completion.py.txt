"""One preregistered Qwen correction-objective comparison, with streamed logits.

Run once in a new output directory. Held-out model access requires an immutable
selection file and unchanged correction checkpoint. No held-out tuning/resume.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from fused_sparse_swiglu import RunLog, encode_prompts, replaced_mlp, write_csv, write_json
from torch import nn

from depth_router.fused_swiglu import PrepackedSwiGLU
from depth_router.output_gap import (
    CorrectedSwiGLU,
    cheap_candidates,
    contrast_rows,
    train_pair,
)
from depth_router.output_gap_data import (
    load_historical_prompts,
    make_output_gap_manifest,
    validate_output_gap_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("calibration", "development", "heldout_matched", "heldout_ood")


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def state_hash(module: nn.Module) -> str:
    text = json.dumps({k: tensor_hash(v) for k, v in module.state_dict().items()}, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


def command_output(args: list[str]) -> str:
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=20, check=False)
        return (p.stdout + p.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return str(exc)


def ci(values: list[float] | np.ndarray, seed: int, draws: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.concatenate(
        [
            array[rng.integers(0, len(array), (min(250, draws - i), len(array)))].mean(1)
            for i in range(0, draws, 250)
        ]
    )
    return np.quantile(means, [0.025, 0.975]).tolist()


@torch.inference_mode()
def forward_capture(
    model: nn.Module,
    layer: nn.Module,
    module: nn.Module,
    encoded: dict[str, torch.Tensor],
    *,
    activation: bool = False,
):
    captured: dict[str, torch.Tensor] = {}

    def input_hook(_m, args):
        captured["state"] = args[0][:, -1].detach().clone()

    def base_hook(_m, args):
        captured["base"] = args[0][:, -1].detach().clone()

    def output_hook(_m, _args, output):
        captured["output"] = output[:, -1].detach().clone()

    def activation_hook(_m, args):
        captured["activation"] = args[0][:, -1].detach().clone()

    def correction_hook(_m, _args, output):
        captured["correction"] = output[:, -1].detach().clone()

    handles = [
        module.register_forward_pre_hook(input_hook),
        module.register_forward_hook(output_hook),
        layer.post_attention_layernorm.register_forward_pre_hook(base_hook),
    ]
    if activation:
        handles.append(module.down_proj.register_forward_pre_hook(activation_hook))
    if isinstance(module, CorrectedSwiGLU):
        handles.append(module.residual.register_forward_hook(correction_hook))
    try:
        with replaced_mlp(layer, module):
            logits = model(**encoded, use_cache=False, logits_to_keep=1).logits[:, -1].float()
            assert bool(torch.isfinite(logits).all()), "nonfinite model logits"
    finally:
        for h in handles:
            h.remove()
    return logits, captured


def encoded_batches(tokenizer, examples, config):
    size = config["quality_batch_size"]
    for start in range(0, len(examples), size):
        batch = examples[start : start + size]
        yield (
            batch,
            encode_prompts(
                tokenizer,
                [r["prompt"] for r in batch],
                torch.device("cuda"),
                config["max_prompt_tokens"],
            ),
        )


@torch.inference_mode()
def calibrate_mask(model, tokenizer, layer, dense, examples, config, log):
    """Same output-sensitive first-order score as the previous experiment.

    Retain only small captures and top-two contrasts; full logits are discarded
    after each batch. Quartile threshold is computed over all calibration rows.
    """
    effects, margins = [], []
    norm = model.model.norm
    for batch, encoded in encoded_batches(tokenizer, examples, config):
        logits, cap = forward_capture(model, layer, dense, encoded, activation=True)
        top = logits.topk(2, dim=-1)
        direction = (
            model.lm_head.weight[top.indices[:, 0]].float()
            - model.lm_head.weight[top.indices[:, 1]].float()
        ) * norm.weight.float()
        h = (cap["base"] + cap["output"]).to(torch.bfloat16).float()
        rho = (h.square().mean(-1) + norm.variance_epsilon).sqrt()
        gradient = direction / rho[:, None] - (
            (direction * h).sum(-1)[:, None] * h / (h.shape[-1] * rho[:, None].pow(3))
        )
        effects.append(
            (cap["activation"].float() * (gradient @ dense.down_proj.weight.float())).cpu()
        )
        margins.append((top.values[:, 0] - top.values[:, 1]).cpu())
        log.write("calibration_mask_batch", last_id=batch[-1]["id"])
    effects, margins = torch.cat(effects), torch.cat(margins)
    threshold = margins.quantile(0.25)
    weights = torch.where(margins <= threshold, config["low_margin_weight"], 1.0)
    score = (effects.abs() * weights[:, None]).mean(0)
    count = round(dense.gate_proj.out_features * config["retention_fraction"])
    selection = score.topk(count).indices.sort().values
    return selection, {
        "threshold": float(threshold),
        "score_sha256": tensor_hash(score),
        "method": "calibration output-sensitive absolute neuron effects, low-margin weight 4",
    }


@torch.inference_mode()
def calibration_targets(model, tokenizer, layer, dense, sparse, examples, config, log):
    pieces = {key: [] for key in ("state", "target", "contrast", "base", "sparse", "candidates")}
    max_identity_error = 0.0
    for batch, encoded in encoded_batches(tokenizer, examples, config):
        _, full = forward_capture(model, layer, dense, encoded)
        cheap, cap = forward_capture(model, layer, sparse, encoded)
        assert torch.equal(full["state"], cap["state"])
        assert torch.equal(full["base"], cap["base"])
        target = full["output"].float() - cap["output"].float()
        ids = cheap_candidates(cheap)
        contrast = contrast_rows(model.lm_head.weight, model.model.norm.weight, ids)
        a = model.lm_head.weight[ids].float() * model.model.norm.weight.float()
        direct = torch.einsum("bkd,bd->bk", a, target)
        projected = torch.einsum("bkd,bd->bk", contrast, target)
        torch.testing.assert_close(projected, direct[:, 1:] - direct[:, :1], rtol=1e-4, atol=1e-4)
        max_identity_error = max(
            max_identity_error, float((projected - (direct[:, 1:] - direct[:, :1])).abs().max())
        )
        for key, value in {
            "state": cap["state"],
            "target": target,
            "contrast": contrast,
            "base": cap["base"],
            "sparse": cap["output"],
            "candidates": ids,
        }.items():
            pieces[key].append(value.cpu())
        log.write("calibration_labels_batch", last_id=batch[-1]["id"])
    return {k: torch.cat(v) for k, v in pieces.items()}, {
        "gap_identity_max_abs_error": max_identity_error
    }


@torch.inference_mode()
def evaluate(model, tokenizer, layer, arms, examples, config, log):
    rows = []
    for batch, encoded in encoded_batches(tokenizer, examples, config):
        reference, dense_cap = forward_capture(model, layer, arms["dense"], encoded)
        cheap, sparse_cap = forward_capture(model, layer, arms["sparse75"], encoded)
        ids = cheap_candidates(cheap)
        contrast = contrast_rows(model.lm_head.weight, model.model.norm.weight, ids)
        target = dense_cap["output"].float() - sparse_cap["output"].float()
        winner = reference.argmax(-1)
        top2 = reference.topk(2, dim=-1).values
        logp = F.log_softmax(reference, -1)
        prob = logp.exp()
        for name, module in arms.items():
            if name == "dense":
                candidate, cap = reference, dense_cap
            elif name == "sparse75":
                candidate, cap = cheap, sparse_cap
            else:
                candidate, cap = forward_capture(model, layer, module, encoded)
            assert torch.equal(cap["state"], dense_cap["state"]), "upstream model changed"
            assert torch.equal(cap["base"], dense_cap["base"]), "upstream residual changed"
            if name in ("dense", "full_width_control"):
                error = dense_cap["output"].float() - cap["output"].float()
            else:
                error = cap.get("correction", torch.zeros_like(target)).float() - target
            hidden_error = error.square().mean(-1)
            gap_error = torch.einsum("bkd,bd->bk", contrast, error).square().mean(-1)
            divergence = (prob * (logp - F.log_softmax(candidate, -1))).sum(-1).clamp_min(0)
            candidate_winner = candidate.argmax(-1)
            ties = (candidate == candidate.max(-1, keepdim=True).values).sum(-1)
            actual_error = (dense_cap["output"].float() - cap["output"].float()).square().mean(-1)
            for i, example in enumerate(batch):
                rows.append(
                    {
                        "method": name,
                        "id": example["id"],
                        "split": example["split"],
                        "task": example["task"],
                        "top1_agreement": int(winner[i] == candidate_winner[i]),
                        "kl": float(divergence[i]),
                        "hidden_residual_mse": float(hidden_error[i]),
                        "output_gap_mse": float(gap_error[i]),
                        "actual_mlp_output_mse": float(actual_error[i]),
                        "reference_top1_id": int(winner[i]),
                        "candidate_top1_id": int(candidate_winner[i]),
                        "reference_margin": float(top2[i, 0] - top2[i, 1]),
                        "dense_winner_outside_cheap8": bool(not (ids[i] == winner[i]).any()),
                        "candidate_top_tie_count": int(ties[i]),
                        "max_abs_logit_error": float((candidate[i] - reference[i]).abs().max()),
                    }
                )
        log.write("evaluation_batch", split=batch[0]["split"], last_id=batch[-1]["id"])
    return rows


def summarize(rows, config):
    summaries = []
    for split in sorted({r["split"] for r in rows}):
        reference = [r for r in rows if r["split"] == split and r["method"] == "dense"]
        cuts = np.quantile([r["reference_margin"] for r in reference], [0.25, 0.5, 0.75])
        for name in sorted({r["method"] for r in rows}):
            subset = [r for r in rows if r["split"] == split and r["method"] == name]
            if not subset:
                continue
            agreement = np.array([r["top1_agreement"] for r in subset])
            kl = np.array([r["kl"] for r in subset])
            summary = {
                "method": name,
                "split": split,
                "n": len(subset),
                "top1_agreement": float(agreement.mean()),
                "mean_kl": float(kl.mean()),
                "top1_ci95": ci(agreement, config["seed"], config["bootstrap_draws"]),
                "kl_ci95": ci(kl, config["seed"], config["bootstrap_draws"]),
                "passes_quality_gate": bool(
                    agreement.mean() >= config["quality_gate_top1"]
                    and kl.mean() < config["quality_gate_kl"]
                ),
                "hidden_residual_mse": float(np.mean([r["hidden_residual_mse"] for r in subset])),
                "output_gap_mse": float(np.mean([r["output_gap_mse"] for r in subset])),
                "excluded_winners": sum(r["dense_winner_outside_cheap8"] for r in subset),
                "excluded_winner_failures": sum(
                    r["dense_winner_outside_cheap8"] and not r["top1_agreement"] for r in subset
                ),
                "top_ties": sum(r["candidate_top_tie_count"] > 1 for r in subset),
                "margin_quartile_cuts": cuts.tolist(),
                "margin_quartiles": [],
            }
            for quartile in range(4):
                group = [
                    r
                    for r in subset
                    if np.searchsorted(cuts, r["reference_margin"], side="left") == quartile
                ]
                summary["margin_quartiles"].append(
                    {
                        "quartile": quartile + 1,
                        "n": len(group),
                        "top1_agreement": float(np.mean([r["top1_agreement"] for r in group]))
                        if group
                        else None,
                    }
                )
            summaries.append(summary)
            summary["by_task"] = {
                task: {
                    "n": sum(r["task"] == task for r in subset),
                    "top1_agreement": float(
                        np.mean([r["top1_agreement"] for r in subset if r["task"] == task])
                    ),
                    "mean_kl": float(np.mean([r["kl"] for r in subset if r["task"] == task])),
                }
                for task in sorted({r["task"] for r in subset})
            }
    return summaries


def paired_comparisons(rows, config):
    result = []
    indexed = {(r["method"], r["id"]): r for r in rows}
    for split in sorted({r["split"] for r in rows}):
        ids = [r["id"] for r in rows if r["split"] == split and r["method"] == "dense"]
        for rank in config["ranks"]:
            diffs = []
            for seed in config["seeds"]:
                difference = np.array(
                    [
                        indexed[(f"gap_r{rank}_s{seed}", i)]["top1_agreement"]
                        - indexed[(f"hidden_r{rank}_s{seed}", i)]["top1_agreement"]
                        for i in ids
                    ]
                )
                diffs.append(difference)
                result.append(
                    {
                        "split": split,
                        "rank": rank,
                        "seed": seed,
                        "gap_minus_hidden_agreement": float(difference.mean()),
                        "ci95": ci(difference, config["seed"], config["bootstrap_draws"]),
                        "gap_wins": int((difference > 0).sum()),
                        "hidden_wins": int((difference < 0).sum()),
                    }
                )
            averaged = np.mean(diffs, axis=0)
            interval = ci(averaged, config["seed"], config["bootstrap_draws"])
            consistent = all(d.mean() > 0 for d in diffs)
            result.append(
                {
                    "split": split,
                    "rank": rank,
                    "seed": "paired_seed_mean",
                    "gap_minus_hidden_agreement": float(averaged.mean()),
                    "ci95": interval,
                    "direction_positive_all_seeds": consistent,
                    "established": bool(consistent and interval[0] > 0),
                    "resampling_unit": "prompt, with all seeds kept together",
                }
            )
    return result


def select_development_arm(development):
    candidates = [
        r for r in development if r["method"].startswith("gap_") and r["split"] == "development"
    ]
    if not candidates:
        raise ValueError("no development output-gap candidates")

    def key(row):
        _, rank, seed = row["method"].split("_")
        return -row["top1_agreement"], row["mean_kl"], int(rank[1:]), int(seed[1:])

    return min(candidates, key=key)


def objective_improvement_established(comparison, chosen_rank):
    primary = [
        r
        for r in comparison
        if r["seed"] == "paired_seed_mean"
        and r["rank"] == chosen_rank
        and r["split"].startswith("heldout")
    ]
    return (
        len(primary) == 2
        and {r["split"] for r in primary} == {"heldout_matched", "heldout_ood"}
        and all(r["established"] for r in primary)
    )


@torch.inference_mode()
def bound_audit(model, dense, selection, calibration, corrections):
    omitted = torch.ones(dense.gate_proj.out_features, device="cuda", dtype=torch.bool)
    omitted[selection.to("cuda")] = False
    beta = (
        dense.down_proj.weight[:, omitted].double().norm(dim=0)
        * dense.gate_proj.weight[omitted].double().norm(dim=1)
        * dense.up_proj.weight[omitted].double().norm(dim=1)
    ).sum()
    x = calibration["state"].to("cuda")
    # If ||u_S+r_hat|| <= R, even the maximum possible score spread
    # 2 max_i||A_i|| ||u_S+r_hat|| cannot exceed 2 max_i||A_i|| R.
    # This proves non-acceptance without performing another vocabulary scan.
    bound_rows = []
    for name, correction in corrections.items():
        estimate = correction(x).double()
        radius = beta * x.double().square().sum(-1) + estimate.norm(dim=-1)
        approximate = (
            calibration["base"].to("cuda").double()
            + calibration["sparse"].to("cuda").double()
            + estimate
        )
        ratio = radius / approximate.norm(dim=-1).clamp_min(1e-30)
        bound_rows.append(
            {
                "method": name,
                "calibration_examples": len(x),
                "certain_nonacceptance_count": int((ratio >= 1).sum()),
                "min_radius_over_approximate_state_norm": float(ratio.min()),
                "median_radius_over_approximate_state_norm": float(ratio.median()),
                "certified_examples": 0 if bool((ratio >= 1).all()) else None,
            }
        )
    return {
        "beta_omitted": float(beta),
        "rows": bound_rows,
        "real_arithmetic_bound": "R=beta*||x||^2+||correction(x)||; epsilon=max_i||E_i*gamma||*R",
        "nonacceptance_test": "R >= ||u_S+correction|| implies even a score-spread upper bound cannot pass",
        "interpretation": "diagnostic bound looseness on BF16 captures; exact real sparse u_S would also need rounding accounting",
        "bf16_certificate": "unestablished: no conservative end-to-end floating-point error bound",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/output_gap_completion.json")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text())
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out / "preregistration.json").exists():
        raise SystemExit("Refusing to overwrite/reopen an existing run. Choose a new run ID.")
    log = RunLog(out / "research_log.jsonl")
    sources = [
        Path(__file__),
        ROOT / "src/depth_router/output_gap.py",
        ROOT / "src/depth_router/output_gap_data.py",
        ROOT / "src/depth_router/fused_swiglu.py",
        ROOT / "experiments/fused_sparse_swiglu.py",
        ROOT / "experiments/output_gap_timing.py",
        ROOT / args.config,
    ]
    preregistration = {
        "config": config,
        "source_sha256": {str(p.relative_to(ROOT)): sha_file(p) for p in sources},
        "heldout_model_access": False,
        "training_budget_source": "configs/fused_sparse_swiglu_full.json",
        "target": "actual dense BF16 MLP output minus actual prepacked sparse BF16 MLP output",
        "deployment": "x only -> FP32 low-rank tanh correction -> BF16 MLP add -> original model head once",
        "candidate_ties": "argmax first index anchor, remaining seven via torch.topk after excluding anchor",
        "gating": "no timing or generation unless objective improvement and dev+both heldout quality pass",
    }
    write_json(out / "preregistration.json", preregistration)
    manifest = make_output_gap_manifest()
    historical, historical_audit = load_historical_prompts(ROOT)
    data_audit = validate_output_gap_manifest(manifest, historical_prompts=historical)
    manifest_text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in manifest)
    (out / "evaluation_manifest.jsonl").write_text(manifest_text, encoding="utf-8")
    manifest_hash = sha_file(out / "evaluation_manifest.jsonl")
    write_json(
        out / "data_audit.json",
        {
            "validation": data_audit,
            "historical_sources": historical_audit,
            "manifest_sha256": manifest_hash,
        },
    )
    by_split = {s: [r for r in manifest if r["split"] == s] for s in SPLITS}
    if not torch.cuda.is_available():
        write_json(
            out / "decision.json",
            {"status": "BLOCKED", "blocker": "torch.cuda.is_available() is false"},
        )
        raise SystemExit("CUDA unavailable; no synthetic substitute")
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"], revision=config["revision"], local_files_only=True
    )
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model = (
        AutoModelForCausalLM.from_pretrained(
            config["model"],
            revision=config["revision"],
            dtype=torch.bfloat16,
            local_files_only=True,
        )
        .to("cuda")
        .eval()
        .requires_grad_(False)
    )
    assert model.config.hidden_size == 896 and model.config.intermediate_size == 4864
    assert config["target_layer"] == len(model.model.layers) - 1
    assert model.lm_head.bias is None
    layer = model.model.layers[config["target_layer"]]
    dense = layer.mlp
    write_json(
        out / "environment.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(),
            "gpu_memory": torch.cuda.get_device_properties(0).total_memory,
            "nvidia_smi": command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                ]
            ),
            "model_revision": config["revision"],
            "model_dtype": str(next(model.parameters()).dtype),
            "correction_dtype": "float32",
            "tf32_allowed": torch.backends.cuda.matmul.allow_tf32,
            "model_config_commit": model.config._commit_hash,
            "model_implementation_sha256": sha_file(Path(inspect.getfile(type(layer)))),
            "torch_build": torch.__config__.show(),
        },
    )
    # Tokenization checks inspect prompts only, before any held-out model call.
    encoded_ids = []
    for row in manifest:
        from fused_sparse_swiglu import prompt_text

        tokens = tokenizer(
            prompt_text(tokenizer, row["prompt"]), add_special_tokens=False
        ).input_ids
        assert len(tokens) <= config["max_prompt_tokens"], (
            "truncation would alter the data contract"
        )
        encoded_ids.append(hashlib.sha256(json.dumps(tokens).encode()).hexdigest())
    assert len(set(encoded_ids)) == len(encoded_ids), "tokenized prompts overlap"
    write_json(
        out / "tokenization_audit.json",
        {
            "unique_token_sequences": len(encoded_ids),
            "truncated": 0,
            "ordered_token_hashes": encoded_ids,
        },
    )
    log.write(
        "preregistered",
        manifest_sha256=manifest_hash,
        config_sha256=sha_file(out / "preregistration.json"),
    )
    selection, mask_audit = calibrate_mask(
        model, tokenizer, layer, dense, by_split["calibration"], config, log
    )
    sparse = PrepackedSwiGLU(dense.gate_proj, dense.up_proj, dense.down_proj, selection).eval()
    write_json(
        out / "mask.json",
        {
            "indices": selection.tolist(),
            "retained": len(selection),
            "total": 4864,
            "sha256": tensor_hash(selection),
            **mask_audit,
        },
    )
    calibration, numerical = calibration_targets(
        model, tokenizer, layer, dense, sparse, by_split["calibration"], config, log
    )
    corrections, training = {}, {}
    for rank in config["ranks"]:
        for seed in config["seeds"]:
            pair, audit = train_pair(
                calibration["state"].to("cuda"),
                calibration["target"].to("cuda"),
                calibration["contrast"].to("cuda"),
                rank=rank,
                seed=seed,
                steps=config["correction_steps"],
                batch_size=config["correction_batch_size"],
                learning_rate=config["correction_learning_rate"],
            )
            training[f"r{rank}_s{seed}"] = audit
            for objective, correction in pair.items():
                corrections[f"{objective}_r{rank}_s{seed}"] = correction.eval().requires_grad_(
                    False
                )
            log.write("pair_trained", rank=rank, seed=seed, audit=audit)
    write_json(out / "training_audit.json", training)
    torch.save(
        {"mask": selection, "corrections": {n: m.state_dict() for n, m in corrections.items()}},
        out / "corrections.pt",
    )
    checkpoint_hash = sha_file(out / "corrections.pt")
    weight_hashes = {n: state_hash(m) for n, m in corrections.items()}
    # Check serialization independently before validation.
    restored = torch.load(out / "corrections.pt", weights_only=True)
    numerical["checkpoint_roundtrip_exact"] = all(
        torch.equal(v, restored["corrections"][n][k])
        for n, m in corrections.items()
        for k, v in m.state_dict().items()
    )
    assert numerical["checkpoint_roundtrip_exact"], "checkpoint roundtrip changed weights"
    write_json(
        out / "bound_audit.json", bound_audit(model, dense, selection, calibration, corrections)
    )
    full = PrepackedSwiGLU(
        dense.gate_proj, dense.up_proj, dense.down_proj, torch.arange(4864)
    ).eval()
    arms = {"dense": dense, "full_width_control": full, "sparse75": sparse}
    arms.update({n: CorrectedSwiGLU(sparse, m).eval() for n, m in corrections.items()})
    development_rows = evaluate(model, tokenizer, layer, arms, by_split["development"], config, log)
    development = summarize(development_rows, config)
    write_csv(out / "development_rows.csv", development_rows)
    write_json(out / "development_quality.json", development)
    assert all(
        r["max_abs_logit_error"] == 0
        for r in development_rows
        if r["method"] == "full_width_control"
    ), "invalid development control"
    chosen = select_development_arm(development)
    chosen_rank = int(chosen["method"].split("_")[1][1:])
    freeze = {
        "selected_gap_arm": chosen,
        "primary_rank": chosen_rank,
        "all_arms": list(arms),
        "weights_sha256": weight_hashes,
        "checkpoint_sha256": checkpoint_hash,
        "manifest_sha256": manifest_hash,
        "heldout_model_access": False,
        "negative_audit": not chosen["passes_quality_gate"],
        "selection_rule": config["selection_rule"],
    }
    write_json(out / "development_selection.json", freeze)
    freeze_hash = sha_file(out / "development_selection.json")
    log.write("policy_frozen_before_heldout", selection_sha256=freeze_hash, chosen=chosen)
    heldout_rows = []
    for split in ("heldout_matched", "heldout_ood"):
        assert sha_file(out / "development_selection.json") == freeze_hash
        assert sha_file(out / "corrections.pt") == checkpoint_hash
        assert {n: state_hash(m) for n, m in corrections.items()} == weight_hashes
        log.write("heldout_model_access_opened", split=split, frozen_selection_sha256=freeze_hash)
        split_rows = evaluate(model, tokenizer, layer, arms, by_split[split], config, log)
        heldout_rows.extend(split_rows)
        write_csv(out / f"{split}_rows.csv", split_rows)
    heldout = summarize(heldout_rows, config)
    write_json(out / "heldout_quality.json", heldout)
    comparison = paired_comparisons(development_rows + heldout_rows, config)
    write_json(out / "paired_objective_comparison.json", comparison)
    all_rows = development_rows + heldout_rows
    control_rows = [r for r in all_rows if r["method"] == "full_width_control"]
    numerical["full_width_control_max_logit_error"] = max(
        r["max_abs_logit_error"] for r in control_rows
    )
    numerical["full_width_control_mismatches"] = sum(1 - r["top1_agreement"] for r in control_rows)
    numerical["no_heldout_during_fit"] = True
    numerical["checkpoint_unchanged_after_heldout"] = (
        sha_file(out / "corrections.pt") == checkpoint_hash
    )
    numerical["deployed_input"] = (
        "state only; no labels, candidates, or dense activations stored in correction"
    )
    write_json(out / "numerical_checks.json", numerical)
    assert numerical["full_width_control_max_logit_error"] == 0, "invalid full-width control"
    established = objective_improvement_established(comparison, chosen_rank)
    chosen_heldout = [r for r in heldout if r["method"] == chosen["method"]]
    quality = (
        len(chosen_heldout) == 2
        and chosen["passes_quality_gate"]
        and all(r["passes_quality_gate"] for r in chosen_heldout)
    )
    decision = {
        "objective_improvement": {
            "established": established,
            "primary_rank": chosen_rank,
            "rule": config["improvement_rule"],
        },
        "absolute_quality_pass": bool(quality),
        "selected_arm": chosen["method"],
        "certification_status": "unestablished for BF16; analytic norm bound audit recorded",
        "measured_speedup": None,
        "measured_bandwidth": None,
        "timing_status": "not run: objective improvement and absolute quality are prerequisites",
        "bandwidth_status": "unknown; no fresh counters measured because quality prerequisite failed",
        "status": "QUALITY_PASSED_PENDING_TIMING" if quality and established else "NEGATIVE_STOP",
        "manifest_sha256": manifest_hash,
        "selection_sha256": freeze_hash,
    }
    if quality and established:
        from output_gap_timing import run_timing

        timing_config = dict(config)
        timing_config["timing"] = dict(config["timing"], quality_gates_passed=True)
        hidden_name = chosen["method"].replace("gap_", "hidden_", 1)
        timings = run_timing(
            model,
            tokenizer,
            layer,
            {n: arms[n] for n in ("dense", "sparse75", hidden_name, chosen["method"])},
            by_split["heldout_matched"] + by_split["heldout_ood"],
            timing_config,
            out,
            log,
        )
        write_json(out / "timing.json", timings)
        decision["timing_status"] = "measured; consult timing.json before any speed claim"
        decision["status"] = "QUALITY_PASSED_TIMING_REQUIRES_REVIEW"
        decision["bandwidth_status"] = "unknown unless fresh hardware-counter profiling succeeds"
    else:
        write_json(
            out / "timing.json",
            {
                "status": "skipped_by_preregistered_quality_gate",
                "raw_timings": [],
                "measured_speedup": None,
                "physical_dram_bytes": None,
            },
        )
    write_json(out / "decision.json", decision)
    log.write("completed", decision=decision)


if __name__ == "__main__":
    with contextlib.ExitStack():
        main()
