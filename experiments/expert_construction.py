"""Bounded final-MLP expert capability audit. Oracles are never runtime routers.

Original weights are frozen. Hard forwards use actual packed BF16 projections;
assignment gradients use a documented dense-activation straight-through surrogate.
No held-out model call is allowed until partitions and fixed choices are sealed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import platform
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from depth_router.expert_audit_metrics import (
    candidate_state_metrics,
    paired_context_difference,
    portfolio_gate,
    select_fixed_candidate,
    summarize_candidates,
)
from depth_router.expert_partitions import (
    AssignmentRecipe,
    candidate_combinations,
    coactivation_partition,
    fit_assignment_pair,
    partition_indices,
    random_partition,
    tensor_sha256,
    validate_partition,
)

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("calibration", "development", "heldout_matched", "heldout_ood")


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


class Budget:
    """Single-GPU elapsed wall time upper bound, including CPU waits and loading."""

    def __init__(self, seconds):
        self.start = time.monotonic()
        self.seconds = seconds

    def check(self):
        if time.monotonic() - self.start >= self.seconds:
            raise TimeoutError("preregistered single-GPU wall-time ceiling reached")

    def elapsed(self):
        return time.monotonic() - self.start


class Log:
    def __init__(self, directory):
        self.path = directory / "events.jsonl"

    def __call__(self, event, **fields):
        row = {"utc": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
        line = json.dumps(row, sort_keys=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        print(line, flush=True)


class PackedMLP(nn.Module):
    """Corresponding rows/columns packed once, with explicitly supplied order."""

    def __init__(self, dense, indices):
        super().__init__()
        indices = indices.to(dense.gate_proj.weight.device)
        self.register_buffer("gate", dense.gate_proj.weight.detach()[indices].contiguous())
        self.register_buffer("up", dense.up_proj.weight.detach()[indices].contiguous())
        self.register_buffer("down", dense.down_proj.weight.detach()[:, indices].contiguous())

    def forward(self, x):
        return F.linear(F.silu(F.linear(x, self.gate)) * F.linear(x, self.up), self.down)


def final_logits(model, base, output):
    # Preserve BF16 residual addition, real final RMSNorm, and full vocabulary head.
    return model.lm_head(model.model.norm(base + output)).float()[:, -1]


@torch.no_grad()
def capture_decode(model, dense, kwargs):
    layer = model.model.layers[-1]
    cap = {}

    def pre(name):
        def hook(_module, args):
            cap[name] = args[0].detach().clone()
        return hook

    def post(_module, _args, output):
        cap["output"] = output.detach().clone()

    handles = [dense.register_forward_pre_hook(pre("x")), dense.register_forward_hook(post),
               dense.down_proj.register_forward_pre_hook(pre("activation")),
               layer.post_attention_layernorm.register_forward_pre_hook(pre("base"))]
    try:
        result = model(**kwargs, use_cache=True, logits_to_keep=1)
    finally:
        for handle in handles:
            handle.remove()
    logits = result.logits[:, -1].float()
    cached = final_logits(model, cap["base"], cap["output"])
    if not torch.equal(logits, cached):
        raise AssertionError("cached dense residual/norm/head differs from complete model")
    return result, cap


@torch.no_grad()
def collect_states(model, tokenizer, dense, rows, config, output, log, budget):
    """Teacher-forced cached decoding; no states after an individual context's EOS."""
    split = rows[0]["split"]
    records, groups = [], []
    replay = None
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos
    eos = torch.tensor(eos, device="cuda")
    width = config["batch_size"]
    truncations, zero_states = 0, []
    for start in range(0, len(rows), width):
        budget.check()
        batch = rows[start:start + width]
        if len(batch) != width:
            raise ValueError("preregistered splits must be divisible by decode batch size")
        texts = [tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}], tokenize=False,
            add_generation_prompt=True,
        ) for row in batch]
        lengths = [len(tokenizer(text, add_special_tokens=False).input_ids) for text in texts]
        truncations += sum(length > config["max_prompt_tokens"] for length in lengths)
        encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                            max_length=config["max_prompt_tokens"], add_special_tokens=False)
        encoded = {key: value.to("cuda") for key, value in encoded.items()}
        encoded["position_ids"] = (encoded["attention_mask"].long().cumsum(-1) - 1).clamp_min(0)
        result = model(**encoded, use_cache=True, logits_to_keep=1)
        next_token = result.logits[:, -1].argmax(-1)
        active = ~torch.isin(next_token, eos)
        mask = encoded["attention_mask"]
        cache = result.past_key_values
        prefixes = [encoded["input_ids"][i][mask[i].bool()].tolist() for i in range(width)]
        seen = set()
        for position in range(1, max(config["continuation_positions"]) + 1):
            budget.check()
            for i in range(width):
                prefixes[i].append(int(next_token[i]))
            mask = torch.cat((mask, torch.ones((width, 1), device="cuda", dtype=mask.dtype)), 1)
            pos = mask.long().cumsum(-1)[:, -1:] - 1
            kwargs = {"input_ids": next_token[:, None], "attention_mask": mask,
                      "past_key_values": cache, "position_ids": pos}
            if start == 0 and position == 1:
                replay = copy.deepcopy(kwargs)
            result, cap = capture_decode(model, dense, kwargs)
            if position in config["continuation_positions"]:
                group_index = len(groups)
                winner = result.logits[:, -1].argmax(-1)
                top = result.logits[:, -1].float().topk(2, dim=-1).values
                cap["valid"] = active.detach().clone()
                cap["dense_token"] = winner.detach().clone()
                cap["dense_margin"] = top[:, 0] - top[:, 1]
                groups.append({key: value.cpu() for key, value in cap.items()})
                for i, row in enumerate(batch):
                    if not bool(active[i]):
                        continue
                    seen.add(i)
                    records.append({"state_index": len(records), "group": group_index, "slot": i,
                                    "id": row["id"], "split": split, "task": row["task"],
                                    "stratum": row["stratum"], "position": position,
                                    "prefix_token_ids": prefixes[i].copy(),
                                    "dense_token": int(winner[i]),
                                    "dense_margin": float(cap["dense_margin"][i])})
            next_token = result.logits[:, -1].argmax(-1)
            active &= ~torch.isin(next_token, eos)
            cache = result.past_key_values
        zero_states.extend(row["id"] for i, row in enumerate(batch) if i not in seen)
        if start % (width * 8) == 0:
            log("capture_progress", split=split, contexts=start + width, states=len(records))
    with (output / f"states-{split}.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    # Large deterministic activation caches stay local; exported state metadata + model pin
    # reconstruct them. Every scalar candidate measurement is retained in tracked NPZ files.
    torch.save(groups, output / f"cache-{split}.pt")
    write_json(output / f"capture-{split}.json", {
        "contexts_requested": len(rows), "contexts_evaluated": len(rows) - len(zero_states),
        "states": len(records), "zero_state_context_ids": zero_states,
        "prompt_truncations": truncations, "positions": config["continuation_positions"],
        "stop_on_eos": True, "dense_cache_reconstruction_exact": True,
        "cache_sha256": sha_file(output / f"cache-{split}.pt"),
    })
    return groups, records, replay


def construction_callback(model, dense, calibration, budget, log):
    keys = ("x", "base", "output", "activation")
    device = dense.gate_proj.weight.device
    flat = {key: torch.cat([g[key][g["valid"]] for g in calibration]).to(device)
            for key in keys}
    down_fp32 = dense.down_proj.weight.detach().float()
    counter = {"hidden": 0, "output": 0}

    def callback(objective, row_ids, masks):
        budget.check()
        data = {key: value[row_ids.to(device)] for key, value in flat.items()}
        with torch.no_grad():
            teacher = F.log_softmax(final_logits(model, data["base"], data["output"]), -1)
        losses = []
        for mask in masks:
            indices = torch.nonzero(mask.detach() > 0.5).flatten()
            if len(indices) != 3648:
                raise AssertionError("hard candidate must contain exactly six experts")
            with torch.no_grad():
                actual = PackedMLP(dense, indices)(data["x"]).float()
            surrogate = F.linear(data["activation"].float() * mask, down_fp32)
            corrected_gradient = actual + (surrogate - surrogate.detach())
            if objective == "hidden":
                loss = (corrected_gradient - data["output"].float()).square().mean((-1, -2))
                loss = loss / data["output"].float().square().mean((-1, -2)).clamp_min(1e-8)
            else:
                student = final_logits(model, data["base"], corrected_gradient.to(torch.bfloat16))
                loss = (teacher.exp() * (teacher - F.log_softmax(student, -1))).sum(-1)
            losses.append(loss)
        counter[objective] += 1
        if counter[objective] % 20 == 0:
            log("construction_progress", objective=objective, step=counter[objective],
                sampled_loss=float(torch.stack(losses).detach().min(0).values.mean()))
        return torch.stack(losses)

    return callback, len(flat["x"]), flat["activation"][:, 0]


@torch.no_grad()
def validate_cached_candidates(model, dense, labels, replay, budget):
    """Full-model substitution on 16 calibration prefixes, all 28 plus two controls."""
    _, cap = capture_decode(model, dense, copy.deepcopy(replay))
    records = []
    combinations = list(candidate_combinations().tolist()) + [list(range(8))]
    for combination in combinations:
        budget.check()
        indices = partition_indices(labels, combination).sort().values
        module = PackedMLP(dense, indices)
        offline = final_logits(model, cap["base"], module(cap["x"]))
        layer = model.model.layers[-1]
        layer.mlp = module
        try:
            complete = model(**copy.deepcopy(replay), use_cache=True,
                             logits_to_keep=1).logits[:, -1].float()
        finally:
            layer.mlp = dense
        exact = torch.equal(offline, complete)
        records.append({"experts": combination, "states": len(offline), "exact_logits": exact,
                        "max_abs_logit_error": float((offline - complete).abs().max())})
        if not exact:
            raise AssertionError("cached candidate evaluator differs from full-model execution")
    return records


@torch.no_grad()
def evaluate_partition(model, dense, labels, groups, records, path, budget, log):
    path = Path(path)
    count = len(records)
    matrix = {name: np.empty((count, 28), dtype=np.float32)
              for name in ("kl", "hidden_mse", "candidate_margin", "max_abs_logit_error")}
    matrix["candidate_token"] = np.empty((count, 28), dtype=np.int32)
    for name in ("control_kl", "permuted_control_kl"):
        matrix[name] = np.empty(count, dtype=np.float32)
    for name in ("control_token", "permuted_control_token"):
        matrix[name] = np.empty(count, dtype=np.int32)
    combos = candidate_combinations().tolist()
    modules = [PackedMLP(dense, partition_indices(labels, combo).sort().values) for combo in combos]
    modules += [PackedMLP(dense, torch.arange(dense.gate_proj.out_features)),
                PackedMLP(dense, partition_indices(labels))]
    device = dense.gate_proj.weight.device

    def metadata(selected_records):
        return {
            "dense_token": np.array([r["dense_token"] for r in selected_records]),
            "dense_margin": np.array([r["dense_margin"] for r in selected_records]),
            "context_ids": np.array([r["id"] for r in selected_records]),
            "positions": np.array([r["position"] for r in selected_records]),
            "selected_experts": np.array(combos), "partition": labels.cpu().numpy(),
        }

    cursor = 0
    try:
        for group_index, cpu in enumerate(groups):
            budget.check()
            data = {key: value.to(device) for key, value in cpu.items()}
            reference = final_logits(model, data["base"], data["output"])
            logp = F.log_softmax(reference, -1)
            p = logp.exp()
            valid = data["valid"]
            n = int(valid.sum())
            for index, module in enumerate(modules):
                budget.check()
                output = module(data["x"])
                candidate = final_logits(model, data["base"], output)
                kl = (p * (logp - F.log_softmax(candidate, -1))).sum(-1).clamp_min(0)
                token = candidate.argmax(-1)
                if index < 28:
                    top = candidate.topk(2, dim=-1).values
                    values = {"kl": kl, "candidate_token": token,
                              "hidden_mse": (output.float() - data["output"].float()).square()
                              .mean((-1, -2)), "candidate_margin": top[:, 0] - top[:, 1],
                              "max_abs_logit_error": (candidate - reference).abs().max(-1).values}
                    for key, value in values.items():
                        matrix[key][cursor:cursor + n, index] = value[valid].cpu().numpy()
                else:
                    prefix = "control" if index == 28 else "permuted_control"
                    matrix[prefix + "_kl"][cursor:cursor + n] = kl[valid].cpu().numpy()
                    matrix[prefix + "_token"][cursor:cursor + n] = token[valid].cpu().numpy()
            cursor += n
            if group_index % 64 == 0:
                log("candidate_progress", matrix=path.name, groups=group_index + 1,
                    total_groups=len(groups))
        if cursor != count:
            raise AssertionError("matrix row/state alignment failed")
    except Exception as error:
        # Only complete rows are retained: no uninitialized candidate/control
        # entries may be mistaken for measured values after interruption.
        incomplete = {key: value[:cursor] for key, value in matrix.items()}
        incomplete.update(metadata(records[:cursor]))
        incomplete.update({
            "complete": np.array(False), "states_completed": np.array(cursor),
            "states_requested": np.array(count), "error_type": np.array(type(error).__name__),
            "error": np.array(str(error)),
        })
        np.savez_compressed(path.with_name(path.stem + ".incomplete.npz"), **incomplete)
        raise
    matrix.update(metadata(records))
    np.savez_compressed(path, **matrix)
    return matrix


def summarize(matrix, fixed, split, config):
    agreement = matrix["candidate_token"] == matrix["dense_token"][:, None]
    return summarize_candidates(
        matrix["kl"], agreement, matrix["hidden_mse"], matrix["context_ids"],
        matrix["dense_margin"], fixed_selection=fixed, split=split,
        control_kl=matrix["control_kl"],
        control_agreement=matrix["control_token"] == matrix["dense_token"],
        bootstrap_repetitions=config["bootstrap_repetitions"],
        bootstrap_seed=config["bootstrap_seed"],
    )


def verify_seal(seal, output):
    if not seal:
        raise RuntimeError("held-out access requires a nonempty development seal")
    for filename, expected in seal.items():
        if sha_file(output / filename) != expected:
            raise RuntimeError(f"pre-heldout seal changed: {filename}")


def register(config_path, output):
    config = json.loads(config_path.read_text())
    output.mkdir(parents=True, exist_ok=False)
    files = [config_path, ROOT / config["manifest"], ROOT / "data/expert_audit/data_audit.json",
             ROOT / "data/expert_audit/source_provenance.json", Path(__file__),
             ROOT / "EXPERT_CONSTRUCTION_PREREGISTRATION.md"]
    files += sorted((ROOT / "src/depth_router").glob("expert_*.py"))
    files += [ROOT / "scripts/run_expert_construction.ps1"]
    files += [ROOT / "requirements-photonic.txt", ROOT / "Dockerfile.photonic",
              ROOT / "src/depth_router/output_gap.py"]
    files += sorted((ROOT / "data/expert_audit").glob("historical_*.json*"))
    for dependency in ("scripts/prepare_expert_audit_data.py", "src/depth_router/output_gap_data.py"):
        source = ROOT / dependency
        if source.exists():
            files.append(source)
    source_hashes = {}
    for source in files:
        name = source.relative_to(ROOT).as_posix()
        source_hashes[name] = sha_file(source)
        snapshot = output / "registered_sources" / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(source.read_bytes())
    write_json(output / "preregistration.json", {
        "utc": datetime.now(timezone.utc).isoformat(), "config": config,
        "source_sha256": source_hashes, "before_model_load_fit_and_heldout": True,
        "frozen_assignment_recipe": asdict(AssignmentRecipe(**config["assignment"])),
    })
    return config


def run(config, output):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import __version__ as transformers_version

    budget, log = Budget(config["gpu_wall_budget_seconds"]), Log(output)
    try:
        torch.manual_seed(17)
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = AutoModelForCausalLM.from_pretrained(
            config["model"], revision=config["revision"], torch_dtype=torch.bfloat16,
            attn_implementation=config["attention"], local_files_only=True,
        ).to("cuda").eval()
        model.requires_grad_(False)
        tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["revision"],
                                                 local_files_only=True, padding_side="left")
        tokenizer.truncation_side = "left"
        dense = model.model.layers[-1].mlp
        assert dense.gate_proj.out_features == 4864 and len(model.model.layers) == 24
        assert not any(parameter.requires_grad for parameter in model.parameters())
        model_hash_before = {name: tensor_sha256(tensor) for name, tensor in model.state_dict().items()}
        write_json(output / "environment.json", {
            "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
            "transformers": transformers_version, "cuda": torch.version.cuda,
            "numpy": np.__version__,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "torch_num_threads": torch.get_num_threads(),
            "python": platform.python_version(), "platform": platform.platform(),
            "nvidia_smi": subprocess.run(["nvidia-smi"], capture_output=True, text=True,
                                         check=False).stdout,
            "model_class": type(model).__qualname__, "model_source_sha256": sha_file(
                inspect.getfile(type(model))), "dtype": str(next(model.parameters()).dtype),
            "attention": model.config._attn_implementation, "allow_tf32": False,
            "original_weight_sha256": model_hash_before,
            "cuda_graphs": False, "stage": "offline_quality_not_runtime_benchmark",
        })
        rows = [json.loads(line) for line in (ROOT / config["manifest"]).read_text().splitlines()]
        cal, cal_records, replay = collect_states(
            model, tokenizer, dense, [r for r in rows if r["split"] == "calibration"],
            config, output, log, budget,
        )
        callback, count, activations = construction_callback(model, dense, cal, budget, log)
        partitions = {}
        for seed in config["seeds"]:
            budget.check()
            labels = random_partition(4864, seed=seed)
            partitions[f"A-{seed}"] = labels
            partitions[f"B-{seed}"] = coactivation_partition(
                activations, seed=seed, **config["coactivation"])
            fitted, audit = fit_assignment_pair(
                labels, count, callback, seed=seed, device="cuda",
                recipe=AssignmentRecipe(**config["assignment"]),
            )
            partitions[f"C-{seed}"], partitions[f"D-{seed}"] = fitted["hidden"], fitted["output"]
            write_json(output / f"construction-{seed}.json", audit)
            for arm in "ABCD":
                key = f"{arm}-{seed}"
                validate_partition(partitions[key])
                write_json(output / f"partition-{key}.json", partitions[key].tolist())
            log("seed_fit_complete", seed=seed, elapsed_seconds=budget.elapsed())
        del callback, activations
        numerical = {}
        for key, labels in partitions.items():
            numerical[key] = validate_cached_candidates(model, dense, labels, replay, budget)
        write_json(output / "full_model_cached_validation.json", numerical)
        del replay
        fixed, matrices, summaries = {}, {}, {}
        seal = None
        for split in SPLITS:
            budget.check()
            if split.startswith("heldout"):
                verify_seal(seal, output)
            if split == "calibration":
                groups, records = cal, cal_records
            else:
                groups, records, _ = collect_states(
                    model, tokenizer, dense, [r for r in rows if r["split"] == split],
                    config, output, log, budget,
                )
            matrices[split] = {}
            for key, labels in partitions.items():
                path = output / f"matrix-{split}-{key}.npz"
                matrix = evaluate_partition(model, dense, labels, groups, records, path, budget, log)
                matrices[split][key] = matrix
                control_agreement = (matrix["control_token"] == matrix["dense_token"]).mean()
                if control_agreement < 0.99 or matrix["control_kl"].mean() >= 0.01:
                    raise AssertionError("actual-layout all-eight numerical control failed")
                if split == "development":
                    fixed[key] = select_fixed_candidate(
                        matrix["kl"], matrix["candidate_token"] == matrix["dense_token"][:, None])
                if split != "calibration":
                    summaries.setdefault(key, {})[split] = summarize(matrix, fixed[key], split, config)
                    write_json(output / f"summary-{split}-{key}.json", summaries[key][split])
                log("matrix_complete", split=split, partition=key, states=len(records),
                    elapsed_seconds=budget.elapsed())
            if split == "development":
                write_json(output / "fixed_selection.json", fixed)
                # Any future family choice is determined here, without held-out outputs.
                ranking = sorted("ABCD", key=lambda arm: (
                    -np.mean([summaries[f"{arm}-{seed}"][split]["portfolio_advantage"]
                              for seed in config["seeds"]]),
                    -np.mean([summaries[f"{arm}-{seed}"][split]["cheap_coverage"]
                              for seed in config["seeds"]]), arm))
                write_json(output / "family_selection.json", {"arm": ranking[0],
                                                               "selected_on": "development"})
                names = [f"partition-{key}.json" for key in partitions]
                names += ["fixed_selection.json", "family_selection.json", "preregistration.json"]
                seal = {name: sha_file(output / name) for name in names}
                write_json(output / "heldout_seal.json", seal)
                log("heldout_sealed", selected_family=ranking[0], seal=seal)
        gates = {key: portfolio_gate(value) for key, value in summaries.items()}
        for key in partitions:
            summaries[key]["calibration"] = summarize(
                matrices["calibration"][key], fixed[key], "calibration", config)
        write_json(output / "all_summaries.json", summaries)
        paired = {}
        for split in SPLITS:
            vectors = {}
            for key, matrix in matrices[split].items():
                vectors[key] = candidate_state_metrics(
                    matrix["kl"], matrix["candidate_token"] == matrix["dense_token"][:, None],
                    fixed_selection=fixed[key])
            paired[split] = {}
            for metric in ("cheap_coverage", "fixed_cheap_coverage", "portfolio_advantage",
                           "oracle_kl", "oracle_agreement"):
                lhs = np.stack([vectors[f"D-{s}"][metric] for s in config["seeds"]])
                rhs = np.stack([vectors[f"C-{s}"][metric] for s in config["seeds"]])
                paired[split][metric] = paired_context_difference(
                    lhs, rhs, matrices[split]["A-17"]["context_ids"],
                    bootstrap_repetitions=config["bootstrap_repetitions"],
                    bootstrap_seed=config["bootstrap_seed"])
        write_json(output / "paired_D_minus_C.json", paired)
        family = json.loads((output / "family_selection.json").read_text())["arm"]
        eligible = all(gates[f"{family}-{seed}"]["point_pass"] for seed in config["seeds"])
        original_unchanged = all(tensor_sha256(value) == model_hash_before[name]
                                 for name, value in model.state_dict().items())
        assert original_unchanged
        write_json(output / "promotion_decision.json", {
            "phase1_complete": True, "phase1_gates": gates,
            "development_selected_family": family, "selected_family_eligible": eligible,
            "phase2": "eligible_pending_execution" if eligible else "skipped_phase1_gate_failed",
            "phase3": "not_run_requires_phase2_quality_pass", "promote": False,
            "original_weights_unchanged": original_unchanged,
            "measured_latency": None, "measured_dram_bytes": None,
            "physical_counter_attempt": "not_attempted_prerequisite_quality_gate",
            "gpu_budget_ceiling_seconds": budget.seconds,
            "single_gpu_wall_elapsed_seconds": budget.elapsed(),
            "cost_note": "audit wall time is not inference latency or measured bandwidth",
        })
        log("phase1_complete", selected_family_eligible=eligible, elapsed_seconds=budget.elapsed())
    except Exception as exc:
        if hasattr(exc, "assignment_partial"):
            torch.save(exc.assignment_partial, output / "incomplete-assignment.pt")
            write_json(output / "incomplete-assignment.json", exc.assignment_partial["audit"])
        write_json(output / "incomplete.json", {
            "error_type": type(exc).__name__, "error": str(exc),
            "single_gpu_wall_elapsed_seconds": budget.elapsed(), "promote": False,
        })
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/expert_construction.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = register(args.config.resolve(), args.output_dir.resolve())
    run(config, args.output_dir.resolve())


if __name__ == "__main__":
    main()
