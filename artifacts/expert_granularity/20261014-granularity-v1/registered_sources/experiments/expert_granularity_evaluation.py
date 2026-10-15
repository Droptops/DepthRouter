"""Bounded, exact packed-BF16 capability evaluation for generic candidate sets.

These routines are offline teacher-assisted measurements, not routing or timing
implementations. Candidate banks are tiled. They never approximate a candidate
by masking cached dense activations or by an FP32 down-projection shortcut.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from expert_construction import PackedMLP, capture_decode, final_logits
from torch import Tensor
from torch.nn import functional as F

FLOAT_FIELDS = ("kl", "hidden_mse", "candidate_margin", "max_abs_logit_error")
CANDIDATE_FIELDS = (*FLOAT_FIELDS, "candidate_token")
CONTROL_FIELDS = ("control_kl", "control_token", "permuted_control_kl", "permuted_control_token")
LAYOUT_FIELDS = ("neuron_indices", "candidate_ids", "control_order", "partition",
                 "partition_sha256", "selected_experts")


def validate_indices(neuron_indices: Tensor, width: int) -> Tensor:
    """Require canonical original-neuron order for exact nested-mask equality."""
    indices = torch.as_tensor(neuron_indices).detach().cpu()
    if (indices.dtype != torch.long or indices.ndim != 2 or not indices.numel()
            or indices.shape[1] > width):
        raise ValueError("candidate indices must be nonempty int64 [candidates, active_neurons]")
    if int(indices.min()) < 0 or int(indices.max()) >= width:
        raise ValueError("candidate neuron index is outside the original MLP")
    if indices.shape[1] > 1 and not bool((indices[:, 1:] > indices[:, :-1]).all()):
        raise ValueError("each candidate must have unique ascending original-neuron indices")
    if len(torch.unique(indices, dim=0)) != len(indices):
        raise ValueError("candidate masks must be unique")
    return indices.contiguous()


def _state_metadata(groups, records):
    """Check the exact valid-slot order, including EOS exclusions, before work."""
    order = [(group_index, slot) for group_index, group in enumerate(groups)
             for slot in group["valid"].detach().cpu().nonzero().flatten().tolist()]
    if len(order) != len(records) or not records:
        raise ValueError("valid group slots must match the nonempty state-record sequence")
    for expected, record in zip(order, records):
        if (record["group"], record["slot"]) != expected:
            raise ValueError("state records are not aligned with group/slot order")
    return {
        "dense_token": np.asarray([row["dense_token"] for row in records], dtype=np.int32),
        "dense_margin": np.asarray([row["dense_margin"] for row in records], dtype=np.float32),
        "context_ids": np.asarray([row["id"] for row in records]),
        "positions": np.asarray([row["position"] for row in records], dtype=np.int32),
        "groups": np.asarray([row["group"] for row in records], dtype=np.int32),
        "slots": np.asarray([row["slot"] for row in records], dtype=np.int32),
    }


def _write_npz(path: Path, matrix: dict[str, Any]) -> None:
    """Atomic publication: a truncated archive is never called a complete matrix."""
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **matrix)
    temporary.replace(path)


def _store_candidate_values(matrix, values, valid, cursor, tile_start):
    """Publish only fully computed scalar vectors, including interrupted tiles."""
    if not values:
        return
    transferred = torch.stack(values, dim=1).cpu().numpy()[valid]
    n, columns = transferred.shape[:2]
    for column, key in enumerate(CANDIDATE_FIELDS):
        matrix[key][cursor:cursor + n, tile_start:tile_start + columns] = transferred[:, :, column]
    matrix["measured"][cursor:cursor + n, tile_start:tile_start + columns] = True


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _layout_identity(layout: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(layout.items()):
        array = np.ascontiguousarray(value)
        if array.dtype.hasobject:
            raise ValueError("candidate layout cannot contain pickled object arrays")
        digest.update(key.encode())
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def write_candidate_matrix(path, matrix, *, chunk_size=256):
    """Losslessly publish bounded candidate chunks and a hash-checked JSON index.

    Layout arrays are stored once by content hash in the same directory and
    shared across splits. Candidate fields and measured masks are column-tiled;
    state/control metadata is stored separately. No pickle or numeric downcast
    is used. The manifest appears only after every constituent file is written.
    """
    path = Path(path).with_suffix(".json")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite matrix manifest: {path}")
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or not 1 <= chunk_size <= 256:
        raise ValueError("archive chunks must contain between 1 and 256 candidates")
    states, candidates = matrix["kl"].shape
    if states < 1 or candidates < 1:
        raise ValueError("candidate matrix must be nonempty")
    candidate_keys = (*CANDIDATE_FIELDS, "measured")
    if any(np.asarray(matrix[key]).shape != (states, candidates) for key in candidate_keys):
        raise ValueError("all candidate fields must have the same [states,candidates] shape")
    path.parent.mkdir(parents=True, exist_ok=True)
    layout = {key: np.asarray(matrix[key]) for key in LAYOUT_FIELDS if key in matrix}
    identity = _layout_identity(layout)
    layout_path = path.parent / f"candidate-layout-{identity}.npz"
    if layout_path.exists():
        with np.load(layout_path, allow_pickle=False) as existing:
            observed = {key: existing[key] for key in existing.files}
        if _layout_identity(observed) != identity:
            raise AssertionError("shared candidate layout hash collision or corruption")
    else:
        _write_npz(layout_path, layout)
    metadata_path = path.with_name(path.stem + ".metadata.npz")
    if metadata_path.exists():
        raise FileExistsError(f"refusing to overwrite matrix metadata: {metadata_path}")
    metadata = {key: value for key, value in matrix.items()
                if key not in candidate_keys and key not in LAYOUT_FIELDS}
    _write_npz(metadata_path, metadata)
    chunks = []
    for start in range(0, candidates, chunk_size):
        stop = min(start + chunk_size, candidates)
        chunk_path = path.with_name(path.stem + f".candidates-{start:06d}-{stop:06d}.npz")
        if chunk_path.exists():
            raise FileExistsError(f"refusing to overwrite matrix chunk: {chunk_path}")
        _write_npz(chunk_path, {key: matrix[key][:, start:stop] for key in candidate_keys})
        chunks.append({"file": chunk_path.name, "sha256": _file_sha256(chunk_path),
                       "column_start": start, "column_stop": stop})
    manifest = {
        "format": "depthrouter_candidate_matrix_v1", "states": states, "candidates": candidates,
        "complete": bool(np.asarray(matrix["complete"]).item()), "chunk_size": chunk_size,
        "candidate_ids": np.asarray(matrix["candidate_ids"], dtype=np.int64).tolist(),
        "metadata": {"file": metadata_path.name, "sha256": _file_sha256(metadata_path)},
        "layout": {"file": layout_path.name, "sha256": _file_sha256(layout_path),
                   "identity": identity}, "chunks": chunks,
    }
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return manifest


def read_candidate_matrix(path):
    """Reassemble full CPU arrays after verifying every file hash and shape."""
    path = Path(path).with_suffix(".json")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != "depthrouter_candidate_matrix_v1":
        raise ValueError("unsupported candidate matrix archive format")

    def load(descriptor):
        filename = descriptor["file"]
        if Path(filename).name != filename or "\\" in filename or "/" in filename:
            raise ValueError("matrix archive references must be sibling filenames")
        source = path.parent / filename
        if _file_sha256(source) != descriptor["sha256"]:
            raise AssertionError(f"candidate matrix archive hash mismatch: {filename}")
        with np.load(source, allow_pickle=False) as values:
            return {key: values[key] for key in values.files}

    metadata, layout = load(manifest["metadata"]), load(manifest["layout"])
    if _layout_identity(layout) != manifest["layout"]["identity"]:
        raise AssertionError("candidate layout identity mismatch")
    if set(metadata) & set(layout):
        raise ValueError("candidate matrix metadata/layout fields overlap")
    matrix = {**metadata, **layout}
    states, candidates = int(manifest["states"]), int(manifest["candidates"])
    expected_keys = {*CANDIDATE_FIELDS, "measured"}
    pieces = {key: [] for key in expected_keys}
    cursor = 0
    for descriptor in manifest["chunks"]:
        start, stop = descriptor["column_start"], descriptor["column_stop"]
        if start != cursor or not start < stop <= candidates or stop - start > 256:
            raise ValueError("candidate chunk columns do not form an exact bounded partition")
        values = load(descriptor)
        if set(values) != expected_keys or any(value.shape != (states, stop - start) for value in values.values()):
            raise ValueError("candidate chunk fields or dimensions differ from the manifest")
        for key, value in values.items():
            pieces[key].append(value)
        cursor = stop
    if cursor != candidates or not pieces["kl"]:
        raise ValueError("candidate archive is missing columns")
    matrix.update({key: np.concatenate(value, axis=1) for key, value in pieces.items()})
    if not np.array_equal(matrix["candidate_ids"], manifest["candidate_ids"]):
        raise AssertionError("candidate IDs differ between layout and manifest")
    if len(matrix["candidate_ids"]) != candidates or len(matrix["dense_token"]) != states:
        raise ValueError("candidate metadata dimensions differ from the manifest")
    if bool(np.asarray(matrix["complete"]).item()) != manifest["complete"]:
        raise ValueError("candidate completion flags disagree")
    if manifest["complete"] and (not matrix["measured"].all() or not matrix["control_measured"].all()):
        raise ValueError("archive claims completion with unmeasured candidate/control cells")
    return matrix


def assert_inherited_equal(matrix, inherited_matrix, inherited_candidate_ids):
    """Require exact old-mask membership, raw measurements and oracle inclusion."""
    inherited_ids = np.asarray(inherited_candidate_ids)
    old_count = inherited_matrix["kl"].shape[1]
    lookup = {int(value): column for column, value in enumerate(matrix["candidate_ids"])}
    if (inherited_ids.dtype.kind not in "iu" or inherited_ids.shape != (old_count,)
            or len(np.unique(inherited_ids)) != old_count
            or any(int(value) not in lookup for value in inherited_ids)):
        raise ValueError("inherited candidate IDs must uniquely map every old candidate")
    if not np.array_equal(inherited_ids, inherited_matrix["candidate_ids"]):
        raise ValueError("inherited global candidate IDs differ from the old matrix order")
    columns = [lookup[int(value)] for value in inherited_ids]
    for key in ("context_ids", "positions", "dense_token", "dense_margin", "groups", "slots"):
        if not np.array_equal(matrix[key], inherited_matrix[key]):
            raise AssertionError(f"inherited state metadata differs: {key}")
    if not np.array_equal(matrix["neuron_indices"][columns], inherited_matrix["neuron_indices"]):
        raise AssertionError("inherited masks are not exact original-neuron subsets")
    for key in CANDIDATE_FIELDS:
        if not np.array_equal(matrix[key][:, columns], inherited_matrix[key]):
            raise AssertionError(f"inherited candidate raw measurements differ: {key}")
    for key in CONTROL_FIELDS:
        if not np.array_equal(matrix[key], inherited_matrix[key]):
            raise AssertionError(f"inherited numerical control differs: {key}")
    old_good = ((inherited_matrix["kl"] < 0.01)
                & (inherited_matrix["candidate_token"] == inherited_matrix["dense_token"][:, None])).any(1)
    new_good = ((matrix["kl"] < 0.01)
                & (matrix["candidate_token"] == matrix["dense_token"][:, None])).any(1)
    if np.any(old_good & ~new_good):
        raise AssertionError("nested oracle coverage lost a state accepted by an inherited mask")
    return {"exact_masks": True, "exact_all_raw_candidate_fields": True,
            "exact_controls": True, "old_acceptable_states": int(old_good.sum()),
            "new_acceptable_states": int(new_good.sum()), "monotone_oracle_coverage": True}


@torch.no_grad()
def validate_cached_candidates(
    model, dense, neuron_indices, replay, budget, *, candidate_ids, control_order=None,
):
    """Exact full-model substitution on the prospectively selected mask IDs."""
    indices = validate_indices(neuron_indices, dense.gate_proj.out_features)
    ids = [int(index) for index in candidate_ids]
    if not ids or len(set(ids)) != len(ids) or min(ids) < 0 or max(ids) >= len(indices):
        raise ValueError("parity candidate IDs must be nonempty, unique and in range")
    _, captured = capture_decode(model, dense, copy.deepcopy(replay))
    layer, original = model.model.layers[-1], model.model.layers[-1].mlp
    checks = []
    try:
        layouts = [(candidate_id, indices[candidate_id], "cheap") for candidate_id in ids]
        if control_order is not None:
            width = dense.gate_proj.out_features
            order = torch.as_tensor(control_order).detach().cpu()
            if order.dtype != torch.long or not torch.equal(order.sort().values, torch.arange(width)):
                raise ValueError("full-model parity control_order must be a full neuron permutation")
            layouts += [(-1, torch.arange(width), "control"), (-2, order, "permuted_control")]
        for candidate_id, neuron_order, kind in layouts:
            budget.check()
            module = PackedMLP(dense, neuron_order)
            cached = final_logits(model, captured["base"], module(captured["x"]))
            layer.mlp = module
            actual = model(**copy.deepcopy(replay), use_cache=True,
                           logits_to_keep=1).logits[:, -1].float()
            layer.mlp = original
            check = {"candidate_id": candidate_id, "kind": kind, "states": len(cached),
                     "exact_logits": torch.equal(cached, actual),
                     "max_abs_logit_error": float((cached - actual).abs().max())}
            checks.append(check)
            if not check["exact_logits"]:
                raise AssertionError(f"cached/full-model candidate mismatch: {candidate_id}")
    finally:
        layer.mlp = original
    return checks


@torch.no_grad()
def evaluate_candidates(
    model, dense, neuron_indices, groups, records, path, budget, log,
    *, tile_size=28, control_order=None, inherited_matrix=None, inherited_candidate_ids=None,
    candidate_ids=None, partition=None, partition_sha256=None, selected_experts=None, split=None,
):
    """Measure every candidate with <=28 resident banks and one teacher group.

    Tile order is outermost and group order innermost. The full-vocabulary
    reference/log-probability/probability cache contains only the current group,
    not every context. Compact scalar metrics transfer once per group/tile,
    avoiding five CUDA-to-host synchronizations for every individual candidate.

    On a Python exception, a full-shaped ``*.incomplete.json`` chunk archive is published with
    NaN/-1 sentinels, explicit per-cell measurement masks, all input identities,
    and the completed position. It is inspectable/resumable data, not permission
    to restart or extend the registered budget. This entry point never resumes
    automatically and refuses to overwrite either a complete or partial run.
    """
    path = Path(path).with_suffix(".json")
    incomplete_path = path.with_name(path.stem + ".incomplete.json")
    if path.exists() or incomplete_path.exists():
        raise FileExistsError("refusing to overwrite complete or incomplete candidate measurements")
    if not isinstance(tile_size, int) or isinstance(tile_size, bool) or not 1 <= tile_size <= 28:
        raise ValueError("candidate tile_size must be an integer in [1,28]")
    indices = validate_indices(neuron_indices, dense.gate_proj.out_features)
    metadata = _state_metadata(groups, records)
    if model.lm_head.weight.shape[0] >= 2**24:
        raise ValueError("compact scalar transfer requires vocabulary token IDs below 2**24")
    if (inherited_matrix is None) != (inherited_candidate_ids is None):
        raise ValueError("inherited matrix and its candidate mapping must be supplied together")
    if inherited_matrix is not None and not bool(np.asarray(inherited_matrix.get("complete", False)).item()):
        raise ValueError("inherited baseline measurements must be complete")
    width = dense.gate_proj.out_features
    order = torch.arange(width) if control_order is None else torch.as_tensor(control_order).detach().cpu()
    if (order.dtype != torch.long or order.shape != (width,)
            or not torch.equal(order.sort().values, torch.arange(width))):
        raise ValueError("control_order must be a full original-neuron permutation")
    count, candidates = len(records), len(indices)
    global_ids = np.arange(candidates, dtype=np.int64) if candidate_ids is None else np.asarray(candidate_ids)
    if (global_ids.dtype.kind not in "iu" or global_ids.shape != (candidates,)
            or len(np.unique(global_ids)) != candidates or (global_ids < 0).any()):
        raise ValueError("candidate_ids must be unique nonnegative global integer IDs")
    matrix = {key: np.full((count, candidates), np.nan, dtype=np.float32) for key in FLOAT_FIELDS}
    matrix["candidate_token"] = np.full((count, candidates), -1, dtype=np.int32)
    matrix.update({key: np.full(count, np.nan, dtype=np.float32)
                   for key in ("control_kl", "permuted_control_kl")})
    matrix.update({key: np.full(count, -1, dtype=np.int32)
                   for key in ("control_token", "permuted_control_token")})
    matrix.update(metadata)
    matrix.update({
        "neuron_indices": indices.numpy().astype(np.int32),
        "candidate_ids": global_ids.astype(np.int32),
        "control_order": order.numpy().astype(np.int32),
        "measured": np.zeros((count, candidates), dtype=bool),
        "control_measured": np.zeros(count, dtype=bool),
        "complete": np.array(False), "tile_size": np.array(tile_size),
        "states_requested": np.array(count), "candidates_requested": np.array(candidates),
        "method": np.array("actual_prepacked_BF16_gate_up_down_real_norm_full_head"),
        "teacher_cache_scope": np.array("one_captured_group_at_a_time"),
    })
    if partition is not None:
        labels = np.asarray(partition)
        if labels.dtype.kind not in "iu" or labels.shape != (width,):
            raise ValueError("partition metadata must give one integer label per original neuron")
        matrix["partition"] = labels.astype(np.int64)
    if partition_sha256 is not None:
        if (not isinstance(partition_sha256, str) or len(partition_sha256) != 64
                or any(character not in "0123456789abcdef" for character in partition_sha256)):
            raise ValueError("partition_sha256 must be a lowercase hexadecimal SHA-256 digest")
        matrix["partition_sha256"] = np.array(partition_sha256)
    if selected_experts is not None:
        experts = np.asarray(selected_experts)
        if experts.dtype.kind not in "iu" or experts.ndim != 2 or len(experts) != candidates:
            raise ValueError("selected_experts must provide an integer expert set per candidate")
        matrix["selected_experts"] = experts.astype(np.int64)
    if split is not None:
        matrix["split"] = np.array(str(split))
    device = dense.gate_proj.weight.device
    path.parent.mkdir(parents=True, exist_ok=True)
    phase, tile_start, group_index, pending = "controls", -1, -1, None
    try:
        # Control modules are released before candidate tiles are constructed.
        controls = [PackedMLP(dense, torch.arange(width)), PackedMLP(dense, order)]
        cursor = 0
        for group_index, cpu in enumerate(groups):
            budget.check()
            data = {key: cpu[key].to(device) for key in ("x", "base", "output")}
            valid = cpu["valid"].detach().cpu().numpy().astype(bool)
            n = int(valid.sum())
            reference = final_logits(model, data["base"], data["output"])
            if not np.array_equal(reference.argmax(-1).cpu().numpy()[valid], matrix["dense_token"][cursor:cursor + n]):
                raise AssertionError("stored dense-token metadata does not reproduce the captured state")
            logp = F.log_softmax(reference, -1)
            p = logp.exp()
            for prefix, module in zip(("control", "permuted_control"), controls):
                budget.check()
                candidate = final_logits(model, data["base"], module(data["x"]))
                if prefix == "control" and not torch.equal(reference, candidate):
                    raise AssertionError("original-order all-neuron control is not bitwise dense")
                kl = (p * (logp - F.log_softmax(candidate, -1))).sum(-1).clamp_min(0)
                values = torch.stack((kl, candidate.argmax(-1).float()), dim=-1).cpu().numpy()[valid]
                matrix[prefix + "_kl"][cursor:cursor + n] = values[:, 0]
                matrix[prefix + "_token"][cursor:cursor + n] = values[:, 1].astype(np.int32)
            matrix["control_measured"][cursor:cursor + n] = True
            cursor += n
        del controls, module
        phase = "candidates"
        for tile_start in range(0, candidates, tile_size):
            budget.check()
            tile_stop = min(tile_start + tile_size, candidates)
            modules = [PackedMLP(dense, index) for index in indices[tile_start:tile_stop]]
            cursor = 0
            for group_index, cpu in enumerate(groups):
                budget.check()
                data = {key: cpu[key].to(device) for key in ("x", "base", "output")}
                valid = cpu["valid"].detach().cpu().numpy().astype(bool)
                n = int(valid.sum())
                reference = final_logits(model, data["base"], data["output"])
                logp = F.log_softmax(reference, -1)
                p = logp.exp()
                values = []
                pending = (values, valid, cursor, tile_start)
                for module in modules:
                    budget.check()
                    output = module(data["x"])
                    candidate = final_logits(model, data["base"], output)
                    kl = (p * (logp - F.log_softmax(candidate, -1))).sum(-1).clamp_min(0)
                    top = candidate.topk(2, dim=-1).values
                    hidden = (output.float() - data["output"].float()).square().mean((-1, -2))
                    values.append(torch.stack((kl, hidden, top[:, 0] - top[:, 1],
                                               (candidate - reference).abs().max(-1).values,
                                               candidate.argmax(-1).float()), dim=-1))
                _store_candidate_values(matrix, values, valid, cursor, tile_start)
                pending = None
                cursor += n
            del modules, module, values
            log("candidate_tile_complete", matrix=path.name, tile_start=tile_start,
                tile_stop=tile_stop, total_candidates=candidates, states=count)
        if not matrix["measured"].all() or not matrix["control_measured"].all():
            raise AssertionError("matrix is incomplete despite normal evaluation exit")
        if inherited_matrix is not None:
            assert_inherited_equal(matrix, inherited_matrix, inherited_candidate_ids)
            matrix["inherited_candidate_ids"] = np.asarray(inherited_candidate_ids, dtype=np.int32)
            matrix["inherited_exact"] = np.array(True)
        matrix["complete"] = np.array(True)
        write_candidate_matrix(path, matrix)
        return matrix
    except Exception as error:
        if pending is not None:
            try:
                _store_candidate_values(matrix, *pending)
            except (RuntimeError, ValueError, TypeError, MemoryError) as flush_error:
                matrix["partial_scalar_flush_error"] = np.array(f"{type(flush_error).__name__}: {flush_error}")
        matrix.update({"complete": np.array(False), "interrupted_phase": np.array(phase),
                       "interrupted_tile_start": np.array(tile_start),
                       "interrupted_group_index": np.array(group_index),
                       "candidate_cells_completed": np.array(int(matrix["measured"].sum())),
                       "error_type": np.array(type(error).__name__), "error": np.array(str(error))})
        write_candidate_matrix(incomplete_path, matrix)
        raise
