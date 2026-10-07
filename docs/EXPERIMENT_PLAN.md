# Experiment plan

## Thesis

DepthRouter is a bandwidth-constrained fixed-point solver, not an adaptive-depth layer stack.

The state is:

```text
s_k = (h_k, c_k, p_k)
```

and the legal moves are `spin`, `fault`, `write`, and `halt`.

The hot operator is an interpreter over `h_k`. The cold table is program memory. A fault loads cold program state; a spin advances the resident computation; KV write is legal once.

## First falsification experiment

Pin one dense mid-stack block of a small decoder and run the same decode tokens under four schedules:

1. `stock`: one ordinary block evaluation.
2. `naive4`: four full passes with independent per-pass KV.
3. `kv_once4`: four full passes with a single KV write.
4. `gated4`: one KV write, resident hot-core spins, and a cold MLP only when previous-spin logit KL clears the threshold.

The fourth schedule must batch tokens by predicted cold slice so identical misses are coalesced into one dense call.

## Measurement

Use hardware counters, not an analytic byte model.

Record:

- DRAM / memory-controller bytes read and written
- L2 read hit rate
- final cross entropy on the same tokens
- wall-clock decode latency as supporting telemetry

The sole promotion metric is:

```text
nats_per_byte = (reference_CE - schedule_CE) / measured_DRAM_bytes
```

The gate's own traffic belongs in the denominator.

## Gate

**FALSIFIED:** `gated4 nats_per_byte <= kv_once4 nats_per_byte`.

**SUPPORTED FOR NEXT STAGE:** `gated4 nats_per_byte > kv_once4 nats_per_byte`.

No LoRA variant, learned halter, Anderson accelerator, or more elaborate policy should be promoted before this result exists.

## After a pass

Only after the hardware gate passes:

- train a policy over the finite move set;
- keep residency in the state;
- train against measured move costs;
- preserve coalesced faults as a hard systems constraint.

The v0 pointer-chasing harness remains useful only as a software sanity check, not as evidence for the systems thesis.
