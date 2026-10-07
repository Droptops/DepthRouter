# DepthRouter

**A bandwidth-constrained fixed-point solver with an explicit memory hierarchy.**

The state is not only the residual stream. The current formulation is

```text
s_k = (h_k, c_k, p_k)
```

where `h_k` is the residual/scratch state, `c_k` is the residency set for cold operator slices, and `p_k` is the previous prediction distribution.

The solver chooses from four moves:

- `spin`: execute resident work without intentionally fetching a cold slice
- `fault`: bring in one cold slice, coalesced across all tokens requesting it
- `write`: emit the single legal KV update
- `halt`: commit the current prediction

Depth is therefore the number of spins the policy buys. It is not the primary state variable.

## Promotion gate

Do not promote another routing, adapter, acceleration, or halting mechanism until the hardware claim survives one measurement.

On the same decode tokens, profile:

1. stock forward
2. naive four-pass loop
3. four-pass loop with KV written once
4. KV written once plus a cold MLP that is skipped unless previous-spin logit KL exceeds a threshold

Measure actual memory-controller / DRAM bytes and L2 hit rate. The decision metric is:

```text
(reference cross entropy - schedule cross entropy) / measured bytes moved
```

The placement thesis is falsified if schedule 4 does not beat schedule 3 after charging the gate's own traffic.

Tokens requesting the same cold slice must be batched together. Per-token depth without coalesced misses is not a bandwidth optimization.

## Code

The original toy recurrent-transformer harness remains in `src/depth_router/model.py` as an ablation scaffold.

The memory-explicit state and coalescing primitives are in:

```text
src/depth_router/placement.py
```

The DGX Spark/Qwen implementation and Nsight Compute falsification harness are carried in the companion `loopkit` package.

## Claim boundary

No hardware-efficiency claim is established until a named GPU reports profiler-derived bytes and cache behavior. Parameter sharing alone is not evidence of reduced memory traffic.
