# DepthRouter

**A bandwidth-constrained solver with an explicit memory hierarchy.**

The project started as an adaptive recurrent transformer experiment. The current
formulation is narrower and more falsifiable: memory placement is part of the
solver state rather than a penalty attached after the model has already moved
the bytes.

```text
s_k = (h_k, c_k, p_k)
```

- `h_k`: residual stream / scratchpad
- `c_k`: residency set, which cold operator slices are currently hot
- `p_k`: previous prediction distribution

The legal moves are `spin`, `fault`, `write`, and `halt`. Depth is the
number of spins the policy bought. A KV write is legal once; cold program memory
is fetched only on a fault.

## First gate

Before training a learned policy, run one systems experiment on the same decode
tokens:

1. stock forward
2. naive K=4 recurrence with KV per pass
3. K=4 recurrence with one KV write
4. one-KV-write recurrence with the cold MLP faulted only when the prior
   prediction KL is large enough

The only promotion metric is:

```text
(stock cross-entropy - schedule cross-entropy) / measured DRAM bytes
```

Schedule 4 must beat schedule 3 **after including the gate's own memory traffic**.
If it does not, the residency thesis fails and we stop before adding more learned
machinery.

## Why batching changes

Per-token depth is not enough. Tokens are batched by predicted fault so one cold
slice requested by many tokens becomes one packed operation over those tokens.
The runtime should optimize coalesced misses, not merely token-level FLOPs.

## Code

`src/depth_router/solver.py` contains the state machine primitives, one-write
invariant, KL value signal, and coalesced cold-fault execution.

`docs/BANDWIDTH_SOLVER.md` defines the falsification boundary.

The original pointer-chasing harness remains as a baseline for recurrence
experiments, but it is no longer the main architectural thesis.

## Quick start

    python -m pip install -e ".[dev]"
    pytest

## Principle

> Route bytes through computation, not computation around a byte penalty.

## Status

Research prototype. No bandwidth, latency, or quality win is claimed until the
hardware-counter experiment passes.

The isolated minimum-sufficient parallel-inference falsification run is documented
in [`RESEARCH_REPORT.md`](RESEARCH_REPORT.md). Its measured CUDA and simulated
photonic outputs are kept separate; the combined promotion gate did not pass.

The larger causal working-set follow-up is documented in
[`PREDICTIVE_WORKING_SET_REPORT.md`](PREDICTIVE_WORKING_SET_REPORT.md). It retained
the negative neuron-level result, skipped the fused benchmark because quality
failed, and recorded the first structured operator-page pivot.

The RTX 4070 SUPER fused-kernel follow-up is documented in
[`FUSED_SPARSE_SWIGLU_REPORT.md`](FUSED_SPARSE_SWIGLU_REPORT.md). A prepacked 75%
Triton kernel beats dense execution only as a non-deployable upper bound; no
sub-full policy passes the quality gate, while graph-captured full-width dense
execution remains the fastest exact implementation.

The separate final-MLP expert-construction diagnostic is preregistered in
[`EXPERT_CONSTRUCTION_PREREGISTRATION.md`](EXPERT_CONSTRUCTION_PREREGISTRATION.md).
Its capability, conditional selection, and execution findings are separated in
[`EXPERT_CONSTRUCTION_REPORT.md`](EXPERT_CONSTRUCTION_REPORT.md). Exhaustive subset
oracles are diagnostic only, not deployment routers or measured savings.

The separate frozen-C-29 recognition follow-up is documented in
[`EXPERT_RECOGNITION_REPORT.md`](EXPERT_RECOGNITION_REPORT.md) and its
[`prospective protocol`](EXPERT_RECOGNITION_PREREGISTRATION.md). Fresh OOD
confirmation coverage was 75.78%, below the 80% engineering gate (95% context
interval 70.69%-81.25%). It stopped before selector training or timing; the
interval spans the threshold, so this is not a resolved population-level failure.
Prior negative findings and the project's hardware-counter gate remain unchanged.

The bounded nested-granularity follow-up is preregistered in
[`EXPERT_GRANULARITY_PREREGISTRATION.md`](EXPERT_GRANULARITY_PREREGISTRATION.md).
It compares the original 28-mask portfolio with calibration-selected 28-mask
portfolios drawn from finer balanced subdivisions, keeping 75% active width.
The full 1,820-mask oracle is diagnostic only; this cycle excludes router
training and runtime claims.
