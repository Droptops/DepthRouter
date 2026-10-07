# DepthRouter

**A bandwidth-constrained recurrent solver with explicit memory state.**

DepthRouter is an experimental research project for treating recurrent inference
as a control problem over both model state and memory placement.

The primary state is:

```text
s_k = (h_k, c_k, p_k)
```

where `h_k` is solver state, `c_k` is cache / residency state, and `p_k` is
the current next-token distribution.

The solver chooses among a finite set of moves:

```text
spin | fault(j) | write | halt
```

Depth is the number of `spin` moves purchased before `halt`. A `fault(j)`
loads a cold operator slice. `write` is the one-shot KV update. The goal is not
"fewer layers"; the goal is more predictive gain per byte moved.

## Primary falsification target

The first systems experiment uses one pinned mid-stack block on one small model
and measures four schedules on the same token stream:

1. stock forward
2. naive recurrence at K=4
3. K=4 with KV written once
4. schedule 3 with a cold MLP slice faulted only when logit movement justifies it

The primary metric is incremental cross-entropy improvement per incremental byte
after the first useful pass:

```text
eta = (CE_after_pass1 - CE_schedule)
      / (bytes_schedule - bytes_after_pass1)
```

If schedule 4 does not beat schedule 3 after including the gate's own traffic,
the cache-aware residency thesis is considered falsified on that
hardware/model pair.

## Why memory placement is part of the state

A post-hoc byte penalty is not enough. If identical hidden states incur different
cost depending on what is resident, they are different solver states.

The local cost model is checked against profiler bytes using an accounting
residual:

```text
A(gamma) = B_hw(gamma) - sum_k b(s_k, a_k)
```

A persistent non-zero residual means the state is missing some hardware variable
such as residency, eviction, coalescing, or graph-capture behavior. The response
is to fix the state description, not add another mechanism.

See `docs/MEMORY_STATE_SOLVER.md` for the formulation and
`docs/EXPERIMENT_PLAN.md` for the measurement protocol.

## Existing prototype

The initial Python harness still contains the earlier recurrent / stacked toy
baselines. They are useful sanity checks, but they are no longer the primary
systems claim.

```text
python -m pip install -e ".[dev]"
pytest
python experiments/toy_pointer_chase.py --device auto
```

## Principle

> Placement is part of the dynamics.

## Status

Research prototype. No hardware-residency performance claim is established until
the profiler experiment passes.
