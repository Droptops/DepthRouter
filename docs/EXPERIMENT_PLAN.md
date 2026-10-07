# Experiment plan

DepthRouter should be falsified with a hardware measurement before adding more
architecture.

The current target is not "does looping work?" It is:

> Does explicit cache-aware scheduling improve predictive gain per byte moved?

## State under test

The solver state is

```text
s_k = (h_k, c_k, p_k)
```

with actions

```text
spin | fault(j) | write | halt
```

The cache state `c_k` includes residency and one-shot KV-write status.

## One-box experiment

Use one pinned mid-stack block from one small model. Hold model weights, token
stream, batch shape, decode length, seeds, and evaluation examples constant.

Run four schedules:

1. **stock**: ordinary forward;
2. **naive-k4**: the selected block is recurrent for four passes;
3. **kv-once-k4**: four passes, but KV is written once and later passes are read-only;
4. **gated-cold-k4**: schedule 3, with a cold MLP slice skipped unless the previous
   spin's logit movement exceeds a fixed threshold.

Do not train another router or add another adapter before this experiment.

## Measurements

For every schedule record:

- cross-entropy in nats/token;
- HBM / memory-controller bytes read and written;
- L2 hit rate;
- total bytes moved;
- decode latency;
- token count and exact schedule parameters.

Profiler-derived bytes are authoritative. Analytic parameter-size estimates are
diagnostic only.

## Primary metric

Let pass 1 be the common reference point. For schedule `sigma`:

```text
eta_sigma =
    (CE_after_pass1 - CE_sigma)
    / max(B_sigma - B_after_pass1, epsilon)
```

Units: nats / byte.

Report the distribution across examples as well as the aggregate value.

The cache-aware residency thesis passes the first gate only if
`gated-cold-k4` produces a strictly better quality-per-byte frontier than
`kv-once-k4` after including the gate's own memory traffic.

If not, stop and record the falsification.

## State-sufficiency check

The measured transition costs should compose.

For any recorded execution trajectory `gamma`, compute:

```text
A(gamma) = B_hw(gamma) - sum_k b(s_k, a_k)
```

where `B_hw` is profiler-measured traffic and `b(s_k, a_k)` is the local
transition-cost model.

A systematic non-zero `A` means `c_k` is incomplete. Add only the missing
hardware state required to explain the discrepancy, then rerun the same
experiment.

This is the only "anomaly" concept used by the project: a failure of local
accounting to reproduce a global measured quantity. It is a mathematical
consistency test, not a claim that the model is a physical TQFT.

## Batch scheduling requirement

When multiple tokens predict the same `fault(j)`, batch them so the slice is
transferred once and reused.

Report both:

- per-token requested faults;
- unique/coalesced cold transfers.

The second number is the systems target.

## Decision rule

**Pass:** schedule 4 improves nats/byte over schedule 3 by a repeatable margin,
while the accounting residual is within profiler noise.

**Fail:** schedule 4 does not improve the frontier after gate traffic, or the
claimed byte savings disappear in profiler counters.

Only after a pass should the next experiment train the policy over
`{spin, fault(j), write, halt}`.

## Claim boundary

A pass establishes only a model-and-hardware-specific result. It does not prove
that recurrence is universally bandwidth-efficient or that the same placement
policy transfers to a different memory hierarchy.
