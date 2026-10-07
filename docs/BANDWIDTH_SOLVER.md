# Bandwidth-constrained solver

DepthRouter's state is now

```
s_k = (h_k, c_k, p_k)
```

where `h_k` is the residual/scratch state, `c_k` is the residency set, and
`p_k` is the previous prediction distribution.

The legal moves are finite: `spin`, `fault`, `write`, and `halt`.
Depth is the number of spins the policy buys. A KV update is a one-time write.
A cold operator is paid for only on a fault.

This replaces the earlier framing where bytes were a penalty attached to an
otherwise unconstrained hidden-state recurrence.

## Coalescing is part of the problem

Tokens must be grouped by predicted fault, not merely by sequence position. If
30 tokens request the same cold slice, the executor should issue one packed
operation over those tokens. Charging 30 independent faults recreates the
bandwidth behavior of a naive loop.

## Falsification before training

Do not train another router, halting head, LoRA bank, or accelerator until one
measurement passes.

On the same decode tokens, profile:

1. stock forward
2. naive K=4 recurrence with KV per pass
3. K=4 recurrence with one KV write
4. one-KV-write recurrence where the cold MLP is faulted only when the previous
   prediction KL exceeds a threshold

The gate's own traffic belongs inside the measured region.

The decision metric is:

```
(stock cross-entropy - schedule cross-entropy) / measured DRAM bytes
```

Schedule 4 must beat schedule 3 on this metric. Otherwise the placement thesis
has failed its first systems test and further policy training is premature.

The uploaded Spark toolkit contains the concrete Nsight Compute harness for this
measurement.
