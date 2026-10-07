# Experiment plan

DepthRouter should earn its claims empirically. The first harness is intentionally
small and separates architectural questions from hardware claims.

## Hypotheses

### H1: parameter efficiency

At equal width and nominal depth, a shared recurrent transformer should use
substantially fewer parameters than an unshared stack.

This is a bookkeeping claim and should hold by construction.

### H2: iterative-task quality

On tasks that naturally require serial state updates, recurrent depth should
recover useful accuracy as the number of passes increases.

The first task is random pointer chasing: given a functional graph, a start node,
and a hop count, predict the node reached after K hops.

### H3: low-rank depth specialization

Pass-specific low-rank residual adapters should recover some of the expressive
capacity lost by exact weight sharing while remaining much smaller than cloning
the full block.

### H4: adaptive depth

A halting rule should reduce average logical sample-passes at a controlled
quality loss.

The v0 router uses normalized hidden-state residual:

    rho_k = RMS(h_(k+1) - h_k) / max(RMS(h_k), eps)

and halts when rho_k is at or below the configured threshold.

This is a baseline heuristic, not a claim that hidden-state convergence is the
optimal halting signal.

## Required comparisons

Run at least these arms:

1. independent stacked blocks, fixed depth
2. one shared block, fixed recurrence
3. one shared block plus pass-specific low-rank adapters
4. shared variants with adaptive halting enabled at evaluation

Two fairness regimes matter:

- iso-width / iso-evaluation-count: exposes the parameter-memory advantage
  of sharing.
- iso-parameter: widen the shared model until parameter counts are similar,
  then compare quality and throughput.

Do not collapse these into one comparison.

## Metrics

Record:

- task accuracy
- total parameters
- parameter storage bytes
- mean logical sample-passes
- physical dense block evaluations
- examples/second
- peak accelerator memory
- wall-clock latency percentiles

Later hardware work should add profiler-derived:

- HBM bytes read/written
- L2 hit rate
- achieved FLOP/s
- arithmetic intensity

Analytic traffic bounds in the Python package are estimates only. They are not
evidence of actual residency.

## Claim boundary

The current adaptive router masks halted examples semantically, but it does not
compact the active batch. Therefore lower logical sample-passes do not yet imply
proportional wall-clock or energy savings.

A later sparse/compacted execution path is required before making that claim.

## Promotion gates

Before calling the idea promising:

- recurrent fixed-depth accuracy must show a repeatable advantage on at least
  one explicitly iterative task under an appropriate budget comparison;
- adaptive routing must produce a Pareto point: fewer logical passes for a
  bounded, predeclared accuracy loss;
- results must repeat across at least 5 seeds;
- hardware residency claims require profiler evidence on a named device.

Before calling it a systems win:

- active-batch compaction or an equivalent sparse execution mechanism must show
  measured latency/throughput improvement;
- memory-traffic improvements must appear in profiler counters, not only in the
  analytic model.
