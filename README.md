# DepthRouter

**Hardware-aware adaptive recurrent inference.**

DepthRouter is an experimental research project for routing computation through transformer depth rather than treating model depth as fixed.

The core hypothesis is that a model can reuse a transformer block recurrently, allocate different numbers of passes to different inputs, and optimize quality against compute, memory traffic, and latency.

A simplified objective is:

    min task_loss + lambda * FLOPs + mu * bytes_moved + gamma * latency

subject to a minimum quality target.

## First milestone

The initial harness compares:

- fixed unshared transformer depth
- fixed shared recurrence
- adaptive shared recurrence
- shared recurrence with pass-specific low-rank residual adapters

The first task is random pointer chasing, chosen because the target naturally requires serial state updates.

## Quick start

    python -m pip install -e ".[dev]"
    pytest
    python experiments/toy_pointer_chase.py --device auto

The experiment prints JSON containing accuracy, parameter count, parameter storage, and mean logical passes.

## Important claim boundary

The v0 adaptive router masks halted samples semantically but does not yet compact the active batch. Fewer logical sample-passes therefore do not imply proportional wall-clock or energy savings.

Likewise, the analytic weight-traffic model is only a bound. HBM traffic, cache residency, arithmetic intensity, and throughput must be measured with hardware profilers before making systems claims.

See docs/EXPERIMENT_PLAN.md for the falsifiable evaluation plan.

## Research questions

1. Can shared recurrent depth match or beat an unshared stack at a fixed parameter budget?
2. Can adaptive halting reduce average passes without materially reducing task quality?
3. Do pass-specific low-rank adapters recover useful depth-specific capacity?
4. Can fixed-point or residual-based acceleration reduce physical block evaluations?
5. When does weight reuse improve arithmetic intensity or memory traffic on real hardware?

## Principle

> Route computation, not just tokens.

## Status

Bootstrap research harness. No performance claims are established yet.
