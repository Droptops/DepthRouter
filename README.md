# DepthRouter

**Hardware-aware adaptive recurrent inference.**

DepthRouter is an experimental research project for routing computation through transformer depth rather than treating model depth as fixed.

The core hypothesis is that a model can reuse a transformer block recurrently, allocate different numbers of passes to different inputs/tokens, and optimize quality against compute, memory traffic, and latency.

A simplified objective is:

```text
min  task_loss + λ * FLOPs + μ * bytes_moved + γ * latency
```

subject to a minimum quality target.

Initial research questions:

1. Can shared recurrent depth match or beat an unshared stack at a fixed parameter budget?
2. Can adaptive halting reduce average passes without materially reducing task quality?
3. Do pass-specific low-rank adapters recover useful depth-specific capacity?
4. Can fixed-point / residual-based acceleration reduce physical block evaluations?
5. When does weight reuse improve arithmetic intensity or memory traffic on real hardware?

## Status

Bootstrap phase. No performance claims are established yet.

The first milestone is a reproducible toy harness that compares:
- fixed unshared depth,
- fixed shared recurrence,
- adaptive shared recurrence,
- shared recurrence with pass-specific low-rank adapters.

## Principle

> Route computation, not just tokens.
