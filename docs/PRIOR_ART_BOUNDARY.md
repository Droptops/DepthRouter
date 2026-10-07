# Prior-art boundary

DepthRouter should not claim novelty for the ingredients below. The research
question is the intersection and the systems evidence.

## Known neighboring ideas

- **Contextual sparsity / dynamic parameter selection.** Deja Vu predicts
  input-dependent subsets of MLP parameters and attention heads and executes
  only those subsets.
  - Liu et al., ICML 2023: https://proceedings.mlr.press/v202/liu23am.html

- **Hot/cold neural placement.** PowerInfer exploits activation locality and
  places frequently activated neurons close to the accelerator while treating
  other neurons as cold.
  - Song et al., SOSP 2024 / arXiv: https://arxiv.org/abs/2312.12456

- **Predictive / causal-state compression.** Causal rate-distortion theory
  compresses histories while retaining information about futures.
  - Marzen & Crutchfield: https://arxiv.org/abs/1412.2859

- **Nested uncertainty across adaptive exits.** Early-exit networks have been
  paired with anytime-valid confidence sequences specifically to make
  prediction sets consistent across exits.
  - Jazbec et al., UAI 2024:
    https://proceedings.mlr.press/v244/jazbec24a.html

- **Conformal deferral / cascades.** Conformal set size has been used as a rule
  for escalating LLM requests to more expensive inference tiers.
  - Conformal Cascade, 2026: https://arxiv.org/abs/2607.25018

- **Cache-aware expert routing.** Recent MoE work explicitly couples expert
  routing and cache placement. A 2026 pre-registered study also reports an
  important negative result: stronger learned locality can carry a quality tax.
  - Cacheable by Design?: https://arxiv.org/abs/2608.18261

## The gap this repository is testing

The working hypothesis is narrower:

> Treat **future operator-page demand** as the random variable predicted by the
> recurrent solver; use calibrated prediction sets as physical residency sets;
> learn the physical page layout from co-demand traces; schedule tokens by page
> fault; and optimize the resulting system against **measured bytes**, not a
> compute proxy.

The candidate contribution is therefore the composition:

```text
recurrent belief refinement
    -> calibrated operator-page set
    -> predictive physical working set
    -> co-demand-aware page layout
    -> fault-grouped batch scheduling
    -> profiler-measured nats / byte
```

No individual arrow is presumed novel.

A literature search has not established that this exact composition is new.
Before any "first" or "novel" publication claim, perform a dedicated scholarly
and patent search around all of the following terms:

- conformal operator / weight / expert residency;
- confidence-sequence cache management;
- learned page layout from co-demand or co-posterior traces;
- neural semantic MMU / page-fault prediction;
- causal-state cache placement;
- rate-distortion-aware neural memory hierarchy.

## Strong falsification boundary

The idea is not interesting merely because a toy posterior can be packed.

The systems claim survives only if, on a real model and target device:

1. a calibrated operator set is substantially smaller than the full cold set;
2. coverage of the oracle-required operator set meets its declared risk level;
3. recurrent refinement reduces that set before the cold fetch;
4. page-fault grouping reduces *measured* transfers;
5. the quality loss remains inside a predeclared bound; and
6. the bytes saved exceed the bytes spent on prediction/refinement.

The sixth condition is the break-even test. If it fails, the extra intelligence
in the memory manager is overhead rather than an inference win.
