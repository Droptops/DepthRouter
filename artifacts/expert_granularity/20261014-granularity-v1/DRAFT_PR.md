## Summary

Run one preregistered frozen-parent granularity study, stacked on the existing
recognition study. No changes to model weights, 75% active final-MLP width,
28-choice primary portfolio budget, or local quality gates.

- Split frozen C-29's eight 608-neuron parents into sixteen balanced 304-neuron
  children, independently for seeds 17/29/43.
- Measure all 1,820 twelve-of-sixteen masks offline; select exactly 28 using
  calibration only. Fresh development-selected fixed comparators are sealed
  before held-outs. The full 1,820-choice oracle is diagnostic only.
- Use 1,600 previously unopened contexts; quarantine all 128 failed confirmation
  contexts. No router training or inference timing in this cycle.

## Result

The registered capability gate passed for every seed on development and both
held-outs. All individual lower confidence bounds also cleared the unchanged
absolute coverage and ideal-width-advantage thresholds.

| Portfolio | Matched coverage | OOD coverage |
| --- | --- | --- |
| Original 28 | 85.855% | 84.896% |
| Selected 28, seed 17 | 88.261% | 87.891% |
| Selected 28, seed 29 | 88.261% | 88.542% |
| Selected 28, seed 43 | 87.967% | 88.151% |

Paired seed-mean OOD coverage improvement: **+3.299 percentage points**, 95%
whole-context bootstrap interval **+1.649 to +4.861**. OOD rescues per seed were
37/40/41 states, with 14/12/16 losses. The full 1,820-mask diagnostic reached
95.052%-95.833%; that larger search is not the equal-budget result.

The comparison uses the original portfolio on these same new contexts, not the
earlier 75.78% confirmation score. The old negative result is preserved. Finer
masks expose coverage headroom; this does not establish the sole cause of the
earlier small-sample failure or solve OOD coverage.

## Evidence and validation

- 6,367 captured states across all four splits; every measured candidate scalar,
  token, mask, state/prefix record and selection retained in lossless hashed chunks.
- Exact inherited-mask equality for all 12 seed/split pairs; original-order dense
  controls exact; parent-permuted controls reported separately.
- Sampled cached/full-model parity: 144 layouts x 16 prefixes, exact logits.
- All 291 original tensor hashes unchanged; 29 registered execution/data sources
  unchanged. Completed in 2,499.684 seconds within the 14,400-second ceiling.
- 336 tests and repository-wide Ruff pass, including a fresh checkout with Linux
  CPU-only execution. Explicit checkout attributes preserve sealed historical
  CRLF source/manifest bytes. Post-run verifier recomputes the complete decision.

See `EXPERT_GRANULARITY_REPORT.md`, `EXPERT_GRANULARITY_PREREGISTRATION.md`, and
`artifacts/expert_granularity/20261014-granularity-v1/`.

## Boundaries

Capability only. No causal selector trained, no latency/throughput/physical DRAM
or free-generation quality measurement, and no net-savings claim. Global promotion
remains false. A recognition follow-up requires separate authorization.

Draft, stacked on `codex/causal-expert-recognition`. Do not merge automatically or
move the earlier research branches.
