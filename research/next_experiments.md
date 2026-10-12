# Next experiments

## 1. Independent quality confirmation

**Prediction:** After freezing candidates, layer 12 at 90% and layer 23 at 95% can retain the quality gate on a prompt-disjoint, provenance-tracked corpus.

**Falsification:** Either mean KL reaches 0.01, top-1 agreement falls below 99%, or task-level free-running correctness violates its preregistered non-inferiority margin on the full evaluation set.

The 300-prompt run was exploratory and included the original screen distribution, so it is not a final held-out result. Use longer rollouts, task-level parsers, FP32 and FP16, three seeds, and uncertainty intervals.

## 2. Multi-layer behavioral frontier

**Prediction:** Modest exact-current-activation omissions spread across multiple layers may preserve the quality gate while exceeding a 1% global MLP-payload reduction.

**Falsification:** No tested multi-layer group exceeds 1% ideal global MLP-payload reduction while meeting mean KL < 0.01 and top-1 agreement >= 0.99.

The granularity experiment is complete: every layout and seed touched 100% of pages at page sizes from 8 through 128 neurons. One-neuron pages retained only the trivial 90%/95% payload fractions while requiring 4,378/4,621 requests per token. Freeze the multi-layer candidates before the expanded evaluation.

## 3. Conditional selective kernel and profiling

**Prediction:** Only if the multi-layer frontier survives overhead modeling, a real selective kernel can reduce measured bytes/token or latency at matched quality.

**Falsification:** Supported profiler counters show no physical-traffic improvement, end-to-end latency regresses after selector/metadata overhead, or any required quality threshold fails.

Keep physical bytes `null` until this kernel exists. Transition-selector work remains paused because the current representation failed its adjacent-over-control locality prerequisite.
