# Next experiments

## 1. Independent quality confirmation

**Prediction:** After freezing candidates, layer 12 at 90% and layer 23 at 95% can retain the quality gate on a prompt-disjoint, provenance-tracked corpus.

**Falsification:** Either mean KL reaches 0.01, top-1 agreement falls below 99%, or task-level free-running correctness violates its preregistered non-inferiority margin on the full evaluation set.

The 300-prompt run was exploratory and included the original screen distribution, so it is not a final held-out result. Use longer rollouts, task-level parsers, FP32 and FP16, three seeds, and uncertainty intervals.

## 2. Corrected research-job reruns

**Prediction:** Some historical batched research results will not reproduce after evaluating real next-token positions instead of right-padding positions.

**Falsification:** Every affected job reproduces its prior qualitative conclusion and passes its identity controls under left padding.

The global multi-layer frontier is complete and negative above 1% ideal savings. Commit `7cd3cf1` fixes 17 additional harnesses with the same risk. Treat pre-fix outputs as provisional until rerun.

## 3. Conditional structured-sparsity pivot

**Prediction:** A coarser structured unit or trained causal sparsity mechanism may create a quality-compatible working set without thousands of requests per token.

**Falsification:** A dense-activation oracle for the proposed unit cannot exceed 1% ideal global payload reduction at the quality gate, or a causal selector cannot reproduce the oracle on held-out data.

Do not implement a physical kernel for the current neuron-page formulation. Keep physical bytes `null`, and require the oracle and causal-selector gates before any future kernel work.
