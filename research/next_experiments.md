# Next experiments

## 1. Independent quality confirmation

**Prediction:** After freezing candidates, layer 12 at 90% and layer 23 at 95% can retain the quality gate on a prompt-disjoint, provenance-tracked corpus.

**Falsification:** Either mean KL reaches 0.01, top-1 agreement falls below 99%, or task-level free-running correctness violates its preregistered non-inferiority margin on the full evaluation set.

The 300-prompt run was exploratory and included the original screen distribution, so it is not a final held-out result. Use longer rollouts, task-level parsers, FP32 and FP16, three seeds, and uncertainty intervals.

## 2. Temporal locality and compiled layout

**Prediction:** Although adjacent-state locality was falsified for the original layout, a calibration-only neuron permutation may reduce held-out contiguous page demand relative to both the unpermuted layout and random permutations.

**Falsification:** The learned layout does not reduce held-out page fraction against both controls without changing dense behavior.

Measure layers 12 and 23 at quality-passing activation fractions. Learn the permutation only on calibration prompts and keep logical page results separate from physical traffic.

## 3. Conditional causal selector with fallback

**Prediction:** Only if a new representation first shows positive adjacent-over-control locality, a selector using previous hidden state and recent page history can reduce charged transfers relative to previous-token reuse while a dense fallback preserves the quality gate.

**Falsification:** Matched-budget transfer cost does not improve after detector and metadata overhead, or any required quality threshold fails.

The current representation failed its locality prerequisite, so this is paused. Calibrate thresholds on a dedicated split and do not use current-token dense activations for pre-forward selection.
