# Next experiments

## 1. Freeze and independently confirm the tangent oracle

**Prediction:** The final-layer 30% local-logit-tangent oracle retains mean KL below 0.01 and top-1 agreement of at least 99% on a provenance-tracked corpus with longer rollouts.

**Falsification:** Either mean KL reaches 0.01, top-1 agreement falls below 99%, or task-level free-running correctness violates its preregistered non-inferiority margin on the full evaluation set.

The expanded 90-prompt enterprise-synthetic result reached 100% top-1 with mean KL 0.00919, but used dense-generated trajectories and a privileged current-state Jacobian. Use task-level parsers, dense and sparse free-running rollouts, FP32 and FP16, three seeds, and uncertainty intervals.

## 2. Causal tangent-selector gate

**Prediction:** A causal selector using only information available before final-layer MLP execution can approximate the frozen tangent-oracle set closely enough to preserve behavior.

**Falsification:** Held-out top-1 agreement is below 99%, mean KL is at least 0.01, or selector/cache metadata removes the analytic payload margin.

The current distilled selector is falsified at its tested settings: 30% and 40% payloads reached only 78% and 83% top-1. Improve the representation or use a coarser structured unit; do not tune against the test prompts.

## 3. Conditional physical implementation

**Prediction:** A causal, structured tangent selector that clears the quality gate can retain enough whole-model savings after coalescing and dispatch overhead to improve measured latency or physical traffic.

**Falsification:** The causal frontier cannot exceed a preregistered analytic margin large enough to cover overhead, or a fused kernel fails to reduce measured physical bytes and latency against dense execution.

Do not implement a kernel for the current neuron-page formulation. Keep physical bytes `null`; require the oracle and causal-selector gates before writing or profiling selective execution.
