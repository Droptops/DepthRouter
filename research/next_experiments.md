# Next experiments

## 1. Freeze and independently confirm the tangent oracle

**Prediction:** The final-layer 30% local-logit-tangent oracle retains mean KL below 0.01 and top-1 agreement of at least 99% on a provenance-tracked corpus with longer rollouts.

**Falsification:** Either mean KL reaches 0.01, top-1 agreement falls below 99%, or task-level free-running correctness violates its preregistered non-inferiority margin on the full evaluation set.

The expanded 90-prompt enterprise-synthetic result reached 100% top-1 with mean KL 0.00919, but used dense-generated trajectories and a privileged current-state Jacobian. Use task-level parsers, dense and sparse free-running rollouts, FP32 and FP16, three seeds, and uncertainty intervals.

## 2. Block-structured causal selector gate

**Prediction:** A causal selector using only information available before final-layer MLP execution can choose a coarser block-structured unit and directly minimize behavioral/tangent reconstruction loss closely enough to preserve behavior at no more than 90% execution.

**Falsification:** Held-out top-1 agreement is below 99%, mean KL is at least 0.01, execution exceeds 90%, or selector/cache metadata removes the analytic payload margin.

The current neuron-set family is closed at useful widths. Rank 128 at 30% reached only 81.5% top-1; a continuous marginal-utility objective reached 78.5%; a 30%-teacher safety sweep needed the 100% identity control to cross 99%; direct 90%-teacher training reached only 83.5%. Do not spend more runs scaling the same binary/rank head or tune against the test prompts.

## 3. Conditional physical implementation

**Prediction:** A causal, structured tangent selector that clears the revised quality gate can retain enough whole-model savings after coalescing and dispatch overhead to improve measured latency or physical traffic.

**Falsification:** The causal frontier cannot exceed a preregistered analytic margin large enough to cover overhead, or a fused kernel fails to reduce measured physical bytes and latency against dense execution.

Do not implement a kernel for the current neuron-page formulation. Keep physical bytes `null`; require the oracle and causal-selector gates before writing or profiling selective execution.
