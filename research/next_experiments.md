# Next experiments

## 1. Freeze and independently confirm the tangent oracle

**Prediction:** The final-layer 30% local-logit-tangent oracle retains mean KL below 0.01 and top-1 agreement of at least 99% on a provenance-tracked corpus with longer rollouts.

**Falsification:** Either mean KL reaches 0.01, top-1 agreement falls below 99%, or task-level free-running correctness violates its preregistered non-inferiority margin on the full evaluation set.

The expanded 90-prompt enterprise-synthetic result reached 100% top-1 with mean KL 0.00919, but used dense-generated trajectories and a privileged current-state Jacobian. Use task-level parsers, dense and sparse free-running rollouts, FP32 and FP16, three seeds, and uncertainty intervals.

## 2. Block-structured causal selector gate

**Prediction:** A causal selector using only information available before final-layer MLP execution can choose a coarser block-structured unit and directly minimize behavioral/tangent reconstruction loss closely enough to preserve behavior at no more than 90% execution.

**Falsification:** Held-out top-1 agreement is below 99%, mean KL is at least 0.01, execution exceeds 90%, or selector/cache metadata removes the analytic payload margin.

**Outcome (2026-10-12): passed on the synthetic enterprise workload, not yet generalized.** A rank-256 selector over contiguous 32-neuron blocks, trained directly against normalized-hidden reconstruction, top-logit KL, and top-1 margin, selected 136/152 blocks: `89.4737%` execution plus `2.0532%` resident selector metadata. On a deduplicated, shuffled 160-train/45-test split it reached `1.0` top-1, `0.00644` mean KL, and `1.0` exact five-token free-running prompt agreement over 225 held-out states. A second training seed reached `1.0`/`0.00641`; a different shuffled split reached `0.99556`/`0.00714`. Static-energy and random-fixed controls failed. The same family reached only `0.92` top-1 and `0.01782` KL on the more varied default 20/20 corpus, so this is a workload-specific causal gate pass, not a general deployment claim.

Freeze the passing configuration before any new held-out corpus. Do not tune on the external test set. Keep the earlier neuron-set family closed: rank 128 at 30% reached only 81.5% top-1; continuous marginal-utility distillation reached 78.5%; a 30%-teacher safety sweep required 100% execution; direct 90%-teacher training reached only 83.5%.

## 3. Conditional physical implementation

**Prediction:** A causal, structured tangent selector that clears the revised quality gate can retain enough whole-model savings after coalescing and dispatch overhead to improve measured latency or physical traffic.

**Falsification:** The causal frontier cannot exceed a preregistered analytic margin large enough to cover overhead, or a fused kernel fails to reduce measured physical bytes and latency against dense execution.

The synthetic enterprise causal gate now clears the analytic prerequisite, but the varied default corpus does not. Keep physical bytes and latency `null`. First test the frozen selector on an independently sourced corpus and quantify whole-model savings; only then implement a fused block-sparse kernel if the quality and metadata-inclusive margin survive.
