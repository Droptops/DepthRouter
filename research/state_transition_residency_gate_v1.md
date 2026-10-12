# State-transition-aware residency: preregistered gate v1

## Research question
Can a cheap, causal detector of *changes* in predictive working-set demand reduce physical memory transfers without violating model-quality constraints, compared with simply retaining the previous token's resident pages?

This branch deliberately starts from the earlier temporal-address and logit-tangent experiments. It does not assume that oracle-selected neurons are available to a deployed router.

## Arms (same model, prompts, token stream, cache budget and page size)
1. Dense stock forward (quality, latency and physical-traffic reference).
2. Static frequency-based page residency, fitted only on training prompts.
3. Previous-token page reuse with a deterministic replacement policy.
4. Causal transition-triggered residency, using only prior/current cheap observable features, with detector compute and metadata accounted for.
5. Oracle future-demand residency: diagnostic ceiling only, **never** counted as a deployable policy.

Start with Qwen2.5-0.5B-Instruct layers 18 and 23, plus an untouched held-out prompt set. Freeze all hyperparameters before final evaluation. Test several page sizes and cache budgets on calibration data, then lock one configuration.

## Primary decision metric
At fixed hardware, batch size, precision and quality constraints, compare **measured off-chip bytes per generated token**, including page fetches, metadata, detector traffic, KV traffic, and fallback costs. Report median and p95 latency/token as co-primary operational checks. Quality gate: mean KL to dense < 0.01 and top-1 agreement >= 99% on held-out tokens; also report exact free-running rollout agreement.

The transition policy promotes only if it passes quality and improves physical traffic versus previous-token residency **after all overhead**. Repeat with at least three seeds and report uncertainty; no promotion based on idealized traffic alone.

## Staged execution
A. **Causal trace replay**: record actual selected pages and only legal past/current features; compare static, previous-token and transition policies. Account for page replacement, detector bytes and faults. Replay numbers are *analytical*, not physical HBM measurements. Reject label leakage, train/test prompt overlap and oracle-conditioned decisions.
B. **Quality evaluation**: run actual sparse/fallback logits and free-running decode; report mean KL, top-1, rollout, detector false-negative rate, missed-demand fraction, resident fraction and cold bytes/token.
C. **Physical validation**: implement a real selective/faulting kernel or offload boundary, measure physical transfer bytes using profiler/counters where available, and end-to-end latency. If all weights remain resident on GPU, label the experiment a logical cache simulation, not an HBM reduction.

## Current checkpoint (2026-10-12)

Stage B has a workload-specific pass for a new structured unit. A causal rank-256 head uses only the incoming final-layer MLP state and selects 136 of 152 contiguous 32-neuron blocks before conceptual gate/up/down execution. On a deduplicated, shuffled 160-train/45-test enterprise-synthetic split it reached `1.0` top-1, `0.00644` mean KL, and `1.0` exact five-token free-running agreement over 225 held-out states at `89.4737%` execution. Resident selector parameters add `2.0532%` of final-layer MLP payload, leaving an `8.4731%` layer-relative analytic margin. A second training seed and second shuffled split also passed; static and random controls did not.

This does not satisfy Stage C. The varied default prompt corpus failed (`0.92` top-1, `0.01782` KL), the enterprise data come from one synthetic generator family, and the PyTorch evaluation hook still executes the dense MLP before replacing its output. Physical bytes and latency therefore remain `null`. Freeze this selector before independent-corpus evaluation; do not tune against the new test set.

## Falsification
- No improvement over previous-token reuse on disjoint held-out prompts at matched budget.
- Transition detector overhead eliminates traffic or latency gain.
- Any quality gate fails, or gains vanish under free-running decode.
- Oracle-only advantage without a causal deployable policy.
- Physical traffic counters disagree with modeled byte savings.

## Evidence lineage
Earlier prompt-structure PRs #11/#12 and learned-address PR #10 remain separate research evidence. Do not merge or claim deployment based on their oracle-selected demand sets. Existing temporal traces may seed the *design*, but final numbers require new held-out measurement.

## Reporting schema
Record model revision, commit SHA, device/GPU, precision, kernel implementation, page size, resident capacity, split IDs, seeds, profiler settings, per-arm quality metrics, physical bytes/token (or null), modeled bytes/token (separate), detector overhead, p50/p95 latency, cold misses and confidence intervals. A null physical metric means no measured bandwidth result.

## Merge policy
Draft PR until CI, reviews and replicated gates pass. Do not change production routing defaults.
