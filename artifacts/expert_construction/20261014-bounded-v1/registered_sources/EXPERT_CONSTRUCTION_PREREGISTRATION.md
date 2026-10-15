# Final-MLP expert construction: preregistration

This is a separate diagnostic cycle, based on commit
`ec696fddabc59c51292ba1e8f500c9000a50c79b` (PR 16); no prior gate or negative
finding is changed. Execution creates an immutable dated preregistration with
config, manifest, provenance, and executable source SHA256s **before loading the
model, fitting assignments, or examining held-out model outputs**. The bounded
cycle has one dataset and one recipe; no tuning after held-out inspection.

## Question and scope

Does a useful six-of-eight computation exist, can a causal selector recognize it,
and only then does the actual implementation save time or physical memory traffic?
The first question does not answer the other two. All original Qwen weights are
frozen; no correction network is added. Model/revision, seeds, and budgets are in
`configs/expert_construction.json`. Only the final MLP changes: 4,864 neurons,
eight equal experts of 608, all 28 six-expert combinations. Gate/up rows and down
columns always travel together. Width 3,648 is exactly 75%.

The local single GPU is capped at 14,400 elapsed seconds starting before model
load; this conservative bound includes CPU waits, capture, construction and audit.
No more seeds, optimization extensions, or threshold changes are permitted.
Incomplete stages and their available records remain labeled incomplete.

## Four constructions

- A: seeded balanced random assignment.
- B: cosine clustering of absolute calibration activation traces, a seeded
  32-dimensional Rademacher sketch, and eight exact-capacity balanced rounds.
- C/D: 38,912 FP32 assignment logits; the same random partition A initializes
  logits as one-hot times 0.1 plus seeded Gaussian noise with standard deviation
  0.01. Log-domain Sinkhorn has ten iterations; global descending greedy rounding
  enforces exact capacities. Straight-through hard assignments are used.
  Adam, learning rate 0.03, 100 updates, 16 calibration states/update, and four
  uniformly sampled combinations/update. Temperature decreases linearly 1 to 0.2.
  Each token contributes its lowest loss among the four sampled combinations.
  The final iterate is frozen; there is no checkpoint or seed selection.

C minimizes MLP output MSE divided by the reference output mean square per state.
D minimizes KL(dense || candidate) on the actual final RMSNorm and **entire
151,936-token vocabulary**, after BF16 residual addition. Greedy agreement is a
separate metric. Both objectives share initialization, sampled states, sampled
candidate IDs, optimizer settings, update count, and final-iterate selection.
Their shared-input hashes and per-step losses/gradient norms are retained.

The hard forward physically packs the selected original rows/columns and computes
BF16 SwiGLU. The assignment backward uses the cached dense intermediate activation
times soft mask and FP32 down projection, with
`hard.detach() + (surrogate - surrogate.detach())`. Thus the objective value is
the actual hard candidate value, while the gradient is explicitly a surrogate.
This is bounded heuristic assignment optimization, not globally optimal partition
search and not training a deployment router.

## Data, prefixes, and independence

Use the frozen 2,304-context human-written Dolly manifest, pinned source revision,
license attribution, exclusion audit and hashes in `data/expert_audit/`.
Targets are 1,024/256/512/512 calibration/development/matched/OOD contexts.
OOD reserves summarization and creative-writing task categories. Each split has
12.5% enterprise-related instructions identified by the documented keyword screen;
these are not audited production enterprise workloads. Source responses are unused.
Historical inspected prompts are excluded by the frozen audit. Source documents,
normalized instruction families and high lexical-similarity families do not cross
splits. Upstream article IDs are absent, so semantic/article-level independence is
not established; do not overstate this guarantee.

Use the official chat template, left padding and left truncation at 512 tokens;
record truncations. Dense greedy continuations supply all paired prefixes, with
no sampling, no ground-truth responses, and no sparse feedback into future states.
Evaluate positions 1, 3, 7, 11 (zero-based response-token positions), each executed
as one-token cached decoding in batch 16. Stop including a context after EOS;
retain counts and IDs for contexts with fewer/no eligible states. This avoids
prefill-versus-decoding GEMM shape artifacts. Save full prefix token IDs.

Point estimates describe fractions of evaluated token states. Confidence
intervals resample **whole contexts**, keeping all positions and paired seeds
together (2,000 bootstrap draws, seed 17). They are conditional on these three
seeds; they do not estimate variation across unseen seeds. No independent-token
uncertainty claims. Report C versus D paired differences, including all three
seeds averaged within each bootstrap draw. Intervals are descriptive 95%
percentile intervals without multiple-comparison adjustment.

## Numerical controls and raw evidence

Use actual BF16 residual addition, original RMSNorm and full vocabulary head.
For each frozen partition, compare cached candidate evaluation to execution inside
the complete model on the first 16 calibration contexts at continuation position 1,
for every one of the 28 candidates plus all-eight. Require exactly equal logits
for that matched-shape replay, otherwise stop and resolve the implementation.
Every captured dense state separately verifies the cached residual/norm/head
against the actual complete-model logits.

Main candidate layout packs a selected union in original neuron order; all-eight
therefore preserves original ordering. Measure that control across every state,
rather than assuming exactness. Additionally measure an expert-major full-width
permutation control for every partition/state to reveal reduction-order effects.
The main all-eight control must achieve >=99% greedy agreement and mean KL<0.01
before interpreting sparse fidelity. The permuted diagnostic does not replace the
main layout with an unvalidated permutation.

Each compressed NPZ contains every state-by-candidate KL, greedy token, hidden
MSE, candidate margin, maximum logit error, dense token/margin, context/position,
selected experts, partition, original-order and permuted all-eight controls.
JSONL state rows carry prefixes/task/provenance keys. No failed arm is omitted.
Large deterministic activation caches remain local with hashes and can be
reconstructed from the pinned model and retained prefixes; all scalar raw results
and construction records are versioned.

## Phase 1 decision

Local cheap acceptance means identical greedy token **and** KL<0.01.
Choose one fixed candidate per partition on development only: maximum acceptance
coverage, then minimum mean KL, then canonical candidate index. Freeze this with
partitions before any held-out model access. The diagnostic cheap-only oracle
prefers a greedy-agreeing candidate, then minimum KL, then index; if none agrees,
it minimizes KL. Cheap-or-dense and fixed-cheap-or-dense oracles choose cheap only
on locally acceptable states. They are explicitly nondeployable.

Ideal active width = 1 - 0.25 * cheap coverage. This is an arithmetic proxy, not
measured speed, bytes or bandwidth. On development and both held-out splits:

1. Cheap-candidate coverage must be >=80% (ideal average active width <=80%).
2. Portfolio ideal active width must improve by >=5 percentage points over the
   development-selected fixed candidate with the same oracle dense option.

Report each arm/seed point pass and whether both lower confidence bounds meet
their thresholds. These are engineering criteria, not previously measured facts.
The prospective Phase 2 construction family is selected on development by largest
mean portfolio advantage, then coverage, then arm name; freeze it before held-out
access. Require that family's three seeds to pass all three splits. Also report
every other arm/seed's gates. Failure stops router fitting; distinguish insufficient
cheap candidates from insufficient portfolio advantage, without generalizing beyond
these partitions, data, and fitting budget.

## Conditional phases 2 and 3

Only if Phase 1 passes, fit x-only linear or one-hidden-layer 64-unit selectors,
200 Adam updates with batch 64 and learning rate 0.001 per seed. Train with
multi-label BCE retaining all acceptable calibration candidate labels. Runtime
chooses the largest predicted acceptable score if it exceeds the threshold,
otherwise original dense. Choose architecture/threshold on development from the
fixed grid in config, maximizing cheap fraction subject to >=99% greedy agreement
and mean KL<0.01, with smaller architecture then higher threshold as tie breaks.
Compare random, fixed, and a matched binary fixed-versus-dense selector. Freeze
before held-out selector evaluation. No teacher/head pass, omitted activations,
answers, or oracle search is available to a runtime selector. Report false cheap
choices, dense frequency, task and margin strata. Confidence is not a certificate.

Only a selector passing quality on development and both held-out splits may be
timed. Use fixed packed weights, count duplicated candidate memory, include all
selection/synchronization/launches and dense routing, compare graph-captured dense
at equal workload, and measure end-to-end cached decode. Record physical DRAM
counters or their exact access failure, retaining unknown when unavailable.
Free generation/task success is a separate later measurement, only after quality.
Removing 25% of one of 24 MLPs is just 1.04% total MLP arithmetic before overhead.
If prerequisites fail, costs and task success remain null, not estimated gains.

## Research foundations and distinctions

[DOT-MoE](https://proceedings.mlr.press/v306/bamba26a.html), with
[primary method text](https://arxiv.org/html/2606.01666v1), motivates balanced
Sinkhorn assignment and straight-through discrete optimization. Its joint
assignment/router alignment is different from this final-layer, assignment-only,
sampled-candidate objective comparison. No faithful reproduction or novelty claim.

[DynaMoE](https://arxiv.org/abs/2502.12325) motivates predicting sufficient
computation. Its nested widths and training approach differ from these disjoint
experts and frozen original model weights.

[Goal-oriented reduction](https://arxiv.org/abs/1603.00336) motivates distinguishing
output fidelity from intermediate reconstruction. Its linear-system certificates
are not guarantees for a nonlinear Qwen selector.
