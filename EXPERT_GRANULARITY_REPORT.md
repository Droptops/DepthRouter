# Frozen-parent granularity: equal 28-choice portfolio audit

Run: `artifacts/expert_granularity/20261014-granularity-v1`. Preregistered UTC: `2026-10-15T05:06:06.516867+00:00`.

## Outcome

**The preregistered equal-budget capability gate passed.** Every selected 28-mask portfolio
passed the absolute gates on every evaluation split, and the seed-mean paired OOD coverage
gain had a positive lower confidence bound. This is a capability result, not causal routing
or measured savings. A separate recognition study requires new authorization.

On the same fresh OOD states, original-28 coverage was 84.896%; selected-28 coverage ranged from 87.891% to 88.542%.
Seed-mean paired gain: 3.299 pp [1.649, 4.861] (95% CI).

The earlier failed confirmation set is not the comparative baseline. These are paired
original-versus-refined measurements on this cycle's same fresh held-out states.
The original portfolio itself also exceeds 80% OOD coverage on this fresh set.
A positive paired gain supports finer-mask headroom; it does not establish granularity as the
sole cause of the earlier confirmation failure, or show that OOD generalization is solved.

The earlier C-29 confirmation failure remains unchanged. All 128 previously inspected confirmation
contexts were quarantined; none was reused in this cycle. No partition winner was selected
from held-out outcomes, no seed was discarded, and no original model weight was fitted.

## Controlled question and scope

Does finer neuron grouping improve held-out acceptable-state coverage while keeping the same
active width and exactly the same number of deployed choices? The comparisons are:

| Arm | Grouping | Available choices | Role |
| --- | --- | --- | --- |
| Original 28 | 6 of 8 frozen C-29 parents | 28 | Equal-budget reference |
| Selected 28 | 12 of 16 balanced child experts | 28 | Primary calibration-fitted portfolio |
| Full 1,820 oracle | 12 of 16 balanced child experts | 1820 | Diagnostic upper bound only |

Model: `Qwen/Qwen2.5-0.5B-Instruct` at `7ae557604adf67be50417f59c2c2f167def9a775`; bfloat16, sdpa.
All weights and the final-layer C-29 parent partition remain frozen. Each 608-neuron parent
is randomly split into two 304-neuron children using NumPy PCG64 seeds 17, 29 and 43.
Every candidate executes exactly 3,648 of 4,864 neurons (75%). All 28 original masks are exact
members of each 1,820-mask universe. Real packed BF16 gate/up/down projections, residual
addition, final normalization and full-vocabulary head are used; no masked FP32 shortcut.

Execution environment: NVIDIA GeForce RTX 4070 SUPER; PyTorch 2.7.1+cu128; CUDA 12.8; Transformers 4.57.1.

## Data, fitting and uncertainty

The 1,600-context manifest is a byte-preserving derivative of previously unopened recognition
partitions: 768 calibration, 128 development, 512 matched held-out, 192 OOD held-out.
IDs, prompt strings, JSONL lines, original order, task assignments and 12.5% enterprise-keyword
quotas are unchanged. The full prior 1,728-row and filtered 1,600-row exclusion audits were
rerun against historical data and the quarantined confirmation contexts. Source responses are not used.

Freshness means the specifically pinned prior run produced no model outputs on the retained
contexts. It does not prove global source novelty or semantic independence. Dolly's held-out
creative-writing/summarization categories and enterprise keyword screen are not a production
enterprise benchmark; source-article IDs are unavailable and OOD enterprise rows are creative-writing-heavy.

Calibration greedily selects exactly 28 distinct masks per seed by newly covered states, then
lower mean full-vocabulary KL, then canonical mask ID. This is explicit discrete fitting, not
zero fitting. Portfolios are sealed before development. Development independently freezes the
best fixed member of original 28, selected 28 and diagnostic 1,820 portfolios before either
held-out split. The old study's fixed mask is not this study's primary fixed comparator.

Dense-prefix continuation positions: `[1, 3, 7, 11]`; batch 16;
max prompt length 512 tokens; stop recording after individual EOS.
All captured valid states are retained. The 95% intervals resample whole contexts 2,000 times
with seed 17, keeping their token positions together. Seed-mean contrasts retain all three
seeds per draw and do not quantify uncertainty over future refinement seeds. No multiplicity-adjusted
population claim is made.

| Captured split | Requested contexts | Evaluated contexts | States | Zero-state contexts | Truncated prompts |
| --- | --- | --- | --- | --- | --- |
| calibration | 768 | 768 | 3052 | 0 | 0 |
| development | 128 | 128 | 511 | 0 | 0 |
| heldout_matched | 512 | 512 | 2036 | 0 | 0 |
| heldout_ood | 192 | 192 | 768 | 0 | 0 |

## Primary equal-budget result

Local acceptability requires exact dense greedy-token agreement AND KL(dense||candidate)<0.01.
Each selected seed must reach at least 80% coverage and at least 5 percentage points of ideal
active-width advantage over its development-frozen fixed-member-plus-dense oracle on all
three evaluation splits. Point engineering gates and statistically resolved passes are separate.
A failed resolved-pass check means that a pass is not established; it does not by itself
establish population failure. An interval spanning its threshold remains unresolved.

| Portfolio | Seed | Split | Contexts | States | Coverage [95% CI] | Fixed coverage [95% CI] | Ideal width advantage [95% CI] | Point | Resolved pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Original 28 | fixed parent | development | 128 | 511 | 83.757% [80.625, 86.838] | 54.207% [49.805, 58.431] | 7.387 pp [6.507, 8.350] | pass | pass |
| Selected 28 | 17 | development | 128 | 511 | 89.432% [86.667, 92.157] | 55.773% [51.081, 60.078] | 8.415 pp [7.451, 9.424] | pass | pass |
| Selected 28 | 29 | development | 128 | 511 | 89.041% [86.106, 91.602] | 54.990% [50.487, 59.375] | 8.513 pp [7.485, 9.590] | pass | pass |
| Selected 28 | 43 | development | 128 | 511 | 87.867% [84.539, 90.998] | 55.577% [50.781, 60.001] | 8.072 pp [7.094, 9.136] | pass | pass |
| Original 28 | fixed parent | heldout_matched | 512 | 2036 | 85.855% [84.291, 87.451] | 54.371% [52.264, 56.404] | 7.871 pp [7.350, 8.370] | pass | pass |
| Selected 28 | 17 | heldout_matched | 512 | 2036 | 88.261% [86.702, 89.779] | 53.880% [51.746, 55.982] | 8.595 pp [8.088, 9.091] | pass | pass |
| Selected 28 | 29 | heldout_matched | 512 | 2036 | 88.261% [86.732, 89.710] | 55.206% [53.083, 57.332] | 8.264 pp [7.721, 8.808] | pass | pass |
| Selected 28 | 43 | heldout_matched | 512 | 2036 | 87.967% [86.450, 89.450] | 56.041% [53.861, 58.260] | 7.981 pp [7.484, 8.481] | pass | pass |
| Original 28 | fixed parent | heldout_ood | 192 | 768 | 84.896% [82.292, 87.370] | 51.693% [48.304, 55.339] | 8.301 pp [7.487, 9.115] | pass | pass |
| Selected 28 | 17 | heldout_ood | 192 | 768 | 87.891% [85.417, 90.107] | 52.083% [48.307, 55.732] | 8.952 pp [8.040, 9.864] | pass | pass |
| Selected 28 | 29 | heldout_ood | 192 | 768 | 88.542% [86.198, 90.755] | 52.995% [49.219, 56.641] | 8.887 pp [7.975, 9.832] | pass | pass |
| Selected 28 | 43 | heldout_ood | 192 | 768 | 88.151% [85.677, 90.495] | 52.474% [48.828, 56.123] | 8.919 pp [7.975, 9.863] | pass | pass |

Ideal active width = 1 − 0.25 × coverage; advantage = 0.25 × (coverage − fixed coverage).
These are arithmetic proxies excluding routing, dispatch, duplicate storage and every hardware overhead.

### Paired selected-28 minus original-28

Rescue means a state newly has an acceptable choice; loss means a previously covered state
loses all acceptable choices in the selected portfolio. Net gain = rescue − loss.

| Split | Seed | Rescue [CI] | Loss [CI] | Net coverage gain [CI] | Ideal width improvement [CI] |
| --- | --- | --- | --- | --- | --- |
| development | 17 | 7.241 pp [5.273, 9.375] | 1.566 pp [0.586, 2.745] | 5.675 pp [3.516, 8.023] | 1.419 pp [0.879, 2.006] |
| development | 29 | 6.458 pp [4.688, 8.611] | 1.174 pp [0.391, 2.153] | 5.284 pp [3.125, 7.632] | 1.321 pp [0.781, 1.908] |
| development | 43 | 5.479 pp [3.718, 7.255] | 1.370 pp [0.394, 2.353] | 4.110 pp [1.961, 6.250] | 1.027 pp [0.490, 1.563] |
| development | seed mean | 6.393 pp [4.817, 8.153] | 1.370 pp [0.655, 2.214] | 5.023 pp [3.268, 6.915] | 1.256 pp [0.817, 1.729] |
| heldout_matched | 17 | 4.175 pp [3.346, 5.061] | 1.768 pp [1.227, 2.360] | 2.407 pp [1.379, 3.482] | 0.602 pp [0.345, 0.871] |
| heldout_matched | 29 | 3.978 pp [3.157, 4.858] | 1.572 pp [1.034, 2.113] | 2.407 pp [1.423, 3.399] | 0.602 pp [0.356, 0.850] |
| heldout_matched | 43 | 3.782 pp [2.999, 4.612] | 1.670 pp [1.127, 2.253] | 2.112 pp [1.134, 3.142] | 0.528 pp [0.284, 0.786] |
| heldout_matched | seed mean | 3.978 pp [3.307, 4.729] | 1.670 pp [1.274, 2.077] | 2.308 pp [1.488, 3.172] | 0.577 pp [0.372, 0.793] |
| heldout_ood | 17 | 4.818 pp [3.255, 6.510] | 1.823 pp [0.911, 2.865] | 2.995 pp [1.042, 4.948] | 0.749 pp [0.260, 1.237] |
| heldout_ood | 29 | 5.208 pp [3.646, 6.901] | 1.562 pp [0.781, 2.474] | 3.646 pp [1.823, 5.599] | 0.911 pp [0.456, 1.400] |
| heldout_ood | 43 | 5.339 pp [3.776, 7.161] | 2.083 pp [1.172, 3.125] | 3.255 pp [1.302, 5.212] | 0.814 pp [0.326, 1.303] |
| heldout_ood | seed mean | 5.122 pp [3.689, 6.597] | 1.823 pp [1.215, 2.431] | 3.299 pp [1.649, 4.861] | 0.825 pp [0.412, 1.215] |

The added paired improvement rule requires the seed-mean OOD net coverage lower bound to be
strictly greater than zero. It cannot relax the per-seed absolute gates.

| Gate | Result |
| --- | --- |
| All three selected portfolios, all splits: point gates | pass |
| All three selected portfolios, all splits: resolved lower bounds | pass |
| Seed-mean paired OOD gain: positive lower bound | pass |
| Registered capability decision | pass |

### Larger-search diagnostic, not the equal-budget result

| Refinement seed | Split | Full-oracle coverage [CI] | Its fixed coverage | Choices |
| --- | --- | --- | --- | --- |
| 17 | development | 95.695% [93.542, 97.461] | 56.556% | 1820 |
| 17 | heldout_matched | 94.597% [93.529, 95.620] | 55.403% | 1820 |
| 17 | heldout_ood | 95.052% [93.229, 96.615] | 55.859% | 1820 |
| 29 | development | 94.912% [92.759, 96.857] | 54.990% | 1820 |
| 29 | heldout_matched | 94.204% [93.071, 95.252] | 55.206% | 1820 |
| 29 | heldout_ood | 95.182% [93.490, 96.745] | 52.995% | 1820 |
| 43 | development | 94.716% [92.549, 96.667] | 56.556% | 1820 |
| 43 | heldout_matched | 94.646% [93.615, 95.672] | 54.617% | 1820 |
| 43 | heldout_ood | 95.833% [94.271, 97.266] | 54.297% | 1820 |

The full 1,820-mask oracle is teacher-assisted and has about 65 times as many choices as either
primary portfolio. Its inclusion of all original masks makes nondecreasing full-oracle coverage
a numerical nesting control, not evidence of an equally cheap or causal deployed selector.

### Frozen selections

| Seed | Selected choices | Calibration coverage | Selected-28 fixed ID | Full-oracle fixed ID |
| --- | --- | --- | --- | --- |
| 17 | 28 | 91.448% | 1327 | 1761 |
| 29 | 28 | 91.547% | 1761 | 1761 |
| 43 | 28 | 91.579% | 1778 | 1683 |

Original-28 development-frozen fixed canonical ID: `1773`.
All 28 selection steps, marginal gains, KL tie-breaks, candidate IDs and seals are retained in
`portfolio_selection.json` and `fixed_selection.json`. Per-mask arrays are preserved in lossless chunks.

## Numerical integrity

Independent CPU-only post-run verification passed: 29 registered sources, 16 matrix archives and 34,942,096 candidate-state cells checked.
Saved evidence: `postrun_integrity_verification.json`. Selections, metrics, paired intervals,
gates and inherited equality were independently recomputed.
Data exclusions and quarantine were also independently recomputed.

Cached evaluator versus actual full-model substitution: 144 retained layout checks, 16 calibration prefixes per layout. Exact logits in every saved check: `True`.
The prospective sample includes inherited masks, first/last new masks, seeded random new
masks and both full-width controls. This is sampled full-model validation, not replay of every
new mask in the whole model.

Inherited original-mask raw equality was checked for 12 seed/split combinations.
Every saved equality assertion passed: `True`.
The offline verifier independently compares exact neuron indices, all candidate scalar/token
fields and both controls for those inherited columns, then checks full-oracle coverage nesting.

| Split | Original-order agreement | Original-order KL | Parent-permuted agreement | Parent-permuted KL |
| --- | --- | --- | --- | --- |
| calibration | 100.000% | 0 | 99.803% | 0.00015100517 |
| development | 100.000% | 0 | 100.000% | 0.00016006314 |
| heldout_matched | 100.000% | 0 | 99.853% | 0.00016834079 |
| heldout_ood | 100.000% | 0 | 99.870% | 0.00016359296 |

Original-order full width must be exactly dense. Parent-permuted reduction is measured
separately and is not assumed bitwise equal to the original order. Perfect greedy agreement
can coexist with nonzero KL. Original tensor hashes before and after execution are retained.

## Cost boundaries and next-step scope

This cycle ends at capability. No causal router was trained and no inference timing was performed.
Latency, physical DRAM traffic, throughput, free-generation/task quality and net savings are
**unmeasured**, not zero. Global bandwidth-solver promotion remains false even if capability passes.
At most 28 packed candidate banks are resident during offline evaluation; the larger search
cost is research cost, not an inference-speed measurement.

Removing 25% of one of 24 MLPs is only about 1.04% of aggregate MLP arithmetic before overhead.
Nothing here establishes that a selector can recognize the good subsets cheaply enough to
yield real savings. A capability pass would justify proposing a separately authorized
recognition experiment, not automatically running one.

Recorded single-GPU wall elapsed: 2499.684 seconds; ceiling 14400 seconds. This includes research execution/serialization,
not a latency benchmark. The run uses one bounded budget; it is not silently replaced by a smaller audit.

## Reproduction and evidence

Registered source/config/data snapshots, three refined partitions, prospective parity sample IDs,
state-prefix records, every measured candidate value, selections, seals and failures are retained.
Candidate matrices use hashed metadata/layout archives and lossless chunks of at most 256 columns.
Local activation caches may be ignored by Git; capture metadata records their hashes and reconstruction provenance.

Repository attributes preserve the sealed historical source line endings on fresh cross-platform checkouts.

```powershell
python scripts/verify_expert_granularity.py --run-dir artifacts/expert_granularity/20261014-granularity-v1 --validate-data
python experiments/expert_granularity_report.py --run-dir artifacts/expert_granularity/20261014-granularity-v1
.\scripts\run_expert_granularity.ps1  # new output directory; never overwrite the recorded run
```

The verifier uses no language model or GPU. It recomputes partition splits, all 28-mask selections,
development comparators, exact inherited equality, metrics, paired intervals and gates from raw
artifacts. It checks saved before/after tensor hashes and event/seal consistency, not new hardware
measurements or cryptographic proof of real-world chronology.
