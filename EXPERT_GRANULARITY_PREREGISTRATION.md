# Frozen-parent granularity and equal-budget portfolio audit

Prospective bounded capability study, following the failed C-29 OOD confirmation.
The prior result is preserved: 194/256 OOD states (75.78%) were acceptable; 61 of
the 62 failures had no KL-qualified candidate among the original 28 masks. This
motivates testing mask expressivity; it does not establish a causal explanation.

## Question and controls

Does finer grouping improve held-out acceptable-state coverage while retaining
exactly the same width and exactly the same 28 deployment choices? Freeze Qwen
2.5-0.5B-Instruct revision `7ae557604adf67be50417f59c2c2f167def9a775`, all weights,
the final-layer C-29 parent partition, BF16/SDPA, dense reference prefixes, batch
16, max prompt 512, and continuation positions 1/3/7/11. No output correction,
weight fitting, router training, kernel tuning, or inference timing occurs.

Independently split each 608-neuron parent into two balanced 304-neuron children
using NumPy PCG64 seeds 17, 29, 43, ordered parents and sorted original neuron IDs.
Child label is `2*parent+child`. Every candidate retains 12/16 children: exactly
3,648/4,864 neurons (75%). Enumerate all 1,820 masks in lexicographic order. The
28 original six-parent masks are exact members. Pack every candidate in ascending
original neuron order; never substitute a masked FP32 dense reconstruction.

The primary comparison is original 28 versus calibration-selected 28. The full
1,820-mask teacher oracle is diagnostic only and is not an equal-budget result.
Random refinement seeds are repetitions, not a grid from which to choose a winner.

## Data and fitting boundaries

Use only the 1,600 previously unopened contexts from the recognition allocation:
calibration 768, development 128, matched held-out 512, and OOD held-out 192.
Preserve prompt bytes, IDs, split/task/stratum quotas, source revision, and all
historical exclusions. Quarantine all 128 already-inspected confirmation contexts.
Unopened means no prior model-output evaluation, not guaranteed semantic novelty.
OOD creative-writing/summarization tasks remain absent from calibration/development.

For each seed, calibration alone greedily chooses exactly 28 distinct masks:
maximize newly acceptable states; break ties by lower mean full-vocabulary KL over
all calibration states, then lower canonical mask ID. Continue to 28 even after
coverage saturates. This is explicit discrete portfolio fitting, not zero fitting.
Seal all choices before development. Development independently selects a best fixed
member for original28, each selected28, and each diagnostic full1820 portfolio:
maximum acceptable coverage, then lower mean KL, then canonical ID. Seal these
comparators before either held-out split. The old fixed candidate is not the primary
comparator. No portfolio reselection, threshold tuning, or test-set fitting.

## Unchanged gates and added paired improvement requirement

A local candidate is acceptable iff greedy token equals dense AND full-vocabulary
KL(dense || candidate) is strictly below 0.01. Coverage is the fraction of captured
states with at least one acceptable member. Ideal active width is `1-0.25*coverage`;
portfolio advantage is `0.25*(coverage-fixed_coverage)`. This excludes all overhead.
Each selected28 seed must achieve coverage >=80% and advantage >=5 percentage
points on development, matched held-out, and OOD held-out. No seed dropping.

Additionally, paired selected28-minus-original28 OOD coverage must have a positive
lower 95% confidence bound after averaging the three seeds per state and resampling
whole contexts 2,000 times with seed17. This added rule cannot relax any existing
gate. Report rescue, loss and net changes for each seed and the seed mean, both
held-outs and development. Bootstrap intervals retain all token positions per
resampled context; they do not quantify variation over an unseen population of
refinement seeds. Existing absolute gates are point gates; report whether their
individual lower bounds also clear thresholds without substituting that label for
the registered decision. No multiplicity-adjusted population claim is made.

## Integrity, execution budget, and stopping

Before model load, snapshot/hash config, data, source, parent, refined partitions,
candidate schema, and parity sample IDs. Evaluate original28 once per split and
require every corresponding full1820 column in every seed to match all raw scalar
and token measurements exactly. Full1820 coverage cannot lose an original-covered
state. Both original-order full-width and fixed C-29-parent-permuted full-width
controls are measured; the former must be exactly dense. On the first calibration
decode batch, compare cached final-layer substitution with actual full-model
substitution for all 28 inherited masks, first/last new masks, and 16 seeded sampled
new masks per refinement, plus both full-width controls. Exact logits are required
for cached/full-model replay (permuted-vs-dense is separately reported).

Use at most 28 resident packed banks, candidate-tiled evaluation, and lossless
candidate chunks of at most256 columns. Preserve all measured candidate scalars,
state/prefix metadata, masks, selections, controls, failures, hashes and environment.
Dense activation caches are local ignored files, with hashes in capture metadata.
Verify all original model tensors remain unchanged at exit. A single-GPU four-hour
wall ceiling starts before model load and includes capture, evaluation, selection,
serialization and reporting computations. Timeout/error means incomplete, not a
negative or passed study; preserve partial observations. No smaller substitute run.

End this cycle after capability, whether pass or fail. A pass would justify a new,
separately authorized causal-recognition test; it does not authorize one here.
Inference latency, physical DRAM traffic, generation/task quality, and net savings
remain unmeasured/null. Global bandwidth-solver promotion remains false. Keep prior
negative results unchanged; publish a draft stacked PR, never merge automatically.
