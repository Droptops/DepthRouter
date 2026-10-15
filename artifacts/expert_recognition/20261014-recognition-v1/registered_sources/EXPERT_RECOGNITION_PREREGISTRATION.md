# Frozen C-29 expert recognition: prospective protocol

This is a new, bounded recognition study, not a change to the completed expert-construction experiment. That study's all-three-construction-seed failure and its skipped recognition/timing stages remain unchanged. Prior held-out data are now discovery data and are excluded in full.

## Question and fixed artifact

Can a small, causal selector recognize a useful six-of-eight configuration, at the required fidelity, better than an equally trained binary fixed-subset-versus-dense selector? If so, does it actually save time after selector, synchronization, dispatch, and fallback overhead?

Freeze C-29 from commit `a803bf8a967e9437344e897a9bc8677f9f21ffea`: it has the largest **old development** portfolio advantage among the twelve partitions. No fresh result can change this choice. Reuse its old development fixed subset, canonical index 8, experts `[0,1,2,4,6,7]`. Config records byte hashes of both inputs. No repartitioning, assignment training, output-gap fitting, original-weight updates, additional layers, or seed search.

Pinned Qwen2.5-0.5B-Instruct revision `7ae557604adf67be50417f59c2c2f167def9a775`, BF16, SDPA. The final MLP has 4,864 neurons in eight balanced experts; every cheap action executes 3,648 neurons. Sorted original-neuron order, actual prepacked BF16 projections, real residual addition/final RMSNorm/full vocabulary head. Dense fallback is the original MLP, not an eight-expert surrogate. All-eight sorted and permuted controls are retained, plus exact cached-evaluator/full-model parity checks. Weight hashes must remain unchanged.

## Fresh data and budget

Use the same pinned Dolly source and strict source-document/instruction-template/lexical exclusions against all 6,163 historical prompts and all 2,304 prior expert-study contexts. No answer field is used. Freeze manifest/provenance before any model output. New partitions are confirmation 128 (64 matched + 64 OOD), calibration 768, development 128, matched held-out 512, OOD held-out 192; each split/confirmation stratum is 12.5% enterprise-keyword-screened. This is not a production enterprise benchmark.

These sizes were reduced **before model access**, after finding only 41 fresh OOD enterprise candidates under the unchanged exclusions. The original proposed 512 confirmation/1,024 calibration/256 development/512 matched/512 OOD recipe was infeasible. No exclusion or quota was loosened. Smaller confirmation and OOD samples yield wider uncertainty. Category holdout is not proof of semantic independence; the source lacks stable article identifiers and the enterprise OOD remainder is creative-writing-heavy.

Capture dense-prefix cached decoding at batch 16, sampled continuation positions 1,3,7,11, left truncation at 512 prompt tokens, official chat template. Stop recording a context after EOS; retain all exclusions/counts/prefixes. Use all sampled valid states without outcome-dependent filtering. Total single-GPU wall-time ceiling is four hours, including loading, CPU waits, training, evaluation and conditional timing. Budget exhaustion is incomplete, not failure or success. Existing prior results cannot consume or extend this budget.

## Stage 1: independent capability confirmation

Before fitting any selector, enumerate all 28 six-of-eight candidates on the fresh confirmation split. A candidate is locally acceptable iff its greedy token equals dense and KL(dense || candidate) <0.01. For **each** of the matched and OOD confirmation strata require:

- At least 80% of states have at least one acceptable cheap configuration.
- Ideal active-width advantage over the frozen fixed-subset-plus-dense oracle is at least 0.05, equal to 0.25 times the acceptable-coverage difference.
- Actual-layout all-eight numerical control has >=99% agreement and mean KL <0.01.

These are point-estimate engineering gates; 95% whole-context bootstrap intervals (2,000 draws, seed17) are reported separately and are not silently substituted for the gates. Failure stops the cycle without switching partition, relaxing thresholds, or consuming calibration/development/held-out model outputs. Confirmation is never used to train or select a router, even after a pass.

## Stage 2: recognition, not teacher-assisted selection

Freeze architectures: linear, or one 64-wide SiLU hidden layer. Runtime input is only final-MLP input `x`. FP32 normalization uses calibration population mean/std (floor1e-5), and its conversion/cost counts at runtime. No dense teacher, omitted activation, logits, answer, task label, residual base, oracle or future token enters the selector.

Portfolio uses 28 sigmoid outputs trained on **all** acceptable labels with mean binary cross-entropy. The binary baseline predicts acceptability of only frozen candidate8. Each pair shares normalization, initial backbone and corresponding output-head row, calibration minibatch schedule and optimizer budget. Heads differ in size (28 vs1), so this is equal-backbone/equal-update-budget, **not exact parameter-count equality**; disclose counts.

Train both architectures for seeds17,29,43: 200 Adam steps, batch64, lr0.001, final iterate only, no early stopping/augmentation/loss reweighting. Record every loss, schedule/hash, parameter count and all checkpoints. The two models have the same number of optimizer updates but not identical arithmetic cost.

For each seed and policy family, select architecture and threshold only on fresh development. Threshold grid `[.5,.7,.8,.9,.95,.98,.99,.995,1.0]`;1.0 explicitly means all-dense. Choose maximum cheap fraction satisfying >=99% greedy agreement and mean KL <0.01; tie by smaller architecture, then higher threshold. Portfolio chooses maximum sigmoid output (lowest index on ties), then accepts or falls back. Seal choices, checkpoints and source hashes before any fresh held-out model access.

Report each seed on development and both held-out sets: greedy agreement, mean/p95/p99/max KL, cheap fraction, false-cheap fraction (of all states and conditional on cheap), dense fraction, ideal active width and whole-context bootstrap intervals. Report teacher-only portfolio/fixed oracles, always-fixed, original dense, random configurations on identical cheap positions, and separately randomized cheap positions at exactly matched cheap count. Oracles are not deployment results. Report paired portfolio-minus-binary differences, including a seed-mean comparison conditioned on these three seeds, not uncertainty over future seeds.

Recognition promotion requires **all three router seeds** to meet point fidelity gates and have positive cheap use on development/matched/OOD. All-dense is not recognition success. Additionally, portfolio must earn its extra complexity: the 95% context-bootstrap lower bound of seed-mean cheap-fraction advantage over binary must be >0 on **both** held-out sets. No seed is selected using held-out performance. Seed17 is the prospective runtime primary;29/43 are replications. Binary can independently proceed to timing if its three-seed fidelity/nontrivial-use gates pass, even if portfolio does not.

## Stage 3: actual runtime quality, then economics

Only families passing Stage2 are runtime candidates. Recollect genuine batch1 full-model dense-prefix states on development and both held-out sets with the same prompts/positions; do not merely reshape batch16 activations. Check the actual frozen seed17 decision and routed packed execution, true BF16 residual/norm/head, and exact cached/full-model parity. Require >=99% agreement, mean KL <0.01 and positive cheap use on each runtime split before benchmarking that family. No refit or threshold adjustment follows these checks.

Measure paired identical state/prefix sequences chosen by deterministic context order, not routing outcome. Primary isolated final-MLP timing uses 64 matched-heldout states; end-to-end cached one-token decode uses the first8 of the deterministic sequence. Compare eager dense and CUDA-graph dense against the entire routed path; include FP32 normalization, selector, host synchronization, threshold/dispatch, all MLP launches, and dense fallback. Full-model measurement includes attention, residuals, final norm/head and cache updates; prefill is outside scope. Use20 randomly interleaved paired timing blocks,2 warmups,2 repetitions, wall and CUDA timing. These are fixed-prefix one-token replay workloads, not free-generation throughput. Report measured route frequencies and sequence identities.

Prepack once, no per-token weight gathers. Portfolio stores28 overlapping candidate banks; binary stores one. Count duplicate weights, selector/index metadata and memory costs. Resident bytes and ideal FLOPs/width are not physical DRAM traffic. Attempt Nsight counters on a passing frozen runtime only; preserve exact tool/access failure and leave traffic unknown if unmeasurable. No simulated/photonic estimate becomes measured GPU savings.

Economic success requires a paired timing95% CI lower speedup >1 against **both** dense baselines, including graph dense, at isolated and end-to-end scopes. Full-model graph/parity failure makes economic success unestablished, not permission to compare only eager dense. Report individual scope results even when overall success fails. No task-quality claim without a separate free-generation/task study; this cycle makes none.

Full-model graph timing uses a fixed-length StaticCache, whereas the full held-out runtime audit uses DynamicCache. Before timing the eight selected prefixes, require their static dense greedy tokens to match the stored dynamic references and their actual static routed aggregate fidelity to meet the same >=99%/<0.01 gates. A backend mismatch or failed subset gate prevents full-model timing/promotion. Disclose that this eight-prefix backend check is small; it does not replace the independent held-out audit or establish general StaticCache deployment quality.

## Integrity and stop rules

Snapshot this protocol, config, manifest/provenance, frozen partition/fixed-choice inputs and all execution dependencies before model loading. Retain raw candidate matrices, state metadata, selected actions/scores, checkpoints, training logs, runtime results and failure records. Source/choice seal changes abort access to held-out data. Preserve negatives. Do not restart with a different partition, seed, threshold or data after a failed gate. No automatic merge or modification of prior PRs.
