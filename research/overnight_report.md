# DepthRouter overnight research report

## Executive finding

**The original paging hypothesis remains rejected, but a causal block address plus fallback now clears the strict gate on an independent public prompt distribution.** The repaired replay shows that the original layer-23 configuration (top 10% of activation-weight contributions grouped into contiguous 128-neuron pages) touches all 38 pages on every observed token. No constrained working set exists under that definition, and static, previous-token, transition, and noncausal-oracle policies tie once all initialization and prefetch transfers are charged.

The controlled locality test also went the wrong way: sparse page sets at adjacent decode steps were less similar than page sets from different prompts of the same task at the same step. A calibration-only learned layout reduced held-out page demand at unusably sparse fractions, but produced no reduction at the behaviorally viable settings. A finer granularity sweep found that every grouping of eight or more neurons still touched 100% of pages; only one-neuron pages exposed the trivial 10%/5% omission, at 4,378/4,621 requests per token. A corrected multi-layer oracle found no configuration above 1% ideal global MLP savings that met the top-1 gate.

A different final-layer oracle based on the local logit Jacobian did survive an expanded 90-prompt enterprise-synthetic screen: selecting 30% of layer-23 neurons preserved dense top-1 on all 270 evaluated states with mean KL `0.00919`. The original neuron-set distillation path still failed, but changing both the unit and objective produced a causal result: a rank-256 selector trained directly on behavioral reconstruction selected 136 of 152 contiguous 32-neuron blocks before final-layer MLP execution. On a deduplicated and shuffled 160-train/45-test enterprise split it reached `1.0` top-1, `0.00644` mean KL, and `1.0` exact five-token free-running agreement at `89.4737%` execution. Selector metadata is `2.0532%` of final-layer MLP parameters, for a metadata-inclusive analytic fraction of `91.5269%`. A second training seed and a second shuffled split also passed. This remains a benchmark-specific logical result: the varied default corpus failed, PyTorch hooks still execute the dense MLP, and physical bytes and latency remain unmeasured.

The independent-distribution continuation replaced the learned address with a deterministic 2-bit causal gate/up proxy over 16-neuron blocks and added a calibrated probability-margin fallback. On three pairwise-disjoint 200-prompt Stanford Alpaca test sets it produced 1,799/1,800 top-1 matches (`0.99944`, combined 95% Wilson lower `0.99686`), mean KL `0.00820`, and `0.84389` average exact-MLP execution over 1,800 autoregressive states. Packed address metadata raises the modeled final-layer fraction to `0.92834`. The post-freeze third split independently passed at `0.99833` top-1, `0.00806` KL, and `0.84458` execution. Raw 75% sparse selection failed at 92.5%-94.0% top-1, showing that calibrated fallback creates the frontier. Physical bytes and latency remain null because the diagnostic uses dense hooks, reconstructs the packed address into FP32, and does not implement or time fallback completion.

## Infrastructure

- Windows 11 workstation with NVIDIA GeForce RTX 4070 SUPER, driver 610.88, 12,282 MiB GDDR6X.
- Python 3.12.10, PyTorch 2.11.0+cu128, CUDA runtime 12.8, Transformers 5.19.0.
- Qwen2.5-0.5B-Instruct model and tokenizer pinned to revision `7ae557604adf67be50417f59c2c2f167def9a775`.
- Host CUDA smoke and dense Qwen trace passed. Peak allocated GPU memory in the trace was about 1.03 GB.
- Docker Desktop 4.82 and WSL2 are operational after recoverably renaming four orphaned zero-byte AF_UNIX endpoints that made the backend crash during listener initialization. Docker Engine 29.6.1 reports `overlayfs`; `nvidia/cuda:12.4.1-base-ubuntu22.04` successfully ran `nvidia-smi` against the RTX 4070 SUPER. The stale endpoints remain preserved beside the fresh runtime sockets.

## PR #13 repairs

- Commit `56e543d` charges static warmup, demand transfers, transition prefetches, and oracle look-ahead prefetches.
- Every insertion now records transfers and evictions; cache capacity is asserted.
- Demands larger than capacity produce `infeasible_capacity` instead of misleading results.
- Sequence-reset scope, noncausal oracle labeling, and physical-metric nulls are explicit.
- Added replay, locality, layout, padding-position, cache-claim, oracle-schema, sparse-order, and selector-reproducibility invariant tests; Ruff passes and the full local suite passes (`82 passed`).
- The previous remote checkpoint had all 28 checks pass in GitHub Actions run `38205060063` on head `070b622`, including Python 3.10/3.12 and the corrected research jobs; the resulting artifacts were audited. The continuation head is reported in the PR handoff after its full revalidation run completes.
- Commit `e41abb9` adds pinned-revision CUDA trace metadata, atomic outputs, exact environment metadata, held-out-split validation, and correct cold-start/steady latency summaries.
- Commit `6cb9837` adds a deterministic prompt-disjoint locality control.
- Commit `9585cc1` removes false “physical fault” names from the oracle schema and records revision, seed, device, precision, and explicit physical-metric nulls.
- Commit `8faee7d` learns neuron layouts on 210 calibration prompts and evaluates identity, random, and learned layouts on 90 prompt-disjoint test prompts.
- Commit `499b577` adds multi-granularity held-out layout sweeps and a no-LRU mode for isolating demand amplification.
- Commit `080cc79` corrects the global oracle to use real next-token positions, adds full-width controls, and supports dense-trajectory rollouts.
- Commit `7cd3cf1` fixes the same right-padding/last-position hazard in 17 additional batched research harnesses and adds a static regression guard.
- Commit `8774db4` fixes a CUDA device-mutation bug in the tangent oracle, pins its revision/seed metadata, and relabels analytic row misses as logical rather than physical across temporal cache harnesses.
- Commit `414269a` pins the remaining selector runs, separates teacher and execution widths, canonicalizes sparse accumulation order across 20 evaluators, and expands regression coverage.
- Commit `b111378` adds a backward-compatible hybrid objective that distills continuous marginal logit-tangent utility alongside binary working-set membership.
- Commit `f2edbb5` adds the direct behavioral block-selector harness, strict budget/gate accounting, free-running evaluation, static/random controls, regression tests, and a pinned CPU CI smoke.
- Commit `ae3af05` adds the 2-bit causal block address, calibrated fallback accounting, independent Stanford Alpaca evaluation, confidence intervals, committed result payloads, and CI coverage.

## Experiments executed

| Experiment | Configuration | Result |
|---|---|---|
| Replay invariants | No-model fixtures, capacities 1-2 | Passed all accounting and capacity tests |
| Qwen trace smoke | Layer 23, FP16, 10% neurons, 128 neurons/page | All 38/38 pages demanded on every row |
| Corrected replay | Capacities 37 and 38 | 37 infeasible; all policies tied at 38 |
| Logical fraction sweep | Layer 23, 0.5%-5% active neurons | Mean contiguous-page demand ranged from 17.7/38 pages at 0.5% to 37.9/38 at 5% |
| Exact sparse-oracle screen | Layers 12 and 23, 30 prompts, 3 decode steps | Screen suggested 70% at layer 12 and 95% at layer 23 |
| Full-width identity control | Layers 12 and 23, ordered neuron indices | 100% top-1 agreement; removed FP16 permutation artifact |
| Expanded oracle smoke | 300 prompts, 3 decode steps | Layer 12: 70% and 85% failed; 90% passed. Layer 23: 95% passed |
| Controlled temporal locality | Layer 23, 0.5%-5% active, held-out traces | Adjacent Jaccard was below the same-task/same-step between-sequence control at every fraction |
| Calibration-only layout, 3 seeds | Layers 12/23, 128 neurons/page, 210 calibration + 90 test prompts | Learned layout reduced low-fraction page demand, but all layouts touched 38/38 pages at 90% and 95% |
| Page-granularity frontier, 3 seeds | Quality points, 1/8/16/32/64/128 neurons/page | 1-neuron pages retained the 90%/95% fractions; every page size >=8 saturated to 100% |
| Corrected global sparse oracle | 300 prompts, all/half/alternating layer groups, full-width control | Every configuration above 1% ideal global MLP savings failed top-1 >=99%; 100% width passed identity |
| Batched-harness audit | 18 scripts that read the last position of padded batches | Found and fixed right-padding contamination in 17 additional scripts |
| Expanded logit-tangent oracle | Final layer, 90 independent enterprise prompts, 3 steps, 30%-40% selected | All fractions reached 100% top-1; 30% had mean KL 0.00919 and 14.33% steady logical row misses with a 60% cache |
| Causal tangent distillation | Final layer, 20 train + 20 test prompts, 10 steps | Failed: 30%/40% payload reached only 78%/83% top-1 |
| Calibrated selective fallback | Layer 18, 16/16/16 train/calibration/test prompts | 99.49% top-1, but only 8.1% sparse acceptance and 2.24% ideal payload reduction after metadata |
| Causal selector capacity | Final layer, 30% teacher/execution, rank 128, 2,000 steps | Failed: 81.5% top-1, mean KL 0.13685, 57.6% oracle-set recall; metadata grew to 6.55% of the layer |
| Causal safety envelope | Rank 16, 30% teacher, 40%-100% execution | 40%-99% execution reached only 85.5%-98.5% top-1; only the sorted 100% identity control passed |
| Direct 90% causal target | Rank 16, 90% teacher and execution | Failed: 83.5% top-1, mean KL 0.08045 |
| Continuous tangent-utility distillation | Rank 16, 30% execution, hybrid ordinal weights 0.1 and 1.0 | Failed: best run reached 78.5% top-1, mean KL 0.12573, and 57.4% oracle-set recall |
| Expanded calibrated fallback | Layer 18, 100/100/100 prompt-disjoint split | Failed to reproduce: 15% calibrated no threshold; 20% accepted 0% of test states and cost 104.67% of dense payload including metadata |
| Sparse accumulation audit | 20 selective MLP evaluators | Canonicalized selected neuron indices before FP16 reductions and added a static regression guard |
| Block-structured causal selector | Final layer, 32-neuron blocks, rank 256, 160/45 deduplicated shuffled enterprise prompts | Passed: 100% top-1, mean KL 0.00644, and 100% exact five-token free-running agreement at 89.47% execution plus 2.05% metadata; static and random controls failed |
| Block-selector robustness | Second training seed and second shuffled 160/45 split | Passed: 100%/0.00641 and 99.56%/0.00714 top-1/KL; varied default 20/20 corpus failed at 92%/0.01782 |
| Independent low-bit block fallback | Three pairwise-disjoint 200-prompt Stanford Alpaca test sets, 600 decode states each | Passed: aggregate 99.944% top-1, KL 0.00820, 84.39% exact execution, and 92.83% metadata-inclusive layer fraction; post-freeze split passed independently |

## Best configuration

No physically validated winner is reported. The strongest causal quality result is a deterministic 2-bit block address plus calibrated fallback over the final-layer MLP. It passed the strict token gate on three pairwise-disjoint test sets drawn from the commit-pinned Stanford Alpaca corpus, including an untouched post-freeze confirmation split. No selective kernel has executed the advertised sparse path physically.

- Sparse working-set fraction: `75%` of final-layer neurons in contiguous 16-neuron blocks
- Sparse acceptance: `62.444%` across 1,800 held-out states; rejected states are charged as 100% execution
- Average exact execution: `84.389%`
- Packed address metadata: 2,208,256 bytes, `8.44494%` of the FP16 final-layer MLP
- Metadata plus execution: `92.834%`, leaving a `7.166%` layer-relative analytic margin
- KL divergence: aggregate mean `0.00820`; per-split p95 `0.03424`-`0.03825` and maximum `0.16728`-`0.25264`
- Top-1 agreement: 1,799/1,800 = `0.99944`; combined 95% Wilson lower bound `0.99686`
- Free-running correctness: 599/600 = `0.99833` exact three-token prompts; 1,799/1,800 matching recursive tokens
- Control: raw 75% sparse selection reached only `0.925`-`0.940` top-1 and `0.01775`-`0.01912` KL
- Whole-model scale: the ideal metadata-inclusive saving is roughly `0.19%` of FP16 model parameter bytes before candidate-projection, fallback, dispatch, and kernel overhead
- Physical transfer measurements: `null`
- End-to-end latency: `null`

## Novel findings

1. Contiguous neuron layout destroys apparent sparsity quickly: selecting only 0.5% of neurons still touched an average of 17.7 of 38 pages at layer 23.
2. Layer sensitivity differs, but less optimistically than the screen implied. The smallest observed expanded-smoke passes were 90% at layer 12 and 95% at layer 23.
3. An unsorted full-width top-k control can create false FP16 quality loss solely from accumulation order. Sorting selected neuron indices restored the identity control.
4. At 0.5% active neurons, adjacent-step Jaccard was `0.508`, below the same-task/same-step between-sequence control at `0.772`; the adjacent-minus-control lift remained negative through 5%.
5. Calibration-only layout learning generalized at very sparse widths: over three seeds it reduced mean requested pages by `13.9%` versus identity at layer 12 and `13.7%` at layer 23 for 0.5% active neurons. The benefit collapsed to zero at behaviorally viable fractions.
6. At the quality-passing fractions, granularity dominates layout: pages of 8-128 neurons all saturated, while per-neuron paging preserves only a 10%/5% payload reduction and creates thousands of requests per token.
7. Right-padded batch evaluation can manufacture impressive false positives. The invalid padded-position global screen showed 10% ideal savings with 100% agreement; the corrected real-token evaluation at the same point produced only `0.9633` top-1 agreement.
8. Probability-space importance is much sparser than raw activation importance in the final layer: a 30% local logit-tangent oracle preserved all top-1 decisions on the expanded screen, but a small causal predictor reproduced too little of its working set to preserve behavior.
9. Causal-selector failure is not explained by low-rank capacity: increasing the address rank from 16 to 128 and training four times longer improved 30%-width top-1 only to `0.815`, while resident selector metadata grew from `0.852%` to `6.554%` of final-layer MLP parameters.
10. The causal safety envelope is effectively dense. With a 30% tangent teacher, execution widths from 40% through 99% rose monotonically from `0.855` to `0.985` top-1; only the sorted 100% identity control passed the 99% gate. At 99%, rank-16 metadata alone makes the analytic payload exceed a dense layer.
11. The small calibrated-fallback frontier did not reproduce at 100/100/100 prompts. The 15% path could not calibrate a threshold, and the 20% path accepted no held-out states while its resident metadata raised modeled payload to `1.0467x` dense.
12. The Docker failure was environmental rather than a CUDA limitation: four stale AF_UNIX reparse points blocked listener cleanup. Recoverable in-place renames restored the daemon and a CUDA-container GPU smoke passed.
13. Changing the execution unit and loss mattered. A direct behavioral selector over contiguous 32-neuron blocks passed the synthetic enterprise gate at 89.47% execution, while static-energy and random-fixed blocks failed.
14. The pass is workload-specific rather than universal: it replicated across selector seeds and shuffled enterprise splits, but failed on the more varied default prompt corpus.
15. A deterministic 2-bit causal address plus calibrated fallback crosses the independent-distribution gate. Three non-overlapping Stanford Alpaca test sets reached aggregate top-1 `0.99944` and KL `0.00820` at `0.84389` average execution; the combined Wilson lower bound remains above 99%.

## Falsified hypotheses

- The original 10%-neuron, 128-neuron-page configuration does **not** expose a compact contiguous logical page working set at layer 23.
- Logical activation sparsity alone is insufficient evidence for page-level or physical bandwidth savings.
- The 30-prompt layer-12 70% result did not replicate on the expanded synthetic set; 70% and 85% both missed the 99% top-1 gate.
- Adjacent decode states did not share more contiguous-page demand than the prompt-disjoint task-matched control at any tested sparse fraction.
- The one-step-lagged hidden-delta signal did not positively predict current new-page fraction consistently (`r=-0.085`, `-0.528`, `0.035`, and undefined as page sets saturated).
- A calibration-learned neuron layout did not rescue the 128-neuron page abstraction at quality-passing fractions: layer 12 at 90% and layer 23 at 95% requested all 38 pages for every tested layout and seed.
- Reducing page size to 8, 16, 32, or 64 neurons did not expose any quality-compatible page sparsity; each configuration requested 100% of pages across three seeds.
- Spreading exact-current-activation sparsity across layer groups did not produce a >1% ideal global MLP reduction at the quality gate. At 1.25% savings, top-1 agreement ranged only from `0.9233` to `0.9267` despite small mean KL.
- The initial multi-layer positive result was falsified as a padding-position artifact and is excluded from claims.
- The first causal tangent-address attempt did not recover the oracle frontier: 30% and 40% payloads reached only `0.78` and `0.83` top-1 agreement.
- The causal tangent-address failure was not rescued by more rank or training: rank 128 at 30% reached only `0.815` top-1 and incurred `6.55%` layer-relative metadata.
- Expanding the binary selector's execution set did not expose a sparse quality frontier: even 99% execution reached only `0.985` top-1; 100% was required for identity.
- Training directly against a 90% teacher did not fix ordering outside the original 30% target; 90% execution reached only `0.835` top-1.
- Adding continuous first-step tangent-utility ranks to the binary objective did not materially improve useful-width quality; the best 30% run reached `0.785` top-1.
- The 16/16/16 selective-fallback result was a small-sample false frontier. On a prompt-disjoint 100/100/100 split, the calibrated policy accepted 0% of test states and had no net payload advantage.

## Engineering status

- Branch: `research/state-transition-residency-gate-v1`
- PR: <https://github.com/Droptops/DepthRouter/pull/13>
- Repair commit: `56e543d`
- Reproducible trace commit: `e41abb9`
- Controlled locality commit: `6cb9837`
- Logical cache labeling commit: `9585cc1`
- Held-out layout commit: `8faee7d`
- Page-granularity commit: `499b577`
- Corrected global-oracle commit: `080cc79`
- Batched-padding audit commit: `7cd3cf1`
- Logical temporal-cache labeling and CUDA fix: `8774db4`
- Sparse-order and reproducibility hardening: `414269a`
- Continuous tangent-utility distillation: `b111378`
- Block-structured behavioral selector: `f2edbb5`
- Independent low-bit block fallback: `ae3af05`
- Merge status: open and not merged

## Blockers

- No selective kernel or supported hardware counter has measured physical GDDR6X traffic.
- The low-bit diagnostic reconstructs its packed address into FP32 tensors and therefore does not measure the claimed packed execution path.
- Confidence requires a full-vocabulary sparse candidate projection before fallback; rejected states need completion and a second output projection. These costs are not represented by the final-layer MLP fraction.
- The remaining ideal saving is only about `0.19%` of whole-model FP16 parameter bytes, so measurable end-to-end benefit is unproven and may be negative.

## Next three experiments

1. Build a fused packed 2-bit address plus block-sparse final-layer kernel that can complete omitted blocks without recomputing accepted work.
2. Measure candidate-projection, completion, dispatch, physical bytes, and end-to-end latency against dense execution; reject the path if the roughly `0.19%` whole-model ideal margin disappears.
3. Only after physical validation, test whether the same calibrated fallback can be applied to multiple late layers without compounding errors or metadata beyond the dense baseline.
