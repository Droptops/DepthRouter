# DepthRouter overnight research report

## Executive finding

**Inconclusive, with evidence against the current breakthrough hypothesis.** The repaired replay shows that the original layer-23 configuration (top 10% of activation-weight contributions grouped into contiguous 128-neuron pages) touches all 38 pages on every observed token. No constrained working set exists under that definition, and static, previous-token, transition, and noncausal-oracle policies tie once all initialization and prefetch transfers are charged.

The controlled locality test also went the wrong way: sparse page sets at adjacent decode steps were less similar than page sets from different prompts of the same task at the same step. A calibration-only learned layout reduced held-out page demand at unusably sparse fractions, but produced no reduction at the behaviorally viable settings. A finer granularity sweep found that every grouping of eight or more neurons still touched 100% of pages; only one-neuron pages exposed the trivial 10%/5% omission, at 4,378/4,621 requests per token. A corrected multi-layer oracle found no configuration above 1% ideal global MLP savings that met the top-1 gate. These are exact-current-activation oracle results on synthetic data, not causal routing or physical bandwidth improvements.

## Infrastructure

- Windows 11 workstation with NVIDIA GeForce RTX 4070 SUPER, driver 610.88, 12,282 MiB GDDR6X.
- Python 3.12.10, PyTorch 2.11.0+cu128, CUDA runtime 12.8, Transformers 5.19.0.
- Qwen2.5-0.5B-Instruct model and tokenizer pinned to revision `7ae557604adf67be50417f59c2c2f167def9a775`.
- Host CUDA smoke and dense Qwen trace passed. Peak allocated GPU memory in the trace was about 1.03 GB.
- Docker Desktop and WSL2 are installed, but the Docker daemon did not become ready during the initial startup attempt. Host CUDA work continued independently.

## PR #13 repairs

- Commit `56e543d` charges static warmup, demand transfers, transition prefetches, and oracle look-ahead prefetches.
- Every insertion now records transfers and evictions; cache capacity is asserted.
- Demands larger than capacity produce `infeasible_capacity` instead of misleading results.
- Sequence-reset scope, noncausal oracle labeling, and physical-metric nulls are explicit.
- Added replay, locality, layout, padding-position, and oracle-schema invariant tests; Ruff passes and the full local suite passes (`75 passed`).
- GitHub Python 3.10 and 3.12 CI jobs passed after the repair. Long research jobs were still completing at this checkpoint.
- Commit `e41abb9` adds pinned-revision CUDA trace metadata, atomic outputs, exact environment metadata, held-out-split validation, and correct cold-start/steady latency summaries.
- Commit `6cb9837` adds a deterministic prompt-disjoint locality control.
- Commit `9585cc1` removes false “physical fault” names from the oracle schema and records revision, seed, device, precision, and explicit physical-metric nulls.
- Commit `8faee7d` learns neuron layouts on 210 calibration prompts and evaluates identity, random, and learned layouts on 90 prompt-disjoint test prompts.
- Commit `499b577` adds multi-granularity held-out layout sweeps and a no-LRU mode for isolating demand amplification.
- Commit `080cc79` corrects the global oracle to use real next-token positions, adds full-width controls, and supports dense-trajectory rollouts.
- Commit `7cd3cf1` fixes the same right-padding/last-position hazard in 17 additional batched research harnesses and adds a static regression guard.

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

## Best configuration

No winner is reported. The smallest observed expanded-smoke single-layer point that passed the token-distribution gate was layer 12 at 90% selected neurons: mean KL `0.0000463`, top-1 agreement `0.99444`, and exact agreement over all three-token rollouts for `0.9833` of prompts. In the corrected global screen, omitting 1% of neurons in the late 12 layers passed one-step quality but saves only 0.5% of ideal global MLP payload; every tested configuration above 1% global MLP savings failed the top-1 gate. These tuning runs are exploratory and not independent held-out confirmations.

- Working-set fraction: smallest observed behavioral smoke pass is 90% of layer 12's MLP neurons
- Number of physical pages: not established for a compiled layout
- KL divergence: 0.0000463 on the expanded synthetic smoke set
- Top-1 agreement: 0.99444 on dense greedy trajectories
- Free-running correctness: exact three-token agreement 0.9833; task correctness not measured
- Logical transfer costs: every tested page size >=8 neurons saturated; 1-neuron pages required 4,378 requests/token at layer 12
- Ideal modeled upper bound: omitting 10% of one FP16 MLP layer saves at most 2,614,886 weight bytes/token, only `0.2646%` of the model's FP16 parameter bytes before selector and dispatch overhead
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
- Merge status: open and not merged

## Blockers

- Docker Desktop daemon startup has not completed; container GPU passthrough remains unverified.
- The 300-example dataset is synthetic smoke data, not realistic enterprise generalization evidence.
- The best quality points use current dense activations and are noncausal; the 300-prompt tuning run is exploratory rather than a clean final test.
- No selective kernel or supported hardware counter has measured physical GDDR6X traffic.
- Task-level free-running correctness and three-seed uncertainty are not yet available.
- Historical research-job results produced before `7cd3cf1` require rerun or explicit verification if they used batched last-position evaluation.

## Next three experiments

1. Freeze candidate fractions before evaluating a provenance-tracked, prompt-disjoint corpus with longer rollouts, task parsing, FP16/FP32 controls, three seeds, and uncertainty.
2. Rerun affected CI research jobs under the left-padding invariant and invalidate any historical result that does not reproduce at real next-token positions.
3. Do not build the current paging kernel: neither the state-transition, layout, granularity, nor multi-layer oracle gates leave a plausible benefit. A future branch should require a coarser structured unit or trained causal sparsity mechanism before reopening physical profiling.
