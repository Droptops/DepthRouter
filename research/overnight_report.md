# DepthRouter overnight research report

## Executive finding

**Inconclusive, with evidence against the current breakthrough hypothesis.** The repaired replay shows that the original layer-23 configuration (top 10% of activation-weight contributions grouped into contiguous 128-neuron pages) touches all 38 pages on every observed token. No constrained working set exists under that definition, and static, previous-token, transition, and noncausal-oracle policies tie once all initialization and prefetch transfers are charged.

The controlled locality test also went the wrong way: sparse page sets at adjacent decode steps were less similar than page sets from different prompts of the same task at the same step. Expanded 300-prompt oracle smoke tests only passed the behavioral gate at 90% of layer 12 and 95% of layer 23. These are exact-current-activation oracle results on synthetic data, not causal routing or physical bandwidth improvements.

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
- Added replay, locality, and oracle-schema invariant tests; Ruff passes and the full local suite passes (`67 passed`).
- GitHub Python 3.10 and 3.12 CI jobs passed after the repair. Long research jobs were still completing at this checkpoint.
- Commit `e41abb9` adds pinned-revision CUDA trace metadata, atomic outputs, exact environment metadata, held-out-split validation, and correct cold-start/steady latency summaries.
- Commit `6cb9837` adds a deterministic prompt-disjoint locality control.
- Commit `9585cc1` removes false “physical fault” names from the oracle schema and records revision, seed, device, precision, and explicit physical-metric nulls.

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

## Best configuration

No winner is reported. The smallest observed expanded-smoke point that passed the token-distribution gate was layer 12 at 90% selected neurons: mean KL `0.0000463`, top-1 agreement `0.99444`, and exact agreement over all three-token rollouts for `0.9833` of prompts. Layer 12 at 85% failed with top-1 agreement `0.98667`; the apparent 70% screen pass fell to `0.95667` on 300 prompts. This tuning sequence is exploratory and not an independent held-out confirmation.

- Working-set fraction: smallest observed behavioral smoke pass is 90% of layer 12's MLP neurons
- Number of physical pages: not established for a compiled layout
- KL divergence: 0.0000463 on the expanded synthetic smoke set
- Top-1 agreement: 0.99444 on dense greedy trajectories
- Free-running correctness: exact three-token agreement 0.9833; task correctness not measured
- Logical transfer costs: not yet matched to this quality point
- Physical transfer measurements: `null`
- End-to-end latency: `null`

## Novel findings

1. Contiguous neuron layout destroys apparent sparsity quickly: selecting only 0.5% of neurons still touched an average of 17.7 of 38 pages at layer 23.
2. Layer sensitivity differs, but less optimistically than the screen implied. The smallest observed expanded-smoke passes were 90% at layer 12 and 95% at layer 23.
3. An unsorted full-width top-k control can create false FP16 quality loss solely from accumulation order. Sorting selected neuron indices restored the identity control.
4. At 0.5% active neurons, adjacent-step Jaccard was `0.508`, below the same-task/same-step between-sequence control at `0.772`; the adjacent-minus-control lift remained negative through 5%.

## Falsified hypotheses

- The original 10%-neuron, 128-neuron-page configuration does **not** expose a compact contiguous logical page working set at layer 23.
- Logical activation sparsity alone is insufficient evidence for page-level or physical bandwidth savings.
- The 30-prompt layer-12 70% result did not replicate on the expanded synthetic set; 70% and 85% both missed the 99% top-1 gate.
- Adjacent decode states did not share more contiguous-page demand than the prompt-disjoint task-matched control at any tested sparse fraction.
- The one-step-lagged hidden-delta signal did not positively predict current new-page fraction consistently (`r=-0.085`, `-0.528`, `0.035`, and undefined as page sets saturated).

## Engineering status

- Branch: `research/state-transition-residency-gate-v1`
- PR: <https://github.com/Droptops/DepthRouter/pull/13>
- Repair commit: `56e543d`
- Reproducible trace commit: `e41abb9`
- Controlled locality commit: `6cb9837`
- Logical cache labeling commit: `9585cc1`
- Merge status: open and not merged

## Blockers

- Docker Desktop daemon startup has not completed; container GPU passthrough remains unverified.
- The 300-example dataset is synthetic smoke data, not realistic enterprise generalization evidence.
- The best quality points use current dense activations and are noncausal; the 300-prompt tuning run is exploratory rather than a clean final test.
- No selective kernel or supported hardware counter has measured physical GDDR6X traffic.
- Task-level free-running correctness and three-seed uncertainty are not yet available.

## Next three experiments

1. Freeze candidate fractions before evaluating a provenance-tracked, prompt-disjoint corpus with longer rollouts, task parsing, FP16/FP32 controls, three seeds, and uncertainty.
2. Test whether a calibration-only neuron permutation reduces held-out contiguous page demand; reject it if it cannot beat the unpermuted and random-permutation controls without changing dense behavior.
3. Do not invest in a learned transition selector until a representation shows positive adjacent-over-control locality. If one does, compare it with previous-token reuse including detector, metadata, warmup, eviction, and fallback costs.
