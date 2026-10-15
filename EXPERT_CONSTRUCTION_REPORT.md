# DepthRouter expert construction: bounded capability audit

4 of 12 individual partitions passed the user-specified coverage and portfolio-advantage gates on development and both held-out splits: B-17, B-29, C-17, C-29. 4 also passed the corresponding lower-confidence-bound checks. Development selected family C (hidden-output reconstruction), which failed the predeclared additional family-level continuation criterion: every one of the 3 seeds must pass. Selected-family failure: insufficient acceptable cheap candidates. The result applies to these partitions, candidate sets, data, and fitting budget; it does not rule out useful cheap computation generally.

C-43 heldout_ood coverage was 78.620 [76.735, 80.587]%; the interval includes the 80% threshold.

## Experiment and bounded conclusion

Evaluated 48 raw candidate matrices / 3,083,808 state-candidate pairs. Model `Qwen/Qwen2.5-0.5B-Instruct` at `7ae557604adf67be50417f59c2c2f167def9a775`; base `ec696fddabc59c51292ba1e8f500c9000a50c79b`; seeds `[17, 29, 43]`. Only final-layer MLP computation changes: eight balanced 608-neuron experts, every six-of-eight subset, 75% width retained. Original weights are frozen; no residual correction is added.

A = balanced random; B = balanced co-activation; C = hidden reconstruction; D = final full-vocabulary KL. C/D use paired initialization, data order, candidate sampling, constraints and 100 fitting updates per objective and seed. The exhaustive 28-candidate oracle answers existence only and is never a deployable selector.

Individual passes establish useful candidate portfolios within this audit. Family continuation additionally requires consistency across every preregistered seed. That extra rule does not erase successful individual partitions or establish a general impossibility result. No held-out seed, family, threshold, or extra training budget is selected after seeing these outcomes.

## Every partition: coverage and advantage

Each split cell is acceptable-state coverage (%) / portfolio advantage over the development-frozen subset (percentage points). Coverage >=80% and advantage >=5 points are required on development and both held-out splits. The final column aggregates those user-specified engineering gates for each individual partition.

| Arm-seed | Development coverage / advantage | Matched coverage / advantage | OOD coverage / advantage | Individual all-split gate |
| --- | --- | --- | --- | --- |
| A-17 | 65.03 / 5.75 | 63.08 / 5.50 | 58.71 / 5.76 | FAIL |
| A-29 | 65.32 / 5.28 | 63.23 / 5.31 | 61.40 / 4.97 | FAIL |
| A-43 | 61.59 / 5.99 | 60.63 / 6.46 | 56.36 / 5.61 | FAIL |
| B-17 | 87.72 / 5.94 | 86.25 / 5.99 | 84.20 / 6.08 | PASS; resolved |
| B-29 | 85.27 / 6.04 | 83.46 / 5.66 | 83.41 / 6.23 | PASS; resolved |
| B-43 | 77.11 / 6.31 | 75.75 / 6.39 | 72.70 / 5.50 | FAIL |
| C-17 | 87.52 / 7.15 | 84.49 / 7.12 | 84.20 / 6.54 | PASS; resolved |
| C-29 | 86.84 / 7.37 | 84.00 / 6.85 | 84.15 / 7.22 | PASS; resolved |
| C-43 | 83.10 / 5.80 | 80.41 / 6.31 | 78.62 / 6.43 | FAIL |
| D-17 | 67.09 / 5.72 | 62.84 / 5.17 | 61.89 / 5.33 | FAIL |
| D-29 | 65.82 / 6.51 | 63.18 / 6.00 | 59.34 / 5.55 | FAIL |
| D-43 | 65.62 / 5.94 | 65.93 / 6.37 | 62.62 / 5.76 | FAIL |

## Development-selected family, with uncertainty

| Arm-seed | Split | Contexts/states | Coverage % [95% CI] | Fixed coverage % | Ideal active width % | Advantage pp [95% CI] | Engineering gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C-17 | development | 256/1018 | 87.525 [85.419, 89.618] | 58.94 | 78.12 | 7.146 [6.501, 7.833] | PASS; resolved |
| C-17 | heldout_matched | 512/2037 | 84.487 [82.850, 86.058] | 56.01 | 78.88 | 7.118 [6.639, 7.629] | PASS; resolved |
| C-17 | heldout_ood | 512/2044 | 84.198 [82.609, 85.805] | 58.02 | 78.95 | 6.544 [6.088, 6.994] | PASS; resolved |
| C-29 | development | 256/1018 | 86.837 [84.759, 88.933] | 57.37 | 78.29 | 7.367 [6.727, 8.038] | PASS; resolved |
| C-29 | heldout_matched | 512/2037 | 83.996 [82.466, 85.588] | 56.60 | 79.00 | 6.848 [6.379, 7.360] | PASS; resolved |
| C-29 | heldout_ood | 512/2044 | 84.149 [82.574, 85.805] | 55.28 | 78.96 | 7.216 [6.736, 7.710] | PASS; resolved |
| C-43 | development | 256/1018 | 83.104 [80.728, 85.378] | 59.92 | 79.22 | 5.796 [5.162, 6.429] | PASS; resolved |
| C-43 | heldout_matched | 512/2037 | 80.412 [78.498, 82.160] | 55.18 | 79.90 | 6.308 [5.861, 6.768] | PASS; unresolved CI |
| C-43 | heldout_ood | 512/2044 | 78.620 [76.735, 80.587] | 52.89 | 80.34 | 6.433 [5.959, 6.902] | FAIL: coverage |

Intervals resample whole contexts and retain token-state weighting; they are conditional on the evaluated seed set and are not multiplicity-adjusted. A point failure whose interval includes the threshold is not a statistically resolved failure. Ideal active width is 1 - 0.25 × cheap coverage; it excludes routing, movement, synchronization and all hardware overhead.

## Paired D minus C

| Split | Metric | Difference [95% CI] | Units |
| --- | --- | --- | --- |
| heldout_matched | cheap_coverage | -18.982 [-20.527, -17.509] | percentage points |
| heldout_matched | fixed_cheap_coverage | -15.333 [-16.691, -14.026] | percentage points |
| heldout_matched | oracle_agreement | +0.098 [+0.016, +0.196] | percentage points |
| heldout_matched | oracle_kl | +0.008898 [+0.007861, +0.010136] | raw KL |
| heldout_matched | portfolio_advantage | -0.912 [-1.349, -0.492] | percentage points |
| heldout_ood | cheap_coverage | -21.037 [-22.522, -19.618] | percentage points |
| heldout_ood | fixed_cheap_coverage | -16.308 [-17.693, -14.825] | percentage points |
| heldout_ood | oracle_agreement | -0.033 [-0.163, +0.065] | percentage points |
| heldout_ood | oracle_kl | +0.009776 [+0.008990, +0.010647] | raw KL |
| heldout_ood | portfolio_advantage | -1.182 [-1.619, -0.771] | percentage points |

Acceptable coverage requires both the same greedy token and KL<0.01. Greedy agreement is reported separately: a small agreement increase does not imply better local acceptability. Positive coverage/agreement/advantage changes favor D, whereas negative KL changes favor D. These conclusions concern this bounded construction recipe, not every possible output-distribution objective.

## Numerical controls and data completeness

| All-eight layout | Arm/split controls | Minimum agreement % | Maximum mean KL | Point passes |
| --- | --- | --- | --- | --- |
| control | 48 | 100.000000 | 0 | 48/48 |
| permuted_control | 48 | 99.559687 | 0.000170857985 | 48/48 |

The original-index all-eight layout is the gating control and matches the globally sorted sparse layout. The expert-contiguous permutation is an additional reduction-order diagnostic. Full-model substitution checks and every raw control are retained in the detailed report; real-arithmetic identities are not used as evidence of BF16 exactness.

| Split | Contexts requested | Contexts with states | States | Missing planned states (EOS) | Truncated prompts |
| --- | --- | --- | --- | --- | --- |
| calibration | 1024 | 1024 | 4079 | 17 | 0 |
| development | 256 | 256 | 1018 | 6 | 0 |
| heldout_matched | 512 | 512 | 2037 | 11 | 0 |
| heldout_ood | 512 | 512 | 2044 | 4 | 0 |

Dense-prefix continuation positions: `[1, 3, 7, 11]`. No post-EOS states enter denominators. Fresh human-written Dolly source contexts have disjoint documents and exact lexical instruction families; 12.5% are enterprise-keyword topics. Creative writing and summarization are reserved OOD. Lexical exclusions do not guarantee semantic-family independence; source answers are unused.

## Selection and execution

Phase 2: `skipped_phase1_gate_failed`. Phase 3: `not_run_requires_phase2_quality_pass`. Learned selector quality, free-generation task success, selector-plus-MLP latency, end-to-end decoding cost and physical DRAM traffic remain unmeasured when prerequisites fail. Unknown costs are retained as null; arithmetic proxies are not measured savings.

Hardware: `NVIDIA GeForce RTX 4070 SUPER`; dtype `torch.bfloat16`; PyTorch `2.7.1+cu128`, CUDA `12.8`, Transformers `4.57.1`, attention `sdpa`, batch 16. Audit wall time: `449.735045681` seconds, bounded by 14400 seconds; this is not inference latency. Even ideal 25% removal from one of 24 MLPs is only about 1.04% of total MLP arithmetic before overhead.

## Reproduction and retained evidence

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_expert_construction.ps1 -RunId fresh-expert-audit
```

```text
python experiments/expert_construction_report.py --run-dir artifacts/expert_construction/20261014-bounded-v1
```

The [complete report](artifacts/expert_construction/20261014-bounded-v1/REPORT.md) contains every arm/seed/split, raw fidelity, full controls, data provenance, task/margin strata, source hashes, and all raw matrix hashes. The [machine-readable decision](artifacts/expert_construction/20261014-bounded-v1/promotion_decision.json) records the individual gates and additional family-level continuation decision.

Earlier negative results remain intact. Historical contribution-ranking oracles and synthetic notebook checks are distinct from this exhaustive candidate audit. DOT-MoE, DynaMoE, and goal-oriented model reduction are conceptual sources, not claimed reproductions or novelty evidence; source links are in the detailed report.
