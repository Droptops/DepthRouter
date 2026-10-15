# DepthRouter output-gap correction: controlled experiment

Output-gap training did not establish better held-out agreement than hidden-state MSE under the preregistered matched-budget comparison.
At the frozen primary rank 16, the mean paired gap-minus-hidden differences were -2.409 [-4.557, -0.260] percentage points on matched prompts and -4.622 [-6.055, -3.320] on OOD prompts (95% intervals).
0 of 12 correction fits passed the development absolute agreement-and-KL gate.
Run `20261014-controlled-v1` ended with `NEGATIVE_STOP`. Timing and free-running generation were skipped by the preregistered gate; there is no measured speedup or physical-traffic result from this run.

The notebook's synthetic examples motivated changing the training objective. They are not Qwen observations and contribute no evidence to this result.

## Controlled comparison

The model is `Qwen/Qwen2.5-0.5B-Instruct` at revision `7ae557604adf67be50417f59c2c2f167def9a775`. Only final MLP layer 23 changed. The fixed calibration-selected mask retains 3,648/4,864 neurons (75% retained, 25% omitted), physically packed into gate/up/down weights. Every sparse arm uses those same weights. The original full model and a full-width packed control are both included.

Both correction objectives use `Linear(896, rank, bias=False) -> tanh -> Linear(rank, 896, bias=True)` with FP32 parameters and the same final-MLP input. The correction is added to the actual sparse output in FP32 and cast back to BF16; the model's ordinary residual addition, final RMSNorm and full vocabulary head then run. The deployed correction has no teacher, candidate IDs or stored labels.

The target is the actual dense BF16 MLP output minus the actual packed sparse BF16 MLP output, measured by separate whole-model forwards. Hidden training minimizes hidden residual MSE. Gap training minimizes the seven contrasts between the cheap top-eight token rows and the cheap argmax row, using `A = E * gamma`. Each objective is divided by its calibration target mean square. Candidate search and head-row gathering are training-only operations for this architecture.

Ranks: [8, 16]; seeds: [17, 29, 43]; 300 AdamW steps per fit, minibatch 64, learning rate 0.003, weight decay 0.0001. Rank 8 has 15,232 parameters and rank 16 has 29,568. Each seed/rank pair has identical initialization and minibatch order: verified for all pairs. Only the loss objective changes within each pair.

## Agreement and KL for every arm

Each cell shows full-vocabulary greedy agreement / mean KL(full || candidate), using actual normalized model logits. Development has 256 examples; each held-out split has 512. These are first-token comparisons after each prompt.

| Arm | Development | Matched | OOD |
| --- | --- | --- | --- |
| Full Qwen | 100.00% / 0.000000 | 100.00% / 0.000000 | 100.00% / 0.000000 |
| Full-width packed control | 100.00% / 0.000000 | 100.00% / 0.000000 | 100.00% / 0.000000 |
| 75% retained, uncorrected | 94.53% / 0.017130 | 96.48% / 0.133505 | 95.90% / 0.083363 |
| Hidden MSE, r8, s17 | 98.83% / 0.014432 | 93.16% / 0.131384 | 95.51% / 0.110227 |
| Output gap, r8, s17 | 75.00% / 0.128673 | 96.48% / 0.230310 | 95.51% / 0.131631 |
| Hidden MSE, r8, s29 | 63.28% / 0.074100 | 95.70% / 0.171494 | 95.90% / 0.089233 |
| Output gap, r8, s29 | 74.61% / 0.129496 | 92.58% / 0.220910 | 96.68% / 0.185214 |
| Hidden MSE, r8, s43 | 96.09% / 0.029842 | 94.34% / 0.173597 | 82.42% / 0.089872 |
| Output gap, r8, s43 | 78.12% / 0.104505 | 91.99% / 0.203616 | 96.09% / 0.115033 |
| Hidden MSE, r16, s17 | 98.44% / 0.017034 | 94.92% / 0.140270 | 96.29% / 0.092128 |
| Output gap, r16, s17 (selected) | 99.22% / 0.044516 | 92.58% / 0.195081 | 83.98% / 0.152028 |
| Hidden MSE, r16, s29 | 88.67% / 0.042068 | 92.58% / 0.169229 | 96.68% / 0.111047 |
| Output gap, r16, s29 | 79.30% / 0.102078 | 95.51% / 0.194102 | 96.09% / 0.122628 |
| Hidden MSE, r16, s43 | 96.09% / 0.019548 | 93.16% / 0.148497 | 96.09% / 0.101282 |
| Output gap, r16, s43 | 91.41% / 0.114049 | 85.35% / 0.212308 | 95.12% / 0.165959 |

Development selected `gap_r16_s17`, fixing primary rank 16 before either held-out model evaluation. Selection rule: lowest development top1 error, then lowest KL, then lower rank; seed numeric tie-break only. The primary objective test retains all three seeds at that fixed rank; a favorable held-out seed or alternate rank cannot replace it.

## Paired objective differences

Values are output-gap minus hidden-MSE agreement in percentage points, followed by 95% paired bootstrap intervals (10,000 draws). The mean row averages the three seed differences within each prompt before resampling prompts. These intervals are conditional on the fixed template suite; repeated numeric/content variants are not independent natural-language template samples. No generalization to a natural-language population is claimed.

| Rank | Seed | Development | Matched | OOD |
| --- | --- | --- | --- | --- |
| 8 | 17 | -23.828 [-29.297, -18.359] | +3.320 [+1.172, +5.664] | +0.000 [-0.977, +0.977] |
| 8 | 29 | +11.328 [+3.906, +18.359] | -3.125 [-5.273, -0.977] | +0.781 [+0.195, +1.562] |
| 8 | 43 | -17.969 [-23.047, -13.281] | -2.344 [-4.102, -0.586] | +13.672 [+10.547, +16.797] |
| 8 | paired seed mean | -10.156 [-15.234, -5.208] | -0.716 [-1.497, +0.000] | +4.818 [+3.711, +5.924] |
| 16 | 17 | +0.781 [-1.172, +2.734] | -2.344 [-4.883, +0.195] | -12.305 [-15.430, -9.180] |
| 16 | 29 | -9.375 [-15.234, -3.906] | +2.930 [+0.586, +5.469] | -0.586 [-1.758, +0.586] |
| 16 | 43 | -4.688 [-7.422, -2.344] | -7.812 [-11.133, -4.492] | -0.977 [-2.148, +0.000] |
| 16 | paired seed mean | -4.427 [-6.771, -2.083] | -2.409 [-4.557, -0.260] | -4.622 [-6.055, -3.320] |

The frozen primary test requires positive direction for every seed and a paired-mean interval strictly above zero in both held-out splits.

| Primary split | All seeds positive | Mean CI excludes zero above | Established |
| --- | --- | --- | --- |
| heldout_matched | False | False | False |
| heldout_ood | False | False | False |

The absolute gate is agreement >= 99.00% and mean KL < 0.01 on development and both held-out splits. Selected-arm gate: FAIL.

| Selected-arm split | Agreement gate | KL gate | Combined |
| --- | --- | --- | --- |
| development | True | False | False |
| heldout_matched | False | False | False |
| heldout_ood | False | False | False |

## Failure diagnostics and numerical checks

The first margin group contains all examples at or below the empirical first quartile; ties can make its size differ from exactly one quarter. Excluded winners are full-model winners absent from the uncorrected sparse top-eight. Held-out diagnostics are measured after freezing; teacher winners are never added to training candidates.

| Split | Arm | Low-margin agreement (n) | Excluded-winner errors / cases | Hidden MSE | Gap MSE |
| --- | --- | --- | --- | --- | --- |
| development | 75% retained, uncorrected | 92.62% (122) | 0/0 | 0.00848712 | 0.27579 |
| development | Hidden MSE, r16, s17 | 96.72% (122) | 0/0 | 0.0102538 | 0.27384 |
| development | Output gap, r16, s17 | 98.36% (122) | 0/0 | 0.336568 | 0.970519 |
| heldout_matched | 75% retained, uncorrected | 93.75% (144) | 0/0 | 0.0139205 | 3.16425 |
| heldout_matched | Hidden MSE, r16, s17 | 88.19% (144) | 0/0 | 0.0183353 | 3.33972 |
| heldout_matched | Output gap, r16, s17 | 79.86% (144) | 0/0 | 0.340873 | 4.87629 |
| heldout_ood | 75% retained, uncorrected | 89.70% (165) | 0/0 | 0.00931632 | 1.88274 |
| heldout_ood | Hidden MSE, r16, s17 | 90.91% (165) | 0/0 | 0.021647 | 2.05707 |
| heldout_ood | Output gap, r16, s17 | 53.33% (165) | 0/0 | 0.339428 | 2.68792 |

All 45 arm/split summaries in `development_quality.json` and `heldout_quality.json` retain hidden and gap MSE, all margin-quartile counts and agreement rates, excluded-winner failures, and top-logit tie counts. The compact diagnostic table above shows the uncorrected arm and the development-selected objective pair; the raw summaries retain every seed.

The full-width packed control had 0 greedy mismatches and maximum absolute logit error 0. Checkpoint roundtrip exact: True; checkpoint unchanged after held-out evaluation: True. Calibration contrast-identity maximum absolute error: 3.57628e-07. Full-precision algebra is not asserted to be an exact BF16-kernel identity.

## Bound and cost audit

The weight-only bound uses `R = beta * ||x||² + ||correction(x)||` and `epsilon = max_i ||E_i * gamma|| * R`. Including the correction norm is necessary to bound the correction error. If `R >= ||u_sparse + correction||`, even an upper bound on the score spread cannot pass the uniform `gap > 2 * epsilon` test. Here beta is 240.232.

| Correction | Calibration cases | Certain nonacceptance | Certified | Minimum R / approximate-state norm |
| --- | --- | --- | --- | --- |
| Hidden MSE, r8, s17 | 256 | 256 | 0 | 9614.8 |
| Output gap, r8, s17 | 256 | 256 | 0 | 9346.2 |
| Hidden MSE, r8, s29 | 256 | 256 | 0 | 9585.0 |
| Output gap, r8, s29 | 256 | 256 | 0 | 9201.8 |
| Hidden MSE, r8, s43 | 256 | 256 | 0 | 9612.1 |
| Output gap, r8, s43 | 256 | 256 | 0 | 9263.9 |
| Hidden MSE, r16, s17 | 256 | 256 | 0 | 9615.2 |
| Output gap, r16, s17 | 256 | 256 | 0 | 9352.1 |
| Hidden MSE, r16, s29 | 256 | 256 | 0 | 9603.8 |
| Output gap, r16, s29 | 256 | 256 | 0 | 9224.9 |
| Hidden MSE, r16, s43 | 256 | 256 | 0 | 9606.8 |
| Output gap, r16, s43 | 256 | 256 | 0 | 9080.9 |

This is a real-arithmetic bound-looseness audit evaluated on BF16 captures. It does not establish a deterministic BF16 certificate. No deployable certificate, certified fallback policy, or false-certificate rate is claimed.

The registered prerequisites failed, so correction-inclusive MLP latency, cached-decode latency, free-running task quality and physical DRAM traffic were not measured. There are no raw timing samples. The retained-neuron fraction is not a speedup or physical-byte measurement. Historical measurements are not substituted.

## Provenance and reproduction

Artifacts: `artifacts/output_gap_completion/20261014-controlled-v1/`. No result rows or logs were discarded by this report; all fitted ranks/seeds and every saved evaluation split are shown. The notebook is used only as the mathematical rationale.

The manifest has 256 calibration, 256 development, 512 matched and 512 OOD prompts. Template families are disjoint between splits; exact, number-normalized and lexical Jaccard >= 0.72 overlaps with other splits or registered historical prompts are rejected. Numeric/content variants within a family are allowed. The matched split shares task classes, not templates; the OOD split changes task classes. This is a controlled synthetic prompt suite, not natural-language benchmark evidence.

Unique tokenized prompts: 1536; truncated prompts: 0. Training and policy choice preceded held-out model access; `research_log.jsonl` records the frozen checkpoint/selection hash before each held-out opening.

| Artifact | SHA-256 |
| --- | --- |
| evaluation_manifest.jsonl | e2d809957993bb24b8376812ad4ad72332eac0e199171a6c5be2f6cd45dba0ab |
| corrections.pt | 0c5cd4dc75d5f114ea5d47edeec858b8f7dbd231624c0ccd86736e4315fa8825 |
| development_selection.json | 04a592caa17c111f812d3263c3c837bcb6353b59d7925a64a6dae596231c0a5d |
| mask indices | 53b4635f63669ebca0e83a97932805bb36d8198769146eb643a95aab29e1741d |

Historical exclusion sources:

| Source | Rows | SHA-256 |
| --- | --- | --- |
| artifacts/predictive_working_set/20261013-validated-final/evaluation_manifest.jsonl | 896 | 552efc03e4caa86f95ef52a66e4563bdce70f394d0f60536cec36fdec6a0d6c4 |
| artifacts/fused_sparse_swiglu/20261013-validated-final-v2/evaluation_manifest.jsonl | 1536 | 9325a76ac0476f94abfad1a4a4895aef9542bb3897894d763fd743d53a9d5264 |
| data/photonic_parallel_manifest.jsonl | 24 | 8a9f29b1f4f98226e23086557f7a1684bf465d7d64636a662661bfb96d058ad1 |

Frozen executed source fingerprints:

| Source | SHA-256 |
| --- | --- |
| configs/output_gap_completion.json | fdabb0280bdafffb7530ca0a071b231f41ea740fa85314449550b958eb2dba98 |
| experiments/fused_sparse_swiglu.py | 578bc6b044e0da1e699f3c794f4f081c93de694766011672551e4f5ee1230f73 |
| experiments/output_gap_completion.py | ff0482a20b8005775d00f992990a8a1f4f985c703dedb2cabdc3f34b9e0033c5 |
| experiments/output_gap_timing.py | fe4dc28d933a8672fd3edc5b6d9f41677f6ea3a36b04b115b9d3d9d96d94f9cc |
| src/depth_router/fused_swiglu.py | fd17cf407462c4608ba250e5a72d1e1276e0aca506d90127c5219b22f10f8f1b |
| src/depth_router/output_gap.py | b4335cc8fc08e572744cfa0f56fafd2d1cd2d5a4a98e24288dfd526c005b40a6 |
| src/depth_router/output_gap_data.py | 0b4d68a89a210b3ac73d64dcd2d70fbdd9ec3d12472e470a2f5a52bf95ef9368 |

Exact registered sources are preserved in `executed_sources/`; the table above records their original hashes. The snapshots preserve exact original bytes, including Windows CRLF, with Git text conversion disabled. A post-run source change is recorded in `source_change_audit.json`: experiments/output_gap_timing.py: replaced assigned lambda with equivalent named function for Ruff 0.14 E731. The original timing source is `executed_sources/output_gap_timing.py.txt`. Timing executed in this run: False; training/evaluation affected: False. The preregistration and results were retained unchanged.

Execution: NVIDIA GeForce RTX 4070 SUPER (11.99 GiB); Python 3.11.13; PyTorch 2.7.1+cu128; CUDA 12.8; Transformers 4.57.1. Model dtype `torch.bfloat16`; correction dtype `float32`; TF32 allowed: True. Platform: `Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.35`. NVIDIA driver snapshot: `NVIDIA GeForce RTX 4070 SUPER, 610.88, 12282 MiB`. Installed model implementation SHA-256: `a59fa06524227361fb401baf4d177124a27aec146bc01ec931235a9abbab17cb`.

From the repository root in Windows PowerShell, reproduce into a new run directory with this exact command (the pinned model must already exist in the configured local cache):

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_output_gap_completion.ps1 -RunId 20261014-controlled-v1-reproduction
```

The launcher refuses to reuse an existing RunId. The raw CSVs preserve each prompt's reference/candidate token, KL, ties, margins and error metrics. Training hashes/losses, full CIs, bound audit, source fingerprints and environment are retained in the JSON files.

| Finding | Outcome |
| --- | --- |
| Tested | Changing only the correction training objective at matched rank, mask and budget. |
| Passed | paired initialization/minibatches; exact full-width packed control; checkpoint integrity |
| Failed | preregistered held-out objective-improvement test; selected-arm absolute quality gate; useful acceptance from this analytic norm bound |
| Objective improvement | Not established. |
| Cost | Unmeasured: timing and generation skipped by the registered quality/objective gate. Physical DRAM savings remain unknown. |
| Next experiment | None prescribed by this bounded result; no follow-up experiment was run. |
