# DepthRouter final-MLP expert construction audit

4 of 12 individual partitions passed the user-specified coverage and portfolio-advantage gates on development and both held-out splits: B-17, B-29, C-17, C-29. 4 also passed the corresponding lower-confidence-bound checks. Development selected family C (hidden-output reconstruction), which failed the predeclared additional family-level continuation criterion: every one of the 3 seeds must pass. Selected-family failure: insufficient acceptable cheap candidates. The result applies to these partitions, candidate sets, data, and fitting budget; it does not rule out useful cheap computation generally.

C-43 heldout_ood coverage was 78.620 [76.735, 80.587]%; the interval includes the 80% threshold.

Run `20261014-bounded-v1` preserved 48 complete candidate matrices and 3,083,808 state/candidate measurements. Model weights unchanged: `True`. Promotion: `False`.

## Scope and frozen construction recipe

Model `Qwen/Qwen2.5-0.5B-Instruct` at `7ae557604adf67be50417f59c2c2f167def9a775`; research base commit `ec696fddabc59c51292ba1e8f500c9000a50c79b`. Only the last of 24 MLPs is partitioned: 4,864 neurons in eight exactly balanced experts of 608 neurons. Every cheap candidate retains six experts (3,648 neurons, 75% retained), and all 28 subsets are measured. The original gate/up rows and matching down columns remain frozen. There is no residual correction network.

| Arm | Construction |
| --- | --- |
| A | balanced random |
| B | balanced co-activation |
| C | hidden-output reconstruction |
| D | final-distribution KL |

Seeds: `[17, 29, 43]`. Assignment recipe: `{"batch_size": 16, "candidates_per_step": 4, "initial_assignment_strength": 0.1, "initial_noise_std": 0.01, "learning_rate": 0.03, "sinkhorn_iterations": 10, "steps": 100, "temperature_end": 0.2, "temperature_start": 1.0}`. Co-activation recipe: `{"iterations": 8, "sketch_dim": 32}`. C and D share initialization, minibatches, sampled candidate order, exact capacity constraints, optimization budget, and final-iterate selection. Per-token fitting uses the minimum loss among the shared sampled candidates. Hard forwards execute packed BF16 projections; a dense-activation straight-through gradient is an optimization surrogate. C minimizes normalized hidden reconstruction MSE; D minimizes KL(full || candidate) between actual normalized full-vocabulary distributions after residual addition, final RMSNorm and vocabulary head. No shortlist replaces the full distribution.

| Seed | Init/batches/candidates matched | C steps | D steps | Neurons changed C/D |
| --- | --- | --- | --- | --- |
| 17 | True/True/True | 100 | 100 | 4282/4028 |
| 29 | True/True/True | 100 | 100 | 4274/4016 |
| 43 | True/True/True | 100 | 100 | 4302/4018 |

## Data, evaluated positions, and exclusions

Source: `databricks/databricks-dolly-15k` at `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`, license `CC-BY-SA-3.0`. Human-written prompts combine source instructions and available context; source answers are discarded. Matched tasks exclude creative writing and summarization; those categories form the OOD split. Each split has a 12.5% instruction-keyword enterprise stratum, which is a topic screen rather than a production enterprise benchmark. Source documents and exact instruction families are disjoint. Lexical duplicate screening does not establish independence of all semantic template families.

Historical prompt exclusions: 6163 prompt entries from the registered historical manifests. Historical outputs are not training or evaluation evidence in this cycle. Dataset manifest file SHA-256: `ce5d7b1a8126a1ab310d01f4bdf1d887078679f15e64b5a4d00567bda2aed89b`.

Requested continuation positions: `[1, 3, 7, 11]`; these are cached decoding states after that many dense-generated continuation tokens. Every candidate uses the same dense prefix at each state. Contexts stop at EOS; no post-EOS states are treated as observed examples. Prompt token limit: 512 with left truncation. Missing positions are reported below and are excluded from state-level denominators; results are conditional on the observed dense continuations.

| Split | Contexts requested | Contexts with states | States | Missing planned states (EOS) | Truncated prompts |
| --- | --- | --- | --- | --- | --- |
| calibration | 1024 | 1024 | 4079 | 17 | 0 |
| development | 256 | 256 | 1018 | 6 | 0 |
| heldout_matched | 512 | 512 | 2037 | 11 | 0 |
| heldout_ood | 512 | 512 | 2044 | 4 | 0 |

| Split | Continuation position | Observed states |
| --- | --- | --- |
| calibration | 1 | 1024 |
| calibration | 3 | 1024 |
| calibration | 7 | 1020 |
| calibration | 11 | 1011 |
| development | 1 | 256 |
| development | 3 | 256 |
| development | 7 | 255 |
| development | 11 | 251 |
| heldout_matched | 1 | 512 |
| heldout_matched | 3 | 512 |
| heldout_matched | 7 | 511 |
| heldout_matched | 11 | 502 |
| heldout_ood | 1 | 512 |
| heldout_ood | 3 | 512 |
| heldout_ood | 7 | 511 |
| heldout_ood | 11 | 509 |

## Capability for every arm and seed

A locally acceptable candidate matches the original dense greedy token and has KL(full || candidate) strictly below 0.01. Fixed subsets are selected on development by highest acceptable coverage, then lowest mean KL, then canonical candidate index. The same frozen subset is used for every subsequent split. The cheap-only oracle prefers greedy agreement, then minimum KL, then candidate index. The portfolio oracle takes an acceptable cheap candidate when available and dense otherwise. The fixed oracle applies the identical acceptance test to its single subset.

Coverage is the fraction of evaluated token states with an acceptable cheap candidate. Whole contexts, including all their continuation positions, are resampled jointly for 95% percentile intervals (2000 draws). Token-state weighting is retained when EOS creates unequal context lengths. Paired seed comparisons average the specified seeds within each state before the same context draws; intervals are conditional on this seed set. Marginal intervals are not multiplicity-adjusted or deterministic certificates.

Engineering gates require coverage >=80% and portfolio ideal-active-width advantage >=5 percentage points on development and both held-out splits. An ideal oracle with cheap fraction a has average active width 1 - 0.25a; portfolio advantage is 0.25 times (portfolio coverage - fixed coverage). This arithmetic proxy excludes routing, synchronization, movement, kernel launches, and head cost. Point passes are distinguished from findings whose corresponding lower confidence bounds pass.

A separate, predeclared family-level continuation rule requires all three seeds of the development-selected family to pass those gates. Thus individual partition successes remain valid capability findings even when that additional continuation rule stops the cycle. Family ranking uses development mean portfolio advantage, then coverage, then arm name; held-out results cannot replace the selected family or discard a failing seed.

| Arm-seed | Split | Contexts/states | Coverage % [95% CI] | Fixed coverage % | Ideal active width % | Advantage pp [95% CI] | Engineering gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A-17 | calibration | 1024/4079 | 63.300 [61.768, 64.821] | 41.06 | 84.18 | 5.559 [5.240, 5.891] | fit-data diagnostic |
| A-17 | development | 256/1018 | 65.029 [61.942, 68.142] | 42.04 | 83.74 | 5.747 [5.113, 6.422] | FAIL: coverage |
| A-17 | heldout_matched | 512/2037 | 63.083 [60.823, 65.275] | 41.09 | 84.23 | 5.498 [5.051, 5.937] | FAIL: coverage |
| A-17 | heldout_ood | 512/2044 | 58.708 [56.347, 61.163] | 35.67 | 85.32 | 5.761 [5.286, 6.226] | FAIL: coverage |
| A-29 | calibration | 1024/4079 | 64.893 [63.480, 66.339] | 43.86 | 83.78 | 5.259 [4.961, 5.545] | fit-data diagnostic |
| A-29 | development | 256/1018 | 65.324 [62.401, 68.473] | 44.20 | 83.67 | 5.280 [4.652, 5.913] | FAIL: coverage |
| A-29 | heldout_matched | 512/2037 | 63.230 [60.944, 65.307] | 41.97 | 84.19 | 5.314 [4.880, 5.774] | FAIL: coverage |
| A-29 | heldout_ood | 512/2044 | 61.399 [59.149, 63.596] | 41.54 | 84.65 | 4.966 [4.584, 5.379] | FAIL: coverage, portfolio advantage |
| A-43 | calibration | 1024/4079 | 60.628 [58.976, 62.249] | 36.11 | 84.84 | 6.129 [5.809, 6.462] | fit-data diagnostic |
| A-43 | development | 256/1018 | 61.591 [58.333, 64.995] | 37.62 | 84.60 | 5.992 [5.354, 6.642] | FAIL: coverage |
| A-43 | heldout_matched | 512/2037 | 60.628 [58.362, 62.869] | 34.81 | 84.84 | 6.456 [5.979, 6.917] | FAIL: coverage |
| A-43 | heldout_ood | 512/2044 | 56.360 [53.940, 58.688] | 33.90 | 85.91 | 5.614 [5.144, 6.061] | FAIL: coverage |
| B-17 | calibration | 1024/4079 | 86.565 [85.451, 87.727] | 63.32 | 78.36 | 5.810 [5.503, 6.140] | fit-data diagnostic |
| B-17 | development | 256/1018 | 87.721 [85.588, 89.825] | 63.95 | 78.07 | 5.943 [5.315, 6.593] | PASS; resolved |
| B-17 | heldout_matched | 512/2037 | 86.254 [84.698, 87.801] | 62.30 | 78.44 | 5.989 [5.539, 6.440] | PASS; resolved |
| B-17 | heldout_ood | 512/2044 | 84.198 [82.526, 85.847] | 59.88 | 78.95 | 6.079 [5.599, 6.562] | PASS; resolved |
| B-29 | calibration | 1024/4079 | 85.438 [84.241, 86.639] | 61.93 | 78.64 | 5.878 [5.544, 6.197] | fit-data diagnostic |
| B-29 | development | 256/1018 | 85.265 [83.104, 87.463] | 61.10 | 78.68 | 6.041 [5.482, 6.672] | PASS; resolved |
| B-29 | heldout_matched | 512/2037 | 83.456 [81.666, 85.111] | 60.82 | 79.14 | 5.658 [5.220, 6.093] | PASS; resolved |
| B-29 | heldout_ood | 512/2044 | 83.415 [81.654, 85.162] | 58.51 | 79.15 | 6.226 [5.764, 6.711] | PASS; resolved |
| B-43 | calibration | 1024/4079 | 76.563 [75.190, 77.983] | 52.02 | 80.86 | 6.135 [5.815, 6.478] | fit-data diagnostic |
| B-43 | development | 256/1018 | 77.112 [74.314, 79.902] | 51.87 | 80.72 | 6.311 [5.722, 6.936] | FAIL: coverage |
| B-43 | heldout_matched | 512/2037 | 75.749 [73.650, 77.707] | 50.17 | 81.06 | 6.394 [5.931, 6.870] | FAIL: coverage |
| B-43 | heldout_ood | 512/2044 | 72.701 [70.611, 74.632] | 50.68 | 81.82 | 5.504 [5.049, 5.951] | FAIL: coverage |
| C-17 | calibration | 1024/4079 | 86.786 [85.707, 87.868] | 58.99 | 78.30 | 6.950 [6.601, 7.284] | fit-data diagnostic |
| C-17 | development | 256/1018 | 87.525 [85.419, 89.618] | 58.94 | 78.12 | 7.146 [6.501, 7.833] | PASS; resolved |
| C-17 | heldout_matched | 512/2037 | 84.487 [82.850, 86.058] | 56.01 | 78.88 | 7.118 [6.639, 7.629] | PASS; resolved |
| C-17 | heldout_ood | 512/2044 | 84.198 [82.609, 85.805] | 58.02 | 78.95 | 6.544 [6.088, 6.994] | PASS; resolved |
| C-29 | calibration | 1024/4079 | 86.296 [85.204, 87.408] | 56.93 | 78.43 | 7.342 [7.011, 7.679] | fit-data diagnostic |
| C-29 | development | 256/1018 | 86.837 [84.759, 88.933] | 57.37 | 78.29 | 7.367 [6.727, 8.038] | PASS; resolved |
| C-29 | heldout_matched | 512/2037 | 83.996 [82.466, 85.588] | 56.60 | 79.00 | 6.848 [6.379, 7.360] | PASS; resolved |
| C-29 | heldout_ood | 512/2044 | 84.149 [82.574, 85.805] | 55.28 | 78.96 | 7.216 [6.736, 7.710] | PASS; resolved |
| C-43 | calibration | 1024/4079 | 82.005 [80.825, 83.268] | 55.68 | 79.50 | 6.582 [6.251, 6.927] | fit-data diagnostic |
| C-43 | development | 256/1018 | 83.104 [80.728, 85.378] | 59.92 | 79.22 | 5.796 [5.162, 6.429] | PASS; resolved |
| C-43 | heldout_matched | 512/2037 | 80.412 [78.498, 82.160] | 55.18 | 79.90 | 6.308 [5.861, 6.768] | PASS; unresolved CI |
| C-43 | heldout_ood | 512/2044 | 78.620 [76.735, 80.587] | 52.89 | 80.34 | 6.433 [5.959, 6.902] | FAIL: coverage |
| D-17 | calibration | 1024/4079 | 64.746 [63.191, 66.275] | 42.76 | 83.81 | 5.498 [5.166, 5.824] | fit-data diagnostic |
| D-17 | development | 256/1018 | 67.092 [63.956, 70.207] | 44.20 | 83.23 | 5.722 [5.049, 6.422] | FAIL: coverage |
| D-17 | heldout_matched | 512/2037 | 62.838 [60.570, 65.032] | 42.17 | 84.29 | 5.167 [4.724, 5.636] | FAIL: coverage |
| D-17 | heldout_ood | 512/2044 | 61.888 [59.578, 64.202] | 40.56 | 84.53 | 5.333 [4.905, 5.782] | FAIL: coverage |
| D-29 | calibration | 1024/4079 | 64.599 [62.969, 66.169] | 39.64 | 83.85 | 6.239 [5.893, 6.574] | fit-data diagnostic |
| D-29 | development | 256/1018 | 65.815 [62.795, 68.952] | 39.78 | 83.55 | 6.508 [5.820, 7.230] | FAIL: coverage |
| D-29 | heldout_matched | 512/2037 | 63.181 [60.855, 65.440] | 39.18 | 84.20 | 6.001 [5.530, 6.428] | FAIL: coverage |
| D-29 | heldout_ood | 512/2044 | 59.344 [56.954, 61.765] | 37.13 | 85.16 | 5.553 [5.093, 6.003] | FAIL: coverage |
| D-43 | calibration | 1024/4079 | 66.021 [64.492, 67.638] | 41.53 | 83.49 | 6.123 [5.803, 6.463] | fit-data diagnostic |
| D-43 | development | 256/1018 | 65.619 [62.524, 68.690] | 41.85 | 83.60 | 5.943 [5.305, 6.582] | FAIL: coverage |
| D-43 | heldout_matched | 512/2037 | 65.930 [63.672, 68.126] | 40.45 | 83.52 | 6.370 [5.885, 6.863] | FAIL: coverage |
| D-43 | heldout_ood | 512/2044 | 62.622 [60.255, 64.853] | 39.58 | 84.34 | 5.761 [5.296, 6.226] | FAIL: coverage |

Calibration rows use the development-frozen fixed subset retrospectively for training-data diagnostics and do not contribute to promotion gates.

## Raw fidelity and oracle dense routing

Original dense self-reference is 100% greedy agreement and zero KL by definition. The following table retains raw fixed and cheap-only oracle errors separately from the dense-option oracles. Both dense-option oracles have 100% greedy agreement by construction and use teacher information unavailable at deployment.

| Arm-seed | Split | Frozen fixed experts | Fixed agree % / KL | Cheap oracle agree % / KL | Portfolio+dense KL / dense % | Fixed+dense KL / dense % |
| --- | --- | --- | --- | --- | --- | --- |
| A-17 | calibration | 1,2,3,4,6,7 | 89.14/0.144295 | 99.93/0.0172098 | 0.00135582/36.70 | 0.000917534/58.94 |
| A-17 | development | 1,2,3,4,6,7 | 89.59/0.138424 | 100.00/0.0162722 | 0.00135458/34.97 | 0.000953062/57.96 |
| A-17 | heldout_matched | 1,2,3,4,6,7 | 87.92/0.145708 | 99.90/0.0180642 | 0.00139806/36.92 | 0.00096337/58.91 |
| A-17 | heldout_ood | 1,2,3,4,6,7 | 88.85/0.137659 | 99.95/0.0174879 | 0.0012956/41.29 | 0.00078305/64.33 |
| A-29 | calibration | 1,2,4,5,6,7 | 88.72/0.156269 | 99.95/0.0155378 | 0.00136196/35.11 | 0.000931918/56.14 |
| A-29 | development | 1,2,4,5,6,7 | 89.10/0.128973 | 99.90/0.0149675 | 0.00138432/34.68 | 0.00100116/55.80 |
| A-29 | heldout_matched | 1,2,4,5,6,7 | 89.64/0.14591 | 99.95/0.0156455 | 0.00134367/36.77 | 0.000943121/58.03 |
| A-29 | heldout_ood | 1,2,4,5,6,7 | 87.87/0.144216 | 100.00/0.0167526 | 0.00140642/38.60 | 0.00102613/58.46 |
| A-43 | calibration | 0,3,4,5,6,7 | 89.78/0.13574 | 99.98/0.0172011 | 0.00121194/39.37 | 0.000833829/63.89 |
| A-43 | development | 0,3,4,5,6,7 | 87.52/0.1384 | 99.80/0.0155267 | 0.00117209/38.41 | 0.000874179/62.38 |
| A-43 | heldout_matched | 0,3,4,5,6,7 | 87.43/0.138596 | 100.00/0.0166857 | 0.00128248/39.37 | 0.000855181/65.19 |
| A-43 | heldout_ood | 0,3,4,5,6,7 | 88.11/0.125559 | 100.00/0.0187034 | 0.00124653/43.64 | 0.000871668/66.10 |
| B-17 | calibration | 0,1,3,4,5,7 | 93.65/0.0688245 | 99.93/0.00555403 | 0.0015745/13.43 | 0.00153941/36.68 |
| B-17 | development | 0,1,3,4,5,7 | 93.03/0.0540016 | 100.00/0.00457866 | 0.00154338/12.28 | 0.00150966/36.05 |
| B-17 | heldout_matched | 0,1,3,4,5,7 | 94.31/0.0637119 | 99.95/0.00540291 | 0.00161143/13.75 | 0.00145852/37.70 |
| B-17 | heldout_ood | 0,1,3,4,5,7 | 94.57/0.0431028 | 100.00/0.00524479 | 0.00168473/15.80 | 0.00152348/40.12 |
| B-29 | calibration | 0,3,4,5,6,7 | 92.79/0.0722574 | 99.93/0.00613537 | 0.00166331/14.56 | 0.00143523/38.07 |
| B-29 | development | 0,3,4,5,6,7 | 92.44/0.0729189 | 100.00/0.00534655 | 0.00155655/14.73 | 0.00139935/38.90 |
| B-29 | heldout_matched | 0,3,4,5,6,7 | 93.37/0.0777826 | 99.95/0.00642149 | 0.0016318/16.54 | 0.00140735/39.18 |
| B-29 | heldout_ood | 0,3,4,5,6,7 | 92.17/0.0790185 | 99.90/0.00651518 | 0.00169705/16.59 | 0.00147733/41.49 |
| B-43 | calibration | 0,1,2,3,4,5 | 90.44/0.121509 | 99.98/0.00869146 | 0.00155928/23.44 | 0.0011198/47.98 |
| B-43 | development | 0,1,2,3,4,5 | 90.96/0.0962253 | 99.90/0.00772525 | 0.00147271/22.89 | 0.00101951/48.13 |
| B-43 | heldout_matched | 0,1,2,3,4,5 | 91.51/0.101068 | 99.95/0.0100754 | 0.00157289/24.25 | 0.00113741/49.83 |
| B-43 | heldout_ood | 0,1,2,3,4,5 | 90.46/0.107357 | 99.95/0.0104002 | 0.00156898/27.30 | 0.00119679/49.32 |
| C-17 | calibration | 1,2,3,5,6,7 | 92.33/0.0966719 | 99.85/0.00626199 | 0.00152794/13.21 | 0.00115102/41.01 |
| C-17 | development | 1,2,3,5,6,7 | 93.32/0.0835769 | 99.90/0.00430555 | 0.00146694/12.48 | 0.00103387/41.06 |
| C-17 | heldout_matched | 1,2,3,5,6,7 | 92.93/0.0939877 | 99.80/0.00588716 | 0.00149137/15.51 | 0.00115254/43.99 |
| C-17 | heldout_ood | 1,2,3,5,6,7 | 92.91/0.0806971 | 99.90/0.00556793 | 0.00156622/15.80 | 0.00126945/41.98 |
| C-29 | calibration | 0,1,2,4,6,7 | 91.96/0.106442 | 99.90/0.00551959 | 0.00156878/13.70 | 0.00114713/43.07 |
| C-29 | development | 0,1,2,4,6,7 | 91.36/0.0957775 | 100.00/0.00717683 | 0.00150927/13.16 | 0.00113928/42.63 |
| C-29 | heldout_matched | 0,1,2,4,6,7 | 91.70/0.105036 | 99.95/0.00584009 | 0.00151199/16.00 | 0.00119768/43.40 |
| C-29 | heldout_ood | 0,1,2,4,6,7 | 91.73/0.0950817 | 100.00/0.00625109 | 0.00173946/15.85 | 0.00131168/44.72 |
| C-43 | calibration | 1,2,3,4,6,7 | 92.55/0.0916877 | 99.90/0.00620386 | 0.00153416/17.99 | 0.00109247/44.32 |
| C-43 | development | 1,2,3,4,6,7 | 92.83/0.0848065 | 100.00/0.00829559 | 0.00149665/16.90 | 0.00126611/40.08 |
| C-43 | heldout_matched | 1,2,3,4,6,7 | 91.56/0.0903168 | 99.95/0.00656464 | 0.00146773/19.59 | 0.00116496/44.82 |
| C-43 | heldout_ood | 1,2,3,4,6,7 | 92.42/0.0696735 | 99.95/0.00676085 | 0.00151826/21.38 | 0.00120689/47.11 |
| D-17 | calibration | 0,2,4,5,6,7 | 89.56/0.151389 | 100.00/0.014355 | 0.00136405/35.25 | 0.000853356/57.24 |
| D-17 | development | 0,2,4,5,6,7 | 88.31/0.113656 | 99.90/0.0134864 | 0.00144389/32.91 | 0.000870982/55.80 |
| D-17 | heldout_matched | 0,2,4,5,6,7 | 88.91/0.14838 | 100.00/0.016156 | 0.0012614/37.16 | 0.000818085/57.83 |
| D-17 | heldout_ood | 0,2,4,5,6,7 | 88.85/0.119812 | 99.90/0.0172298 | 0.00146573/38.11 | 0.000958782/59.44 |
| D-29 | calibration | 0,2,3,4,6,7 | 88.06/0.17202 | 99.93/0.016802 | 0.0013689/35.40 | 0.000798021/60.36 |
| D-29 | development | 0,2,3,4,6,7 | 88.90/0.130982 | 100.00/0.0145851 | 0.00131253/34.18 | 0.000785107/60.22 |
| D-29 | heldout_matched | 0,2,3,4,6,7 | 89.35/0.156578 | 100.00/0.0154296 | 0.00130168/36.82 | 0.000824244/60.82 |
| D-29 | heldout_ood | 0,2,3,4,6,7 | 88.31/0.117199 | 99.90/0.0171942 | 0.00130097/40.66 | 0.000756714/62.87 |
| D-43 | calibration | 1,2,3,4,6,7 | 89.53/0.140942 | 99.98/0.0129581 | 0.00141523/33.98 | 0.000874591/58.47 |
| D-43 | development | 1,2,3,4,6,7 | 89.29/0.126419 | 100.00/0.0120257 | 0.00127487/34.38 | 0.000853545/58.15 |
| D-43 | heldout_matched | 1,2,3,4,6,7 | 89.25/0.137596 | 100.00/0.0134005 | 0.00142632/34.07 | 0.000856313/59.55 |
| D-43 | heldout_ood | 1,2,3,4,6,7 | 89.58/0.112199 | 99.95/0.0134846 | 0.00137818/37.38 | 0.000844649/60.42 |

## Paired D minus C construction-objective comparison

| Split | Metric | Difference [95% CI] | Units |
| --- | --- | --- | --- |
| calibration | cheap_coverage | -19.907 [-20.942, -18.823] | percentage points |
| calibration | fixed_cheap_coverage | -15.886 [-16.830, -14.881] | percentage points |
| calibration | oracle_agreement | +0.082 [+0.000, +0.180] | percentage points |
| calibration | oracle_kl | +0.008710 [+0.007839, +0.009660] | raw KL |
| calibration | portfolio_advantage | -1.005 [-1.306, -0.711] | percentage points |
| development | cheap_coverage | -19.646 [-21.732, -17.662] | percentage points |
| development | fixed_cheap_coverage | -16.798 [-18.861, -14.663] | percentage points |
| development | oracle_agreement | +0.000 [-0.098, +0.098] | percentage points |
| development | oracle_kl | +0.006773 [+0.002810, +0.009260] | raw KL |
| development | portfolio_advantage | -0.712 [-1.341, -0.090] | percentage points |
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

Positive coverage, agreement, or portfolio-advantage differences favor D; negative KL differences favor D. These paired estimates do not imply that either arm passes the absolute engineering gates.

## Numerical controls and actual model execution

The gating all-eight control uses original neuron index order, matching the globally index-sorted candidate layout. A separate expert-contiguous permutation control measures reduction-order sensitivity. Its results are shown explicitly; it is not silently substituted for the validated sorted layout. BF16 exactness is assessed from observations, not inferred from real-arithmetic permutation identities.

| Arm-seed | Split | All-eight layout | Agreement % [95% CI] | Mean KL | >=99% and KL<0.01 |
| --- | --- | --- | --- | --- | --- |
| A-17 | calibration | control | 100.000 [100.000, 100.000] | 0 | PASS |
| A-17 | calibration | permuted_control | 99.779 [99.632, 99.902] | 0.00013296795 | PASS |
| A-29 | calibration | control | 100.000 [100.000, 100.000] | 0 | PASS |
| A-29 | calibration | permuted_control | 99.779 [99.632, 99.902] | 0.0001494244 | PASS |
| A-43 | calibration | control | 100.000 [100.000, 100.000] | 0 | PASS |
| A-43 | calibration | permuted_control | 99.877 [99.730, 99.976] | 0.0001351317 | PASS |
| B-17 | calibration | control | 100.000 [100.000, 100.000] | 0 | PASS |
| B-17 | calibration | permuted_control | 99.853 [99.730, 99.951] | 0.00014822857 | PASS |
| B-29 | calibration | control | 100.000 [100.000, 100.000] | 0 | PASS |
| B-29 | calibration | permuted_control | 99.779 [99.632, 99.902] | 0.00013716501 | PASS |
| B-43 | calibration | control | 100.000 [100.000, 100.000] | 0 | PASS |
| B-43 | calibration | permuted_control | 99.730 [99.535, 99.878] | 0.0001484999 | PASS |
| C-17 | calibration | control | 100.000 [100.000, 100.000] | 0 | PASS |
| C-17 | calibration | permuted_control | 99.828 [99.705, 99.951] | 0.00015275086 | PASS |
| C-29 | calibration | control | 100.000 [100.000, 100.000] | 0 | PASS |
| C-29 | calibration | permuted_control | 99.804 [99.657, 99.926] | 0.00015368518 | PASS |
| C-43 | calibration | control | 100.000 [100.000, 100.000] | 0 | PASS |
| C-43 | calibration | permuted_control | 99.828 [99.682, 99.951] | 0.00014032404 | PASS |
| D-17 | calibration | control | 100.000 [100.000, 100.000] | 0 | PASS |
| D-17 | calibration | permuted_control | 99.828 [99.681, 99.951] | 0.00014370758 | PASS |
| D-29 | calibration | control | 100.000 [100.000, 100.000] | 0 | PASS |
| D-29 | calibration | permuted_control | 99.804 [99.657, 99.926] | 0.00013419013 | PASS |
| D-43 | calibration | control | 100.000 [100.000, 100.000] | 0 | PASS |
| D-43 | calibration | permuted_control | 99.755 [99.583, 99.902] | 0.0001454891 | PASS |
| A-17 | development | control | 100.000 [100.000, 100.000] | 0 | PASS |
| A-17 | development | permuted_control | 99.607 [99.212, 99.902] | 0.00014093214 | PASS |
| A-29 | development | control | 100.000 [100.000, 100.000] | 0 | PASS |
| A-29 | development | permuted_control | 99.902 [99.704, 100.000] | 0.00013392553 | PASS |
| A-43 | development | control | 100.000 [100.000, 100.000] | 0 | PASS |
| A-43 | development | permuted_control | 99.804 [99.506, 100.000] | 0.00014684527 | PASS |
| B-17 | development | control | 100.000 [100.000, 100.000] | 0 | PASS |
| B-17 | development | permuted_control | 99.902 [99.609, 100.000] | 0.00013843543 | PASS |
| B-29 | development | control | 100.000 [100.000, 100.000] | 0 | PASS |
| B-29 | development | permuted_control | 99.902 [99.704, 100.000] | 0.00014081155 | PASS |
| B-43 | development | control | 100.000 [100.000, 100.000] | 0 | PASS |
| B-43 | development | permuted_control | 100.000 [100.000, 100.000] | 0.00013877153 | PASS |
| C-17 | development | control | 100.000 [100.000, 100.000] | 0 | PASS |
| C-17 | development | permuted_control | 100.000 [100.000, 100.000] | 0.00014376304 | PASS |
| C-29 | development | control | 100.000 [100.000, 100.000] | 0 | PASS |
| C-29 | development | permuted_control | 99.804 [99.508, 100.000] | 0.00013100532 | PASS |
| C-43 | development | control | 100.000 [100.000, 100.000] | 0 | PASS |
| C-43 | development | permuted_control | 99.902 [99.704, 100.000] | 0.00015092092 | PASS |
| D-17 | development | control | 100.000 [100.000, 100.000] | 0 | PASS |
| D-17 | development | permuted_control | 99.804 [99.508, 100.000] | 0.00014082946 | PASS |
| D-29 | development | control | 100.000 [100.000, 100.000] | 0 | PASS |
| D-29 | development | permuted_control | 99.902 [99.704, 100.000] | 0.00013843952 | PASS |
| D-43 | development | control | 100.000 [100.000, 100.000] | 0 | PASS |
| D-43 | development | permuted_control | 99.902 [99.704, 100.000] | 0.00015368723 | PASS |
| A-17 | heldout_matched | control | 100.000 [100.000, 100.000] | 0 | PASS |
| A-17 | heldout_matched | permuted_control | 99.902 [99.754, 100.000] | 0.0001443753 | PASS |
| A-29 | heldout_matched | control | 100.000 [100.000, 100.000] | 0 | PASS |
| A-29 | heldout_matched | permuted_control | 99.705 [99.459, 99.902] | 0.0001442567 | PASS |
| A-43 | heldout_matched | control | 100.000 [100.000, 100.000] | 0 | PASS |
| A-43 | heldout_matched | permuted_control | 99.804 [99.559, 99.951] | 0.00015320918 | PASS |
| B-17 | heldout_matched | control | 100.000 [100.000, 100.000] | 0 | PASS |
| B-17 | heldout_matched | permuted_control | 99.853 [99.657, 100.000] | 0.00015415268 | PASS |
| B-29 | heldout_matched | control | 100.000 [100.000, 100.000] | 0 | PASS |
| B-29 | heldout_matched | permuted_control | 99.853 [99.656, 100.000] | 0.00016109996 | PASS |
| B-43 | heldout_matched | control | 100.000 [100.000, 100.000] | 0 | PASS |
| B-43 | heldout_matched | permuted_control | 99.755 [99.509, 99.951] | 0.00014336958 | PASS |
| C-17 | heldout_matched | control | 100.000 [100.000, 100.000] | 0 | PASS |
| C-17 | heldout_matched | permuted_control | 99.853 [99.657, 100.000] | 0.00015525216 | PASS |
| C-29 | heldout_matched | control | 100.000 [100.000, 100.000] | 0 | PASS |
| C-29 | heldout_matched | permuted_control | 99.853 [99.657, 100.000] | 0.00016907277 | PASS |
| C-43 | heldout_matched | control | 100.000 [100.000, 100.000] | 0 | PASS |
| C-43 | heldout_matched | permuted_control | 99.951 [99.852, 100.000] | 0.00016088832 | PASS |
| D-17 | heldout_matched | control | 100.000 [100.000, 100.000] | 0 | PASS |
| D-17 | heldout_matched | permuted_control | 99.755 [99.509, 99.951] | 0.00014636637 | PASS |
| D-29 | heldout_matched | control | 100.000 [100.000, 100.000] | 0 | PASS |
| D-29 | heldout_matched | permuted_control | 99.804 [99.558, 99.951] | 0.00014303942 | PASS |
| D-43 | heldout_matched | control | 100.000 [100.000, 100.000] | 0 | PASS |
| D-43 | heldout_matched | permuted_control | 99.755 [99.509, 99.951] | 0.00014934984 | PASS |
| A-17 | heldout_ood | control | 100.000 [100.000, 100.000] | 0 | PASS |
| A-17 | heldout_ood | permuted_control | 99.755 [99.511, 99.951] | 0.0001346401 | PASS |
| A-29 | heldout_ood | control | 100.000 [100.000, 100.000] | 0 | PASS |
| A-29 | heldout_ood | permuted_control | 99.804 [99.608, 99.951] | 0.00014747527 | PASS |
| A-43 | heldout_ood | control | 100.000 [100.000, 100.000] | 0 | PASS |
| A-43 | heldout_ood | permuted_control | 99.658 [99.412, 99.902] | 0.0001514546 | PASS |
| B-17 | heldout_ood | control | 100.000 [100.000, 100.000] | 0 | PASS |
| B-17 | heldout_ood | permuted_control | 99.706 [99.462, 99.902] | 0.00015280883 | PASS |
| B-29 | heldout_ood | control | 100.000 [100.000, 100.000] | 0 | PASS |
| B-29 | heldout_ood | permuted_control | 99.804 [99.608, 99.951] | 0.00014495365 | PASS |
| B-43 | heldout_ood | control | 100.000 [100.000, 100.000] | 0 | PASS |
| B-43 | heldout_ood | permuted_control | 99.560 [99.267, 99.805] | 0.00015227011 | PASS |
| C-17 | heldout_ood | control | 100.000 [100.000, 100.000] | 0 | PASS |
| C-17 | heldout_ood | permuted_control | 99.560 [99.266, 99.805] | 0.0001557701 | PASS |
| C-29 | heldout_ood | control | 100.000 [100.000, 100.000] | 0 | PASS |
| C-29 | heldout_ood | permuted_control | 99.706 [99.461, 99.902] | 0.00017085798 | PASS |
| C-43 | heldout_ood | control | 100.000 [100.000, 100.000] | 0 | PASS |
| C-43 | heldout_ood | permuted_control | 99.755 [99.556, 99.951] | 0.00016556727 | PASS |
| D-17 | heldout_ood | control | 100.000 [100.000, 100.000] | 0 | PASS |
| D-17 | heldout_ood | permuted_control | 99.755 [99.511, 99.951] | 0.00015299862 | PASS |
| D-29 | heldout_ood | control | 100.000 [100.000, 100.000] | 0 | PASS |
| D-29 | heldout_ood | permuted_control | 99.706 [99.462, 99.902] | 0.00016711053 | PASS |
| D-43 | heldout_ood | control | 100.000 [100.000, 100.000] | 0 | PASS |
| D-43 | heldout_ood | permuted_control | 99.755 [99.511, 99.951] | 0.00013795531 | PASS |

| Arm-seed | Full-model substitution checks | Checked state/candidates | Bitwise exact cached logits | Maximum absolute logit error |
| --- | --- | --- | --- | --- |
| A-17 | 29 | 464 | True | 0.0 |
| A-29 | 29 | 464 | True | 0.0 |
| A-43 | 29 | 464 | True | 0.0 |
| B-17 | 29 | 464 | True | 0.0 |
| B-29 | 29 | 464 | True | 0.0 |
| B-43 | 29 | 464 | True | 0.0 |
| C-17 | 29 | 464 | True | 0.0 |
| C-29 | 29 | 464 | True | 0.0 |
| C-43 | 29 | 464 | True | 0.0 |
| D-17 | 29 | 464 | True | 0.0 |
| D-29 | 29 | 464 | True | 0.0 |
| D-43 | 29 | 464 | True | 0.0 |

Cached dense residual/norm/head reconstruction is checked on each captured batch. Full-model substitution checks cover all 28 candidates and the sorted all-eight control on the recorded calibration replay batch; they do not establish correctness for untested batch shapes or deployment kernels.

## Gate decision, selection, and execution

| Arm-seed | All-split point pass | Resolved pass | Failure reasons |
| --- | --- | --- | --- |
| A-17 | False | False | development:insufficient_cheap_candidate_coverage; heldout:insufficient_cheap_candidate_coverage; ood:insufficient_cheap_candidate_coverage |
| A-29 | False | False | development:insufficient_cheap_candidate_coverage; heldout:insufficient_cheap_candidate_coverage; ood:insufficient_cheap_candidate_coverage; ood:insufficient_advantage_over_fixed_subset |
| A-43 | False | False | development:insufficient_cheap_candidate_coverage; heldout:insufficient_cheap_candidate_coverage; ood:insufficient_cheap_candidate_coverage |
| B-17 | True | True | none |
| B-29 | True | True | none |
| B-43 | False | False | development:insufficient_cheap_candidate_coverage; heldout:insufficient_cheap_candidate_coverage; ood:insufficient_cheap_candidate_coverage |
| C-17 | True | True | none |
| C-29 | True | True | none |
| C-43 | False | False | ood:insufficient_cheap_candidate_coverage |
| D-17 | False | False | development:insufficient_cheap_candidate_coverage; heldout:insufficient_cheap_candidate_coverage; ood:insufficient_cheap_candidate_coverage |
| D-29 | False | False | development:insufficient_cheap_candidate_coverage; heldout:insufficient_cheap_candidate_coverage; ood:insufficient_cheap_candidate_coverage |
| D-43 | False | False | development:insufficient_cheap_candidate_coverage; heldout:insufficient_cheap_candidate_coverage; ood:insufficient_cheap_candidate_coverage |

Phase 2: `skipped_phase1_gate_failed`. Phase 3: `not_run_requires_phase2_quality_pass`. If the capability gate fails, learned routing, free-generation task evaluation and timing are skipped under the stopping rule. An oracle does not establish recognition from pre-execution inputs, and dense fallback fidelity does not establish useful savings.

Measured latency: `None`. Measured physical DRAM traffic: `None`. Counter status: `not_attempted_prerequisite_quality_gate`. Null means unknown, not zero. Without a passing selector there is no selector-plus-MLP timing, end-to-end cached-decoding benchmark, physical traffic measurement, or free-generation task-success result. Offline candidate-search costs are not deployment latency measurements.

Bounded audit elapsed wall time: `449.735045681` seconds; single-GPU ceiling: 14400 seconds. This includes loading, capture, construction, exhaustive offline measurement, and CPU waits; it is not GPU kernel time. Removing 25% of one of 24 equal-width MLPs saves at most approximately 1.04% of total MLP arithmetic before overhead. No model-wide speedup follows from that proxy.

## Execution environment

| Field | Recorded value |
| --- | --- |
| gpu | NVIDIA GeForce RTX 4070 SUPER |
| torch | 2.7.1+cu128 |
| transformers | 4.57.1 |
| cuda | 12.8 |
| python | 3.11.13 |
| platform | Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.35 |
| model_class | Qwen2ForCausalLM |
| model_source_sha256 | a59fa06524227361fb401baf4d177124a27aec146bc01ec931235a9abbab17cb |
| dtype | torch.bfloat16 |
| attention | sdpa |
| allow_tf32 | False |
| cuda_graphs | False |
| stage | offline_quality_not_runtime_benchmark |

Capture batch size: 16. Full environment and NVIDIA driver output are preserved in `environment.json`.

## Relation to earlier evidence and primary sources

This is a separate diagnostic. Earlier individual-neuron prediction and naive contiguous pages failed their quality gates. The fused sparse study measured an isolated kernel benefit but dynamic prediction/top-k/gather overhead overwhelmed it; the graph-captured BF16 dense operator remained the strongest exact baseline. Those historical timings are not new measurements in this audit. PR 16 held the mask fixed while changing correction objectives; none of its 12 correction fits passed development. This audit changes expert construction and adds no correction. Earlier contribution-ranking oracles were heuristics; this audit exhaustively searches only its 28 frozen six-of-eight candidates, not all neuron subsets or possible partitions. Synthetic notebook examples are mathematical checks, not Qwen or hardware evidence.

[DOT-MoE](https://proceedings.mlr.press/v306/bamba26a.html) motivates balanced transport-based neuron assignments and straight-through optimization. [DynaMoE](https://arxiv.org/abs/2502.12325) motivates predicting sufficient computation per token. [Goal-oriented model reduction](https://arxiv.org/abs/1603.00336) motivates evaluating a quantity of interest rather than only intermediate-state reconstruction. These are conceptual foundations; this bounded final-MLP audit claims neither novelty nor faithful reproduction of those methods.

## Reproduction and retained artifacts

Run from the repository root with a fresh run ID:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_expert_construction.ps1 -RunId fresh-expert-audit
```

Regenerate this report without model execution:

```text
python experiments/expert_construction_report.py --run-dir artifacts/expert_construction/20261014-bounded-v1
```

`preregistration.json` and `registered_sources/` preserve source/config hashes before model access. `heldout_seal.json` freezes partitions, development-selected subsets, family selection and preregistration before held-out captures. `construction-*.json` preserves paired schedules/hashes and losses; `partition-*.json` contains every neuron assignment. `states-*.jsonl` preserves context IDs, actual prefix token IDs, positions, dense tokens and margins. All scalar candidate matrices, including failed arms, remain in `matrix-*.npz`. Large activation caches are local reproducible intermediates; capture files record their hashes. `promotion_decision.json` is the machine-readable gate record; `paired_D_minus_C.json` holds paired intervals. `incomplete.json`, when present, records interrupted work without claiming completion.

| Raw matrix | States | SHA-256 |
| --- | --- | --- |
| matrix-calibration-A-17.npz | 4079 | b4f327a4572ef4c7b32791a0a9c0a9cabab029cbd4258b1d6fc9ee6fbea01a29 |
| matrix-calibration-A-29.npz | 4079 | 83fa457ec29c20208156abccdfbb9d1a50daf05b27f1de6d347f6a734fcb6d50 |
| matrix-calibration-A-43.npz | 4079 | b2660e0463745b557fe044552a0ac68937e124d5888c0b16ecbe72225b91e57e |
| matrix-calibration-B-17.npz | 4079 | a41f25dbcdcdcbd70b0129850b4fbe2fa376d392cb4b78c790d4e1c60ba253cc |
| matrix-calibration-B-29.npz | 4079 | 5b590611659ade6c390e7e062f723b969590e73b0d7c04b85d16588eb46b5897 |
| matrix-calibration-B-43.npz | 4079 | 62f72f53504e92ea76520e587ec2f7a0defb6c86b46552bc5a45140a2778c982 |
| matrix-calibration-C-17.npz | 4079 | c26a575aae2709a092b2545bb3a388e8f59f8692f039ed500ae91e5e8ea3418f |
| matrix-calibration-C-29.npz | 4079 | ecb35148cbb6a223321144d83d66bcf5a8aab5c1fff20762e1e4db986e20c0f6 |
| matrix-calibration-C-43.npz | 4079 | fd918805d3cb4624cc5440229a840a043f9f14530a77f5017589dfe95574fd48 |
| matrix-calibration-D-17.npz | 4079 | 0d25f0b8c4cef3bdd4937eba4e8e3d65d9fcf777dae2f38f600552b1ee701ac3 |
| matrix-calibration-D-29.npz | 4079 | 356a92c93c8c944f4d165b0ea6a9c4f41188b0de0a05558359267403851fda8b |
| matrix-calibration-D-43.npz | 4079 | 47993e7357e248b5f3047e43ad76d1a0e84416e831722f584aa43cdd6ed6c412 |
| matrix-development-A-17.npz | 1018 | 6740724c3b3d55c5bf9750656b636c56d2527e77175a06eabd05ed0a4debd2f3 |
| matrix-development-A-29.npz | 1018 | 36718266b67be72c4a0bc03f34ba518751cc7287902c855beb31e9af287bd709 |
| matrix-development-A-43.npz | 1018 | 5be2e554ea35e16ccacc0cc9032d249349b4de9bfef517841b822734928123b2 |
| matrix-development-B-17.npz | 1018 | a5189a7ecdcf93bf914628ebb284f812f8f450c6db6afa97fe39859693d0bfa5 |
| matrix-development-B-29.npz | 1018 | 770e25b9f955673a0aa6c9e4665503202b553044b004bb4d174433058a0770d0 |
| matrix-development-B-43.npz | 1018 | 0cbd2bc42b819eaacf4805fd1af8fe2f6eefc5fe29d48e4a96b4c6494ea08bae |
| matrix-development-C-17.npz | 1018 | 3f9dd6f4dd126eecf08e2476928878f10e17fbb67d07059bc1a032a47eb20c54 |
| matrix-development-C-29.npz | 1018 | 2cb54a9ee3e5533024db4e309f0eb9cc4806ee3e0e69dd0711cd836b76efd634 |
| matrix-development-C-43.npz | 1018 | 51c64906da01f000ce172353fa3fe61efb57ef23fcb23cdcdd7cd400f2c27706 |
| matrix-development-D-17.npz | 1018 | 8dde20856ae108e1d3a9e51346677130ca8d4332752e6e69acbc2c74d301d22c |
| matrix-development-D-29.npz | 1018 | e535eb29890d151fbea9dd8e0f08884b83d7d3af0b5976822c2e700acacbf1a9 |
| matrix-development-D-43.npz | 1018 | 417889f54daed4fa482d7e6a8d744cb0b11c9aad962542affc0da6fa74dc62d5 |
| matrix-heldout_matched-A-17.npz | 2037 | 2d96925730d9276b676f9bb9ca0cf1fc6b5943158de8c17463e09fd3027bdb84 |
| matrix-heldout_matched-A-29.npz | 2037 | 3dceb3481df57d02dd823ae6ac1938112c707fcb996205d5f8e94daf70b516fb |
| matrix-heldout_matched-A-43.npz | 2037 | ee2f169f7c675d49103c46c4f6566934a33f06ce161b5ac74a97e3c1731f6c40 |
| matrix-heldout_matched-B-17.npz | 2037 | 8dcb66f46c9620ffeee502a1df82e8d9cc56ec56b954dde520186af0f8d521b0 |
| matrix-heldout_matched-B-29.npz | 2037 | 6a9f2fede1e7785fb2e8f161b43fb45193d31823aaedfba30ca751faa37e5456 |
| matrix-heldout_matched-B-43.npz | 2037 | 051b06aa9ee5d1913312f2a1be8af7d634520460c866cb82dec8fe8026b9a1bc |
| matrix-heldout_matched-C-17.npz | 2037 | ddbe2d278cff21b0e19f517dabdfd9e34656b04f857f3c0817c1f90694b807b0 |
| matrix-heldout_matched-C-29.npz | 2037 | be8ddba3e6db6b0afc1b2b9122e85d9158318671ed026f7e1a82c8fb7da1ee92 |
| matrix-heldout_matched-C-43.npz | 2037 | 0bf241ee5604e290f9ebbb323867ca8c5ddacedbd8c1266bac84668143e5fc37 |
| matrix-heldout_matched-D-17.npz | 2037 | 0f3079f4b2170f440faf396972d9f1997481e86d7e421e7ba3458c5b374c4580 |
| matrix-heldout_matched-D-29.npz | 2037 | eff70315a57c35e1ee99393a0477c9b34cc0339698b34beab3abd0b7a6f444cd |
| matrix-heldout_matched-D-43.npz | 2037 | afdb0dfddfd903ac65514886948406b9a212beacf261881ef77c0c0dc230279f |
| matrix-heldout_ood-A-17.npz | 2044 | dc817ea15883184d00c38a058f1b27914da442a09e51b571adabd26d8e408466 |
| matrix-heldout_ood-A-29.npz | 2044 | f171b3097da2d91cf6ee2eb16d4a631b85ed3df2f8d41b17a4e719831d2ff741 |
| matrix-heldout_ood-A-43.npz | 2044 | 0c1077dd6d00ffcc09c595f96cbb08f83339ce60c9f45d9a7354e35e469498a0 |
| matrix-heldout_ood-B-17.npz | 2044 | 70dc81099446b891059ec0c03d8e9a02ddb406efd4790d9fc876c738c7e07d96 |
| matrix-heldout_ood-B-29.npz | 2044 | 47578c496a52c5dccd12736b9740e74982bbed525a77cf535f819612a8f579a0 |
| matrix-heldout_ood-B-43.npz | 2044 | d791daafe2a489aba7788b18207b9e3644da588022a5f165129d7eac20a22fc2 |
| matrix-heldout_ood-C-17.npz | 2044 | 8a1ada9e5cea49c76d791465fe68de1813aab035c0dc9000bd9b73be59a91964 |
| matrix-heldout_ood-C-29.npz | 2044 | 52cb76b6d1dcbc7950833b091a4019ddd16e441b25959b7d54338dfa7b73d0bb |
| matrix-heldout_ood-C-43.npz | 2044 | 76cc034c2a52b07347c472ecc80295c84e30d731486249cd9cc450b841e5e576 |
| matrix-heldout_ood-D-17.npz | 2044 | 394e81b69f99a518aa43e727865c3e0a24cf798fae43fa3f35284b16824c5ed9 |
| matrix-heldout_ood-D-29.npz | 2044 | 8350514c49fd9f2d647cd1f3d14c7531b3affd9b5c4d0e762fac88465f8125d9 |
| matrix-heldout_ood-D-43.npz | 2044 | ba5f9b40275253f05e1a4e125764ce6694057534cd58f1908431bd8a36600995 |

## Task-stratified capability diagnostics

These strata are descriptive, use the same fixed acceptance rule, and do not select partitions or change promotion thresholds.

| Arm-seed | Split | Task | States | Coverage % [95% CI] |
| --- | --- | --- | --- | --- |
| A-17 | calibration | brainstorming | 688 | 63.953 [60.029, 67.882] |
| A-17 | calibration | classification | 685 | 57.664 [53.935, 61.338] |
| A-17 | calibration | closed_qa | 681 | 67.841 [64.274, 71.449] |
| A-17 | calibration | general_qa | 681 | 58.003 [54.171, 61.799] |
| A-17 | calibration | information_extraction | 664 | 69.578 [65.861, 73.083] |
| A-17 | calibration | open_qa | 680 | 62.941 [59.265, 66.471] |
| A-29 | calibration | brainstorming | 688 | 66.715 [63.081, 70.349] |
| A-29 | calibration | classification | 685 | 56.788 [53.353, 60.205] |
| A-29 | calibration | closed_qa | 681 | 70.631 [67.059, 74.232] |
| A-29 | calibration | general_qa | 681 | 60.793 [56.955, 64.517] |
| A-29 | calibration | information_extraction | 664 | 70.030 [66.467, 73.534] |
| A-29 | calibration | open_qa | 680 | 64.559 [61.324, 67.794] |
| A-43 | calibration | brainstorming | 688 | 58.576 [54.506, 62.791] |
| A-43 | calibration | classification | 685 | 53.869 [50.291, 57.496] |
| A-43 | calibration | closed_qa | 681 | 67.548 [63.716, 71.408] |
| A-43 | calibration | general_qa | 681 | 54.626 [50.514, 58.565] |
| A-43 | calibration | information_extraction | 664 | 69.127 [65.373, 72.782] |
| A-43 | calibration | open_qa | 680 | 60.294 [56.614, 63.971] |
| B-17 | calibration | brainstorming | 688 | 87.791 [85.029, 90.407] |
| B-17 | calibration | classification | 685 | 82.482 [79.738, 85.298] |
| B-17 | calibration | closed_qa | 681 | 88.693 [86.029, 91.163] |
| B-17 | calibration | general_qa | 681 | 84.435 [81.286, 87.391] |
| B-17 | calibration | information_extraction | 664 | 90.512 [88.341, 92.598] |
| B-17 | calibration | open_qa | 680 | 85.588 [82.794, 88.088] |
| B-29 | calibration | brainstorming | 688 | 86.192 [83.430, 88.953] |
| B-29 | calibration | classification | 685 | 81.752 [78.801, 84.448] |
| B-29 | calibration | closed_qa | 681 | 87.665 [84.852, 90.133] |
| B-29 | calibration | general_qa | 681 | 83.700 [80.381, 86.823] |
| B-29 | calibration | information_extraction | 664 | 89.458 [86.937, 91.554] |
| B-29 | calibration | open_qa | 680 | 83.971 [81.176, 86.765] |
| B-43 | calibration | brainstorming | 688 | 77.035 [73.692, 80.236] |
| B-43 | calibration | classification | 685 | 72.847 [69.677, 75.802] |
| B-43 | calibration | closed_qa | 681 | 81.204 [78.023, 84.311] |
| B-43 | calibration | general_qa | 681 | 73.715 [69.795, 77.321] |
| B-43 | calibration | information_extraction | 664 | 80.873 [77.910, 83.711] |
| B-43 | calibration | open_qa | 680 | 73.824 [70.441, 77.206] |
| C-17 | calibration | brainstorming | 688 | 86.628 [83.866, 89.244] |
| C-17 | calibration | classification | 685 | 84.964 [82.361, 87.500] |
| C-17 | calibration | closed_qa | 681 | 91.483 [89.069, 93.558] |
| C-17 | calibration | general_qa | 681 | 84.288 [81.378, 87.059] |
| C-17 | calibration | information_extraction | 664 | 89.910 [87.556, 92.169] |
| C-17 | calibration | open_qa | 680 | 83.529 [81.029, 86.029] |
| C-29 | calibration | brainstorming | 688 | 85.610 [82.849, 88.227] |
| C-29 | calibration | classification | 685 | 85.401 [82.748, 87.901] |
| C-29 | calibration | closed_qa | 681 | 90.602 [88.365, 92.920] |
| C-29 | calibration | general_qa | 681 | 83.113 [80.059, 85.882] |
| C-29 | calibration | information_extraction | 664 | 88.404 [85.735, 90.937] |
| C-29 | calibration | open_qa | 680 | 84.706 [81.912, 87.210] |
| C-43 | calibration | brainstorming | 688 | 78.343 [75.291, 81.250] |
| C-43 | calibration | classification | 685 | 80.876 [77.891, 83.919] |
| C-43 | calibration | closed_qa | 681 | 89.134 [86.618, 91.605] |
| C-43 | calibration | general_qa | 681 | 77.239 [73.777, 80.470] |
| C-43 | calibration | information_extraction | 664 | 87.952 [85.325, 90.405] |
| C-43 | calibration | open_qa | 680 | 78.676 [75.588, 81.618] |
| D-17 | calibration | brainstorming | 688 | 61.628 [57.558, 65.556] |
| D-17 | calibration | classification | 685 | 57.080 [53.723, 60.553] |
| D-17 | calibration | closed_qa | 681 | 72.981 [69.253, 76.506] |
| D-17 | calibration | general_qa | 681 | 58.884 [55.132, 62.592] |
| D-17 | calibration | information_extraction | 664 | 74.096 [70.359, 77.744] |
| D-17 | calibration | open_qa | 680 | 64.118 [60.588, 67.504] |
| D-29 | calibration | brainstorming | 688 | 63.517 [59.589, 67.442] |
| D-29 | calibration | classification | 685 | 59.416 [55.783, 62.936] |
| D-29 | calibration | closed_qa | 681 | 72.394 [68.879, 75.919] |
| D-29 | calibration | general_qa | 681 | 58.590 [54.611, 62.592] |
| D-29 | calibration | information_extraction | 664 | 72.289 [68.485, 75.977] |
| D-29 | calibration | open_qa | 680 | 61.618 [58.088, 65.151] |
| D-43 | calibration | brainstorming | 688 | 63.372 [59.302, 67.733] |
| D-43 | calibration | classification | 685 | 60.584 [57.226, 64.088] |
| D-43 | calibration | closed_qa | 681 | 72.247 [68.860, 75.772] |
| D-43 | calibration | general_qa | 681 | 60.206 [56.076, 64.107] |
| D-43 | calibration | information_extraction | 664 | 74.398 [70.588, 78.116] |
| D-43 | calibration | open_qa | 680 | 65.588 [61.765, 69.265] |
| A-17 | development | brainstorming | 180 | 66.667 [58.889, 74.444] |
| A-17 | development | classification | 180 | 58.333 [50.556, 66.111] |
| A-17 | development | closed_qa | 171 | 68.421 [60.234, 75.882] |
| A-17 | development | general_qa | 172 | 63.953 [56.395, 71.512] |
| A-17 | development | information_extraction | 147 | 68.707 [61.224, 75.676] |
| A-17 | development | open_qa | 168 | 64.881 [57.396, 72.291] |
| A-29 | development | brainstorming | 180 | 68.889 [62.222, 75.556] |
| A-29 | development | classification | 180 | 56.111 [48.889, 63.333] |
| A-29 | development | closed_qa | 171 | 71.345 [63.953, 78.488] |
| A-29 | development | general_qa | 172 | 64.535 [58.140, 70.930] |
| A-29 | development | information_extraction | 147 | 70.068 [62.162, 77.027] |
| A-29 | development | open_qa | 168 | 61.905 [55.294, 68.905] |
| A-43 | development | brainstorming | 180 | 62.222 [54.444, 70.556] |
| A-43 | development | classification | 180 | 53.333 [46.111, 61.111] |
| A-43 | development | closed_qa | 171 | 66.667 [59.412, 73.684] |
| A-43 | development | general_qa | 172 | 61.628 [54.070, 69.186] |
| A-43 | development | information_extraction | 147 | 69.388 [61.905, 77.401] |
| A-43 | development | open_qa | 168 | 57.738 [49.708, 65.663] |
| B-17 | development | brainstorming | 180 | 86.667 [81.111, 91.667] |
| B-17 | development | classification | 180 | 83.333 [77.222, 88.889] |
| B-17 | development | closed_qa | 171 | 90.643 [86.471, 94.737] |
| B-17 | development | general_qa | 172 | 89.535 [85.465, 93.605] |
| B-17 | development | information_extraction | 147 | 88.435 [82.313, 93.878] |
| B-17 | development | open_qa | 168 | 88.095 [82.941, 92.900] |
| B-29 | development | brainstorming | 180 | 86.667 [81.111, 91.667] |
| B-29 | development | classification | 180 | 79.444 [72.778, 85.556] |
| B-29 | development | closed_qa | 171 | 87.135 [81.871, 92.308] |
| B-29 | development | general_qa | 172 | 88.953 [84.884, 93.023] |
| B-29 | development | information_extraction | 147 | 85.714 [78.912, 91.892] |
| B-29 | development | open_qa | 168 | 83.929 [79.394, 88.347] |
| B-43 | development | brainstorming | 180 | 76.667 [68.889, 83.889] |
| B-43 | development | classification | 180 | 70.000 [62.778, 77.222] |
| B-43 | development | closed_qa | 171 | 81.287 [75.294, 87.209] |
| B-43 | development | general_qa | 172 | 80.814 [75.581, 86.628] |
| B-43 | development | information_extraction | 147 | 78.231 [71.233, 85.034] |
| B-43 | development | open_qa | 168 | 76.190 [70.122, 82.251] |
| C-17 | development | brainstorming | 180 | 88.333 [83.333, 92.778] |
| C-17 | development | classification | 180 | 82.778 [76.111, 88.889] |
| C-17 | development | closed_qa | 171 | 91.228 [87.133, 94.779] |
| C-17 | development | general_qa | 172 | 89.535 [84.884, 94.186] |
| C-17 | development | information_extraction | 147 | 89.796 [85.135, 93.878] |
| C-17 | development | open_qa | 168 | 83.929 [79.638, 88.690] |
| C-29 | development | brainstorming | 180 | 87.222 [82.222, 92.222] |
| C-29 | development | classification | 180 | 82.778 [76.111, 88.889] |
| C-29 | development | closed_qa | 171 | 90.058 [86.044, 94.152] |
| C-29 | development | general_qa | 172 | 87.209 [82.558, 91.860] |
| C-29 | development | information_extraction | 147 | 89.116 [84.354, 93.878] |
| C-29 | development | open_qa | 168 | 85.119 [79.042, 90.855] |
| C-43 | development | brainstorming | 180 | 83.889 [78.889, 88.889] |
| C-43 | development | classification | 180 | 78.889 [72.222, 85.000] |
| C-43 | development | closed_qa | 171 | 85.380 [79.651, 90.698] |
| C-43 | development | general_qa | 172 | 84.884 [79.651, 90.116] |
| C-43 | development | information_extraction | 147 | 87.075 [82.313, 91.838] |
| C-43 | development | open_qa | 168 | 79.167 [73.214, 84.942] |
| D-17 | development | brainstorming | 180 | 65.000 [58.333, 71.667] |
| D-17 | development | classification | 180 | 58.333 [50.000, 66.667] |
| D-17 | development | closed_qa | 171 | 74.269 [67.442, 80.702] |
| D-17 | development | general_qa | 172 | 72.093 [64.535, 79.070] |
| D-17 | development | information_extraction | 147 | 71.429 [63.946, 78.912] |
| D-17 | development | open_qa | 168 | 62.500 [56.140, 69.048] |
| D-29 | development | brainstorming | 180 | 64.444 [56.667, 72.778] |
| D-29 | development | classification | 180 | 59.444 [51.667, 67.778] |
| D-29 | development | closed_qa | 171 | 72.515 [65.497, 79.070] |
| D-29 | development | general_qa | 172 | 65.698 [58.140, 73.256] |
| D-29 | development | information_extraction | 147 | 71.429 [63.514, 79.452] |
| D-29 | development | open_qa | 168 | 62.500 [55.758, 69.412] |
| D-43 | development | brainstorming | 180 | 67.222 [59.444, 75.556] |
| D-43 | development | classification | 180 | 60.000 [53.333, 67.222] |
| D-43 | development | closed_qa | 171 | 71.930 [64.118, 78.947] |
| D-43 | development | general_qa | 172 | 65.116 [58.721, 71.512] |
| D-43 | development | information_extraction | 147 | 70.748 [62.585, 78.235] |
| D-43 | development | open_qa | 168 | 59.524 [51.786, 67.059] |
| A-17 | heldout_matched | brainstorming | 352 | 63.920 [58.523, 69.318] |
| A-17 | heldout_matched | classification | 349 | 55.301 [50.000, 60.522] |
| A-17 | heldout_matched | closed_qa | 351 | 70.085 [64.857, 75.215] |
| A-17 | heldout_matched | general_qa | 350 | 61.714 [56.250, 67.143] |
| A-17 | heldout_matched | information_extraction | 295 | 66.780 [61.017, 71.864] |
| A-17 | heldout_matched | open_qa | 340 | 61.176 [55.193, 66.959] |
| A-29 | heldout_matched | brainstorming | 352 | 64.773 [59.659, 69.886] |
| A-29 | heldout_matched | classification | 349 | 55.301 [50.143, 60.466] |
| A-29 | heldout_matched | closed_qa | 351 | 68.091 [62.784, 73.219] |
| A-29 | heldout_matched | general_qa | 350 | 61.429 [56.160, 66.669] |
| A-29 | heldout_matched | information_extraction | 295 | 66.102 [60.675, 71.186] |
| A-29 | heldout_matched | open_qa | 340 | 64.118 [58.160, 69.706] |
| A-43 | heldout_matched | brainstorming | 352 | 61.932 [56.250, 67.614] |
| A-43 | heldout_matched | classification | 349 | 54.441 [48.997, 59.714] |
| A-43 | heldout_matched | closed_qa | 351 | 66.952 [61.648, 72.159] |
| A-43 | heldout_matched | general_qa | 350 | 54.857 [49.143, 60.458] |
| A-43 | heldout_matched | information_extraction | 295 | 67.119 [61.563, 72.542] |
| A-43 | heldout_matched | open_qa | 340 | 59.412 [53.642, 65.192] |
| B-17 | heldout_matched | brainstorming | 352 | 88.352 [85.227, 91.477] |
| B-17 | heldout_matched | classification | 349 | 81.948 [77.585, 86.168] |
| B-17 | heldout_matched | closed_qa | 351 | 88.604 [85.143, 92.045] |
| B-17 | heldout_matched | general_qa | 350 | 84.286 [80.114, 88.000] |
| B-17 | heldout_matched | information_extraction | 295 | 89.492 [85.424, 93.220] |
| B-17 | heldout_matched | open_qa | 340 | 85.294 [81.360, 88.856] |
| B-29 | heldout_matched | brainstorming | 352 | 86.080 [81.818, 90.057] |
| B-29 | heldout_matched | classification | 349 | 75.358 [70.571, 80.058] |
| B-29 | heldout_matched | closed_qa | 351 | 89.174 [85.714, 92.593] |
| B-29 | heldout_matched | general_qa | 350 | 81.714 [77.429, 85.755] |
| B-29 | heldout_matched | information_extraction | 295 | 87.119 [83.049, 90.816] |
| B-29 | heldout_matched | open_qa | 340 | 81.765 [77.286, 85.883] |
| B-43 | heldout_matched | brainstorming | 352 | 78.693 [74.148, 82.955] |
| B-43 | heldout_matched | classification | 349 | 68.195 [62.571, 73.354] |
| B-43 | heldout_matched | closed_qa | 351 | 82.621 [78.063, 86.937] |
| B-43 | heldout_matched | general_qa | 350 | 73.429 [68.195, 78.324] |
| B-43 | heldout_matched | information_extraction | 295 | 77.627 [72.540, 82.373] |
| B-43 | heldout_matched | open_qa | 340 | 74.118 [68.639, 79.351] |
| C-17 | heldout_matched | brainstorming | 352 | 86.648 [82.670, 90.341] |
| C-17 | heldout_matched | classification | 349 | 79.656 [75.000, 83.714] |
| C-17 | heldout_matched | closed_qa | 351 | 89.459 [86.039, 92.857] |
| C-17 | heldout_matched | general_qa | 350 | 82.000 [77.557, 86.287] |
| C-17 | heldout_matched | information_extraction | 295 | 88.136 [84.122, 91.864] |
| C-17 | heldout_matched | open_qa | 340 | 81.471 [76.627, 85.924] |
| C-29 | heldout_matched | brainstorming | 352 | 85.511 [81.534, 89.205] |
| C-29 | heldout_matched | classification | 349 | 79.656 [75.287, 83.575] |
| C-29 | heldout_matched | closed_qa | 351 | 87.464 [83.476, 91.168] |
| C-29 | heldout_matched | general_qa | 350 | 80.571 [76.571, 84.528] |
| C-29 | heldout_matched | information_extraction | 295 | 86.102 [81.967, 89.869] |
| C-29 | heldout_matched | open_qa | 340 | 85.000 [80.644, 89.150] |
| C-43 | heldout_matched | brainstorming | 352 | 80.398 [75.852, 84.943] |
| C-43 | heldout_matched | classification | 349 | 76.791 [72.334, 80.981] |
| C-43 | heldout_matched | closed_qa | 351 | 86.895 [82.905, 90.909] |
| C-43 | heldout_matched | general_qa | 350 | 76.286 [71.223, 81.250] |
| C-43 | heldout_matched | information_extraction | 295 | 84.407 [79.322, 88.814] |
| C-43 | heldout_matched | open_qa | 340 | 78.235 [73.373, 82.891] |
| D-17 | heldout_matched | brainstorming | 352 | 64.489 [58.807, 69.886] |
| D-17 | heldout_matched | classification | 349 | 54.441 [49.138, 59.600] |
| D-17 | heldout_matched | closed_qa | 351 | 68.946 [63.815, 74.212] |
| D-17 | heldout_matched | general_qa | 350 | 60.286 [54.286, 65.714] |
| D-17 | heldout_matched | information_extraction | 295 | 67.458 [62.034, 72.203] |
| D-17 | heldout_matched | open_qa | 340 | 62.059 [55.424, 68.328] |
| D-29 | heldout_matched | brainstorming | 352 | 65.909 [60.227, 71.591] |
| D-29 | heldout_matched | classification | 349 | 57.020 [51.714, 62.249] |
| D-29 | heldout_matched | closed_qa | 351 | 68.946 [63.530, 74.148] |
| D-29 | heldout_matched | general_qa | 350 | 58.857 [53.140, 64.470] |
| D-29 | heldout_matched | information_extraction | 295 | 69.492 [63.851, 74.576] |
| D-29 | heldout_matched | open_qa | 340 | 59.706 [53.687, 65.689] |
| D-43 | heldout_matched | brainstorming | 352 | 66.477 [60.511, 72.443] |
| D-43 | heldout_matched | classification | 349 | 59.312 [54.000, 64.656] |
| D-43 | heldout_matched | closed_qa | 351 | 71.510 [66.761, 76.420] |
| D-43 | heldout_matched | general_qa | 350 | 60.000 [54.000, 65.616] |
| D-43 | heldout_matched | information_extraction | 295 | 74.237 [69.153, 78.983] |
| D-43 | heldout_matched | open_qa | 340 | 65.294 [59.531, 70.921] |
| A-17 | heldout_ood | creative_writing | 1047 | 51.003 [47.373, 54.537] |
| A-17 | heldout_ood | summarization | 997 | 66.800 [63.827, 69.710] |
| A-29 | heldout_ood | creative_writing | 1047 | 56.638 [53.339, 59.733] |
| A-29 | heldout_ood | summarization | 997 | 66.399 [63.353, 69.277] |
| A-43 | heldout_ood | creative_writing | 1047 | 50.716 [47.042, 54.198] |
| A-43 | heldout_ood | summarization | 997 | 62.287 [59.159, 65.331] |
| B-17 | heldout_ood | creative_writing | 1047 | 81.471 [78.776, 84.160] |
| B-17 | heldout_ood | summarization | 997 | 87.061 [85.070, 89.135] |
| B-29 | heldout_ood | creative_writing | 1047 | 81.853 [79.198, 84.448] |
| B-29 | heldout_ood | summarization | 997 | 85.055 [82.730, 87.401] |
| B-43 | heldout_ood | creative_writing | 1047 | 69.054 [65.997, 72.016] |
| B-43 | heldout_ood | summarization | 997 | 76.530 [73.673, 79.397] |
| C-17 | heldout_ood | creative_writing | 1047 | 81.280 [78.701, 83.748] |
| C-17 | heldout_ood | summarization | 997 | 87.262 [85.256, 89.315] |
| C-29 | heldout_ood | creative_writing | 1047 | 80.802 [78.319, 83.270] |
| C-29 | heldout_ood | summarization | 997 | 87.663 [85.757, 89.569] |
| C-43 | heldout_ood | creative_writing | 1047 | 73.830 [70.909, 76.673] |
| C-43 | heldout_ood | summarization | 997 | 83.651 [81.263, 85.986] |
| D-17 | heldout_ood | creative_writing | 1047 | 55.110 [51.718, 58.453] |
| D-17 | heldout_ood | summarization | 997 | 69.007 [66.000, 71.945] |
| D-29 | heldout_ood | creative_writing | 1047 | 52.245 [48.565, 55.821] |
| D-29 | heldout_ood | summarization | 997 | 66.800 [63.927, 69.688] |
| D-43 | heldout_ood | creative_writing | 1047 | 55.778 [52.195, 59.065] |
| D-43 | heldout_ood | summarization | 997 | 69.809 [67.034, 72.774] |

## Dense-margin diagnostics

| Arm-seed | Split | Dense top-two margin | States | Coverage % | Fixed coverage % |
| --- | --- | --- | --- | --- | --- |
| A-17 | calibration | [0.0, 0.1) | 66 | 12.12 | 0.00 |
| A-17 | calibration | [0.1, 0.5) | 375 | 18.67 | 5.07 |
| A-17 | calibration | [0.5, 1.0) | 456 | 26.32 | 7.68 |
| A-17 | calibration | [1.0, 2.0) | 697 | 37.30 | 12.05 |
| A-17 | calibration | [2.0, inf) | 2485 | 85.47 | 61.85 |
| A-29 | calibration | [0.0, 0.1) | 66 | 15.15 | 3.03 |
| A-29 | calibration | [0.1, 0.5) | 375 | 20.00 | 7.47 |
| A-29 | calibration | [0.5, 1.0) | 456 | 28.07 | 11.18 |
| A-29 | calibration | [1.0, 2.0) | 697 | 37.02 | 16.21 |
| A-29 | calibration | [2.0, inf) | 2485 | 87.57 | 64.19 |
| A-43 | calibration | [0.0, 0.1) | 66 | 13.64 | 0.00 |
| A-43 | calibration | [0.1, 0.5) | 375 | 18.40 | 2.93 |
| A-43 | calibration | [0.5, 1.0) | 456 | 20.61 | 5.48 |
| A-43 | calibration | [1.0, 2.0) | 697 | 30.42 | 6.03 |
| A-43 | calibration | [2.0, inf) | 2485 | 84.06 | 56.14 |
| B-17 | calibration | [0.0, 0.1) | 66 | 48.48 | 18.18 |
| B-17 | calibration | [0.1, 0.5) | 375 | 59.20 | 23.73 |
| B-17 | calibration | [0.5, 1.0) | 456 | 67.98 | 33.55 |
| B-17 | calibration | [1.0, 2.0) | 697 | 77.91 | 44.05 |
| B-17 | calibration | [2.0, inf) | 2485 | 97.55 | 81.37 |
| B-29 | calibration | [0.0, 0.1) | 66 | 46.97 | 18.18 |
| B-29 | calibration | [0.1, 0.5) | 375 | 57.33 | 28.53 |
| B-29 | calibration | [0.5, 1.0) | 456 | 65.57 | 33.11 |
| B-29 | calibration | [1.0, 2.0) | 697 | 74.89 | 38.02 |
| B-29 | calibration | [2.0, inf) | 2485 | 97.30 | 80.12 |
| B-43 | calibration | [0.0, 0.1) | 66 | 25.76 | 7.58 |
| B-43 | calibration | [0.1, 0.5) | 375 | 41.07 | 15.20 |
| B-43 | calibration | [0.5, 1.0) | 456 | 46.71 | 17.32 |
| B-43 | calibration | [1.0, 2.0) | 697 | 58.11 | 25.25 |
| B-43 | calibration | [2.0, inf) | 2485 | 93.92 | 72.64 |
| C-17 | calibration | [0.0, 0.1) | 66 | 57.58 | 18.18 |
| C-17 | calibration | [0.1, 0.5) | 375 | 60.53 | 21.07 |
| C-17 | calibration | [0.5, 1.0) | 456 | 66.89 | 26.10 |
| C-17 | calibration | [1.0, 2.0) | 697 | 76.47 | 33.14 |
| C-17 | calibration | [2.0, inf) | 2485 | 98.07 | 79.07 |
| C-29 | calibration | [0.0, 0.1) | 66 | 59.09 | 18.18 |
| C-29 | calibration | [0.1, 0.5) | 375 | 56.00 | 19.47 |
| C-29 | calibration | [0.5, 1.0) | 456 | 66.89 | 21.49 |
| C-29 | calibration | [1.0, 2.0) | 697 | 77.04 | 30.27 |
| C-29 | calibration | [2.0, inf) | 2485 | 97.75 | 77.59 |
| C-43 | calibration | [0.0, 0.1) | 66 | 54.55 | 19.70 |
| C-43 | calibration | [0.1, 0.5) | 375 | 49.87 | 16.00 |
| C-43 | calibration | [0.5, 1.0) | 456 | 58.11 | 21.71 |
| C-43 | calibration | [1.0, 2.0) | 697 | 68.87 | 29.84 |
| C-43 | calibration | [2.0, inf) | 2485 | 95.65 | 76.10 |
| D-17 | calibration | [0.0, 0.1) | 66 | 21.21 | 7.58 |
| D-17 | calibration | [0.1, 0.5) | 375 | 21.60 | 7.47 |
| D-17 | calibration | [0.5, 1.0) | 456 | 25.00 | 7.68 |
| D-17 | calibration | [1.0, 2.0) | 697 | 36.87 | 13.63 |
| D-17 | calibration | [2.0, inf) | 2485 | 87.53 | 63.62 |
| D-29 | calibration | [0.0, 0.1) | 66 | 18.18 | 1.52 |
| D-29 | calibration | [0.1, 0.5) | 375 | 20.80 | 4.80 |
| D-29 | calibration | [0.5, 1.0) | 456 | 26.10 | 7.68 |
| D-29 | calibration | [1.0, 2.0) | 697 | 37.16 | 10.33 |
| D-29 | calibration | [2.0, inf) | 2485 | 87.20 | 60.00 |
| D-43 | calibration | [0.0, 0.1) | 66 | 16.67 | 4.55 |
| D-43 | calibration | [0.1, 0.5) | 375 | 25.33 | 4.00 |
| D-43 | calibration | [0.5, 1.0) | 456 | 27.41 | 7.89 |
| D-43 | calibration | [1.0, 2.0) | 697 | 39.60 | 10.19 |
| D-43 | calibration | [2.0, inf) | 2485 | 87.97 | 63.14 |
| A-17 | development | [0.0, 0.1) | 20 | 40.00 | 20.00 |
| A-17 | development | [0.1, 0.5) | 108 | 19.44 | 5.56 |
| A-17 | development | [0.5, 1.0) | 99 | 31.31 | 14.14 |
| A-17 | development | [1.0, 2.0) | 164 | 36.59 | 10.98 |
| A-17 | development | [2.0, inf) | 627 | 86.44 | 61.56 |
| A-29 | development | [0.0, 0.1) | 20 | 30.00 | 5.00 |
| A-29 | development | [0.1, 0.5) | 108 | 19.44 | 4.63 |
| A-29 | development | [0.5, 1.0) | 99 | 25.25 | 15.15 |
| A-29 | development | [1.0, 2.0) | 164 | 39.02 | 17.68 |
| A-29 | development | [2.0, inf) | 627 | 87.56 | 63.80 |
| A-43 | development | [0.0, 0.1) | 20 | 30.00 | 5.00 |
| A-43 | development | [0.1, 0.5) | 108 | 16.67 | 2.78 |
| A-43 | development | [0.5, 1.0) | 99 | 22.22 | 5.05 |
| A-43 | development | [1.0, 2.0) | 164 | 28.05 | 6.10 |
| A-43 | development | [2.0, inf) | 627 | 85.33 | 58.05 |
| B-17 | development | [0.0, 0.1) | 20 | 55.00 | 30.00 |
| B-17 | development | [0.1, 0.5) | 108 | 65.74 | 32.41 |
| B-17 | development | [0.5, 1.0) | 99 | 70.71 | 32.32 |
| B-17 | development | [1.0, 2.0) | 164 | 78.05 | 43.29 |
| B-17 | development | [2.0, inf) | 627 | 97.77 | 80.86 |
| B-29 | development | [0.0, 0.1) | 20 | 40.00 | 15.00 |
| B-29 | development | [0.1, 0.5) | 108 | 61.11 | 29.63 |
| B-29 | development | [0.5, 1.0) | 99 | 68.69 | 30.30 |
| B-29 | development | [1.0, 2.0) | 164 | 73.78 | 40.85 |
| B-29 | development | [2.0, inf) | 627 | 96.49 | 78.15 |
| B-43 | development | [0.0, 0.1) | 20 | 40.00 | 15.00 |
| B-43 | development | [0.1, 0.5) | 108 | 39.81 | 17.59 |
| B-43 | development | [0.5, 1.0) | 99 | 46.46 | 17.17 |
| B-43 | development | [1.0, 2.0) | 164 | 57.32 | 25.61 |
| B-43 | development | [2.0, inf) | 627 | 94.74 | 71.29 |
| C-17 | development | [0.0, 0.1) | 20 | 60.00 | 35.00 |
| C-17 | development | [0.1, 0.5) | 108 | 63.89 | 20.37 |
| C-17 | development | [0.5, 1.0) | 99 | 76.77 | 28.28 |
| C-17 | development | [1.0, 2.0) | 164 | 74.39 | 34.15 |
| C-17 | development | [2.0, inf) | 627 | 97.61 | 77.67 |
| C-29 | development | [0.0, 0.1) | 20 | 55.00 | 15.00 |
| C-29 | development | [0.1, 0.5) | 108 | 59.26 | 18.52 |
| C-29 | development | [0.5, 1.0) | 99 | 69.70 | 19.19 |
| C-29 | development | [1.0, 2.0) | 164 | 78.66 | 29.88 |
| C-29 | development | [2.0, inf) | 627 | 97.45 | 78.63 |
| C-43 | development | [0.0, 0.1) | 20 | 50.00 | 30.00 |
| C-43 | development | [0.1, 0.5) | 108 | 57.41 | 25.93 |
| C-43 | development | [0.5, 1.0) | 99 | 61.62 | 20.20 |
| C-43 | development | [1.0, 2.0) | 164 | 70.12 | 35.37 |
| C-43 | development | [2.0, inf) | 627 | 95.37 | 79.43 |
| D-17 | development | [0.0, 0.1) | 20 | 35.00 | 15.00 |
| D-17 | development | [0.1, 0.5) | 108 | 30.56 | 11.11 |
| D-17 | development | [0.5, 1.0) | 99 | 28.28 | 5.05 |
| D-17 | development | [1.0, 2.0) | 164 | 38.41 | 11.59 |
| D-17 | development | [2.0, inf) | 627 | 88.04 | 65.55 |
| D-29 | development | [0.0, 0.1) | 20 | 35.00 | 0.00 |
| D-29 | development | [0.1, 0.5) | 108 | 25.00 | 8.33 |
| D-29 | development | [0.5, 1.0) | 99 | 26.26 | 6.06 |
| D-29 | development | [1.0, 2.0) | 164 | 37.80 | 12.20 |
| D-29 | development | [2.0, inf) | 627 | 87.40 | 59.01 |
| D-43 | development | [0.0, 0.1) | 20 | 20.00 | 0.00 |
| D-43 | development | [0.1, 0.5) | 108 | 23.15 | 7.41 |
| D-43 | development | [0.5, 1.0) | 99 | 24.24 | 4.04 |
| D-43 | development | [1.0, 2.0) | 164 | 40.85 | 12.20 |
| D-43 | development | [2.0, inf) | 627 | 87.40 | 62.84 |
| A-17 | heldout_matched | [0.0, 0.1) | 24 | 12.50 | 4.17 |
| A-17 | heldout_matched | [0.1, 0.5) | 189 | 19.58 | 5.82 |
| A-17 | heldout_matched | [0.5, 1.0) | 234 | 28.63 | 8.12 |
| A-17 | heldout_matched | [1.0, 2.0) | 383 | 36.29 | 10.97 |
| A-17 | heldout_matched | [2.0, inf) | 1207 | 86.08 | 63.30 |
| A-29 | heldout_matched | [0.0, 0.1) | 24 | 4.17 | 0.00 |
| A-29 | heldout_matched | [0.1, 0.5) | 189 | 19.05 | 8.47 |
| A-29 | heldout_matched | [0.5, 1.0) | 234 | 28.63 | 9.83 |
| A-29 | heldout_matched | [1.0, 2.0) | 383 | 33.68 | 14.88 |
| A-29 | heldout_matched | [2.0, inf) | 1207 | 87.41 | 62.88 |
| A-43 | heldout_matched | [0.0, 0.1) | 24 | 8.33 | 0.00 |
| A-43 | heldout_matched | [0.1, 0.5) | 189 | 17.99 | 2.65 |
| A-43 | heldout_matched | [0.5, 1.0) | 234 | 26.50 | 5.98 |
| A-43 | heldout_matched | [1.0, 2.0) | 383 | 31.33 | 7.05 |
| A-43 | heldout_matched | [2.0, inf) | 1207 | 84.26 | 54.93 |
| B-17 | heldout_matched | [0.0, 0.1) | 24 | 33.33 | 8.33 |
| B-17 | heldout_matched | [0.1, 0.5) | 189 | 60.32 | 27.51 |
| B-17 | heldout_matched | [0.5, 1.0) | 234 | 69.23 | 32.05 |
| B-17 | heldout_matched | [1.0, 2.0) | 383 | 77.55 | 44.65 |
| B-17 | heldout_matched | [2.0, inf) | 1207 | 97.43 | 80.28 |
| B-29 | heldout_matched | [0.0, 0.1) | 24 | 29.17 | 4.17 |
| B-29 | heldout_matched | [0.1, 0.5) | 189 | 56.08 | 26.46 |
| B-29 | heldout_matched | [0.5, 1.0) | 234 | 65.81 | 36.75 |
| B-29 | heldout_matched | [1.0, 2.0) | 383 | 69.71 | 37.86 |
| B-29 | heldout_matched | [2.0, inf) | 1207 | 96.60 | 79.29 |
| B-43 | heldout_matched | [0.0, 0.1) | 24 | 12.50 | 4.17 |
| B-43 | heldout_matched | [0.1, 0.5) | 189 | 36.51 | 13.23 |
| B-43 | heldout_matched | [0.5, 1.0) | 234 | 47.01 | 17.52 |
| B-43 | heldout_matched | [1.0, 2.0) | 383 | 58.75 | 23.24 |
| B-43 | heldout_matched | [2.0, inf) | 1207 | 94.12 | 71.75 |
| C-17 | heldout_matched | [0.0, 0.1) | 24 | 20.83 | 8.33 |
| C-17 | heldout_matched | [0.1, 0.5) | 189 | 58.20 | 20.11 |
| C-17 | heldout_matched | [0.5, 1.0) | 234 | 63.68 | 21.79 |
| C-17 | heldout_matched | [1.0, 2.0) | 383 | 73.37 | 31.85 |
| C-17 | heldout_matched | [2.0, inf) | 1207 | 97.43 | 76.88 |
| C-29 | heldout_matched | [0.0, 0.1) | 24 | 33.33 | 4.17 |
| C-29 | heldout_matched | [0.1, 0.5) | 189 | 52.91 | 20.11 |
| C-29 | heldout_matched | [0.5, 1.0) | 234 | 67.52 | 23.50 |
| C-29 | heldout_matched | [1.0, 2.0) | 383 | 72.58 | 29.77 |
| C-29 | heldout_matched | [2.0, inf) | 1207 | 96.69 | 78.29 |
| C-43 | heldout_matched | [0.0, 0.1) | 24 | 16.67 | 4.17 |
| C-43 | heldout_matched | [0.1, 0.5) | 189 | 50.79 | 20.63 |
| C-43 | heldout_matched | [0.5, 1.0) | 234 | 55.56 | 23.93 |
| C-43 | heldout_matched | [1.0, 2.0) | 383 | 67.10 | 31.85 |
| C-43 | heldout_matched | [2.0, inf) | 1207 | 95.36 | 75.06 |
| D-17 | heldout_matched | [0.0, 0.1) | 24 | 0.00 | 0.00 |
| D-17 | heldout_matched | [0.1, 0.5) | 189 | 19.05 | 6.35 |
| D-17 | heldout_matched | [0.5, 1.0) | 234 | 23.93 | 9.83 |
| D-17 | heldout_matched | [1.0, 2.0) | 383 | 36.81 | 10.70 |
| D-17 | heldout_matched | [2.0, inf) | 1207 | 86.74 | 64.87 |
| D-29 | heldout_matched | [0.0, 0.1) | 24 | 0.00 | 0.00 |
| D-29 | heldout_matched | [0.1, 0.5) | 189 | 19.58 | 4.23 |
| D-29 | heldout_matched | [0.5, 1.0) | 234 | 26.07 | 7.26 |
| D-29 | heldout_matched | [1.0, 2.0) | 383 | 37.86 | 12.01 |
| D-29 | heldout_matched | [2.0, inf) | 1207 | 86.50 | 60.23 |
| D-43 | heldout_matched | [0.0, 0.1) | 24 | 12.50 | 4.17 |
| D-43 | heldout_matched | [0.1, 0.5) | 189 | 21.69 | 6.35 |
| D-43 | heldout_matched | [0.5, 1.0) | 234 | 26.50 | 6.84 |
| D-43 | heldout_matched | [1.0, 2.0) | 383 | 44.13 | 8.62 |
| D-43 | heldout_matched | [2.0, inf) | 1207 | 88.48 | 63.13 |
| A-17 | heldout_ood | [0.0, 0.1) | 45 | 13.33 | 0.00 |
| A-17 | heldout_ood | [0.1, 0.5) | 179 | 16.76 | 5.59 |
| A-17 | heldout_ood | [0.5, 1.0) | 236 | 19.49 | 3.39 |
| A-17 | heldout_ood | [1.0, 2.0) | 367 | 30.52 | 5.45 |
| A-17 | heldout_ood | [2.0, inf) | 1217 | 82.66 | 56.78 |
| A-29 | heldout_ood | [0.0, 0.1) | 45 | 20.00 | 4.44 |
| A-29 | heldout_ood | [0.1, 0.5) | 179 | 17.88 | 5.03 |
| A-29 | heldout_ood | [0.5, 1.0) | 236 | 20.76 | 9.32 |
| A-29 | heldout_ood | [1.0, 2.0) | 367 | 34.33 | 16.62 |
| A-29 | heldout_ood | [2.0, inf) | 1217 | 85.37 | 62.04 |
| A-43 | heldout_ood | [0.0, 0.1) | 45 | 4.44 | 0.00 |
| A-43 | heldout_ood | [0.1, 0.5) | 179 | 16.20 | 2.23 |
| A-43 | heldout_ood | [0.5, 1.0) | 236 | 15.25 | 2.97 |
| A-43 | heldout_ood | [1.0, 2.0) | 367 | 25.89 | 4.63 |
| A-43 | heldout_ood | [2.0, inf) | 1217 | 81.35 | 54.64 |
| B-17 | heldout_ood | [0.0, 0.1) | 45 | 48.89 | 17.78 |
| B-17 | heldout_ood | [0.1, 0.5) | 179 | 52.51 | 18.44 |
| B-17 | heldout_ood | [0.5, 1.0) | 236 | 64.83 | 27.97 |
| B-17 | heldout_ood | [1.0, 2.0) | 367 | 73.02 | 40.87 |
| B-17 | heldout_ood | [2.0, inf) | 1217 | 97.29 | 79.46 |
| B-29 | heldout_ood | [0.0, 0.1) | 45 | 40.00 | 11.11 |
| B-29 | heldout_ood | [0.1, 0.5) | 179 | 48.60 | 20.11 |
| B-29 | heldout_ood | [0.5, 1.0) | 236 | 63.14 | 32.20 |
| B-29 | heldout_ood | [1.0, 2.0) | 367 | 73.84 | 39.51 |
| B-29 | heldout_ood | [2.0, inf) | 1217 | 96.96 | 76.75 |
| B-43 | heldout_ood | [0.0, 0.1) | 45 | 24.44 | 6.67 |
| B-43 | heldout_ood | [0.1, 0.5) | 179 | 34.08 | 12.29 |
| B-43 | heldout_ood | [0.5, 1.0) | 236 | 38.56 | 19.49 |
| B-43 | heldout_ood | [1.0, 2.0) | 367 | 52.86 | 23.71 |
| B-43 | heldout_ood | [2.0, inf) | 1217 | 92.77 | 72.14 |
| C-17 | heldout_ood | [0.0, 0.1) | 45 | 33.33 | 11.11 |
| C-17 | heldout_ood | [0.1, 0.5) | 179 | 49.72 | 19.55 |
| C-17 | heldout_ood | [0.5, 1.0) | 236 | 61.44 | 25.42 |
| C-17 | heldout_ood | [1.0, 2.0) | 367 | 76.29 | 34.60 |
| C-17 | heldout_ood | [2.0, inf) | 1217 | 97.95 | 78.80 |
| C-29 | heldout_ood | [0.0, 0.1) | 45 | 44.44 | 17.78 |
| C-29 | heldout_ood | [0.1, 0.5) | 179 | 48.04 | 15.08 |
| C-29 | heldout_ood | [0.5, 1.0) | 236 | 66.10 | 19.92 |
| C-29 | heldout_ood | [1.0, 2.0) | 367 | 74.11 | 28.61 |
| C-29 | heldout_ood | [2.0, inf) | 1217 | 97.45 | 77.49 |
| C-43 | heldout_ood | [0.0, 0.1) | 45 | 31.11 | 8.89 |
| C-43 | heldout_ood | [0.1, 0.5) | 179 | 40.22 | 15.08 |
| C-43 | heldout_ood | [0.5, 1.0) | 236 | 54.24 | 22.46 |
| C-43 | heldout_ood | [1.0, 2.0) | 367 | 64.31 | 26.16 |
| C-43 | heldout_ood | [2.0, inf) | 1217 | 95.07 | 74.03 |
| D-17 | heldout_ood | [0.0, 0.1) | 45 | 13.33 | 4.44 |
| D-17 | heldout_ood | [0.1, 0.5) | 179 | 18.44 | 6.15 |
| D-17 | heldout_ood | [0.5, 1.0) | 236 | 23.31 | 8.05 |
| D-17 | heldout_ood | [1.0, 2.0) | 367 | 36.78 | 16.89 |
| D-17 | heldout_ood | [2.0, inf) | 1217 | 85.13 | 60.39 |
| D-29 | heldout_ood | [0.0, 0.1) | 45 | 8.89 | 0.00 |
| D-29 | heldout_ood | [0.1, 0.5) | 179 | 15.64 | 5.59 |
| D-29 | heldout_ood | [0.5, 1.0) | 236 | 16.95 | 3.39 |
| D-29 | heldout_ood | [1.0, 2.0) | 367 | 34.06 | 6.54 |
| D-29 | heldout_ood | [2.0, inf) | 1217 | 83.48 | 58.92 |
| D-43 | heldout_ood | [0.0, 0.1) | 45 | 20.00 | 2.22 |
| D-43 | heldout_ood | [0.1, 0.5) | 179 | 20.11 | 5.03 |
| D-43 | heldout_ood | [0.5, 1.0) | 236 | 22.88 | 8.05 |
| D-43 | heldout_ood | [1.0, 2.0) | 367 | 34.06 | 10.63 |
| D-43 | heldout_ood | [2.0, inf) | 1217 | 86.77 | 60.89 |
