# Frozen C-29 causal expert recognition

Run: `artifacts/expert_recognition/20261014-recognition-v1`. Preregistered UTC: `2026-10-15T04:41:00.925796+00:00`.

## Outcome

**Independent capability confirmation failed.** The cycle stopped before selector fitting,
development selection, fresh held-out evaluation, runtime quality or timing. No alternative
partition was substituted. Recognition itself was **not tested**; this is not a router failure
or a finding that every partition is unrecognizable.

This is a new study of one discovery-selected partition. The preceding construction study's
all-three-construction-seed failure and skipped later stages remain unchanged. C-29 and fixed
candidate 8 (`[0,1,2,4,6,7]`) were selected using old development evidence, never these new outcomes.

## Frozen scope and data

Model: `Qwen/Qwen2.5-0.5B-Instruct` at `7ae557604adf67be50417f59c2c2f167def9a775`; bfloat16, sdpa.
Original model weights frozen; only the last MLP is sparsified. Eight 608-neuron experts,
28 six-of-eight choices, 3,648/4,864 active neurons for each cheap call; no residual correction.

Manifest source: pinned human-written Dolly. All 6,163 registered historical prompts and all 2,304
prior expert-study contexts are excluded; exact source-document and instruction-family IDs
and lexical similarity screening separate splits. Source responses are omitted. The 12.5%
enterprise stratum is a keyword-topic screen, not a production-workload benchmark. Category
holdout and lexical independence do not establish broad semantic or article independence.

Measured environment: NVIDIA GeForce RTX 4070 SUPER; PyTorch 2.7.1+cu128; CUDA 12.8; Transformers 4.57.1.

| Split | Frozen contexts |
| --- | --- |
| calibration | 768 |
| confirmation | 128 |
| development | 128 |
| heldout_matched | 512 |
| heldout_ood | 192 |

Confirmation is 64 matched + 64 OOD contexts and never trains/selects a router. The 1,728-context
recipe was reduced prospectively from 2,816 because only 41 fresh OOD enterprise rows remained
under the unchanged source/exclusion/quota constraints. OOD enterprise examples are creative-writing-heavy.

Sampled actual continuation positions: `[1, 3, 7, 11]`, dense prefixes,
batch 16, stop recording after EOS; max prompt length 512 tokens.
All intervals below are 95% whole-context bootstrap intervals (2,000 draws), not independent-token
intervals.

| Actually captured | Contexts | States | Zero-state contexts | Truncated prompts |
| --- | --- | --- | --- | --- |
| confirmation | 128 | 511 | 0 | 0 |

## Capability confirmation

Each stratum must achieve >=80% acceptable-candidate coverage and >=5 percentage-point ideal
active-width advantage over the frozen fixed-plus-dense oracle. Acceptable means exact
greedy agreement and KL(dense||candidate)<0.01. Point engineering gates and statistically
resolved passes are reported separately.

| Stratum | Contexts | States | Portfolio coverage [CI] | Fixed coverage [CI] | Ideal width advantage [CI] | Point | Resolved pass |
| --- | --- | --- | --- | --- | --- | --- | --- |
| matched | 64 | 255 | 88.235% [83.465, 92.195] | 56.863% [51.373, 62.745] | 7.843 pp [6.667, 9.020] | pass | pass |
| ood | 64 | 256 | 75.781% [70.693, 81.250] | 49.219% [43.750, 54.688] | 6.641 pp [5.371, 7.910] | fail | fail |

The ood coverage interval crosses the 80% threshold. The observed sample misses
the preregistered point-estimate engineering gate; this is **not** a statistically
resolved finding that population coverage is below 80%. The frozen stop rule still applies.

Retained raw cheap-candidate measurements: 14,308.

| Numerical control | Greedy agreement | Mean KL |
| --- | --- | --- |
| All-eight original order | 100.000% | 0 |
| All-eight expert-permuted order | 100.000% | 0.00016752 |

The expert-permuted all-eight reduction is a numerical control, not an assumed bitwise
equivalent implementation. A nonzero KL can coexist with perfect greedy agreement.

Post-hoc descriptive OOD task breakdown (not additional gates or a tuning rule):

| Task | States with acceptable choice | States | Coverage |
| --- | --- | --- | --- |
| creative_writing | 107 | 144 | 74.306% |
| summarization | 87 | 112 | 77.679% |

Full-model substitution versus cached evaluator: 29 layouts (28 cheap plus all-eight), 16 prefixes per layout. Exact logit equality in every check: `True`.

Ideal active width = 1 − 0.25 × cheap fraction. This is an arithmetic proxy only. Sparsifying 25%
of one of 24 MLPs removes at most about 1.04% of aggregate MLP arithmetic before router
overhead; it is not a latency, whole-model FLOP or memory-traffic measurement.

## Runtime quality and measured economics

Not run: independent capability confirmation failed.
Inference latency, correction/routing savings and physical DRAM traffic are **unknown**, not zero.
No Nsight counter measurement was attempted without the required quality gate.

Physical DRAM traffic remains unknown unless an explicit valid counter artifact establishes it.
No photonic/simulated bandwidth claim and no free-generation/task-quality claim is made.

## Integrity and reproduction

Original weights unchanged: `True` (execution-record assertion).
Single-GPU wall budget used: 13.421 seconds of 14400 seconds; this is not inference latency.

Registered input snapshots and hashes: `preregistration.json` / `registered_sources/`.
This run retains the confirmation matrix, state-prefix metadata, controls and stop decision.
No fitted selector checkpoint, training log, development choice or runtime measurement exists:
those stages were not executed. Their implementation is present but untested on this study's model outputs.
Activation caches
may be local ignored files; their hashes and reconstruction provenance are recorded.

```powershell
python scripts/verify_expert_recognition.py --run-dir artifacts/expert_recognition/20261014-recognition-v1 --validate-data
python experiments/expert_recognition_report.py --run-dir artifacts/expert_recognition/20261014-recognition-v1
.\scripts\run_expert_recognition.ps1  # new output directory; do not overwrite this run
```

The verifier recomputes recorded metrics, gates and selection from raw artifacts without a
language model or GPU; it does not independently remeasure the GPU or prove chronology from hashes alone.
