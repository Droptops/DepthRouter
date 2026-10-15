## Outcome

The preregistered frozen-C-29 follow-up stopped at independent capability confirmation, before any selector fitting or timing.

| Fresh confirmation stratum | Acceptable cheap-candidate coverage (95% context CI) | Gate |
| --- | --- | --- |
| Matched: 64 contexts / 255 states | 88.235% [83.465, 92.195] | Pass |
| OOD: 64 contexts / 256 states | 75.781% [70.693, 81.250] | Fail: requires at least 80% |

Portfolio advantage over the frozen fixed-subset-plus-dense oracle passed in both strata (7.843 and 6.641 percentage points of ideal active width). The failure is insufficient observed OOD cheap-candidate coverage, not lack of advantage over the fixed subset. Its interval spans 80%; this is a point-estimate engineering failure, not a statistically resolved population-level failure. Recognition itself was not tested.

No selector was trained. No new calibration/development/held-out model outputs, runtime-quality runs, latency benchmarks, free-generation evaluations or physical-counter attempts were produced. No speedup or DRAM saving is claimed. Prior negative findings and the global hardware-counter gate are unchanged.

## Scope and integrity

- Stacked on PR #17 (`codex/expert-construction-audit`), base commit `a803bf8a967e9437344e897a9bc8677f9f21ffea`. Do not merge automatically or advance upstream PRs.
- Freeze the highest-old-development-advantage partition C-29 and its old development fixed candidate 8. No partition switching, mask/objective tuning or seed search after confirmation.
- Same pinned Qwen2.5-0.5B-Instruct, frozen original weights, final MLP only, all 28 six-of-eight combinations.
- Fresh manifest has 1,728 contexts, excluding all 6,163 historical prompts and 2,304 prior expert contexts. Sizes were reduced before model access to retain strict exclusions and 12.5% enterprise-topic quota; the original proposed fresh-data allocation was infeasible.
- Snapshot 26 scientific inputs before model loading. Retain 14,308 raw cheap-candidate comparisons and 511 state records. Sorted all-eight control: 100% agreement / KL 0. Permuted control: 100% agreement but small nonzero KL. All 29 layouts × 16 full-model replay prefixes match cached logits exactly.
- Original weights unchanged by execution hash check. Single-GPU cycle wall time 13.421 seconds on RTX 4070 SUPER; this is not inference latency.
- Preserve the entire conditional selector/runtime implementation with CPU tests. Later GPU stages remain unexecuted and cannot be claimed validated by this run.

## Artifacts and reproduction

Validation: 233 tests pass in the pinned container; repository-wide Ruff passes. The complete CPU-only integrity/data audit passes.

- `EXPERT_RECOGNITION_PREREGISTRATION.md`
- `EXPERT_RECOGNITION_REPORT.md`
- `data/expert_recognition/`
- `artifacts/expert_recognition/20261014-recognition-v1/`

```powershell
python -m ruff check .
python -m pytest
python scripts/verify_expert_recognition.py --run-dir artifacts/expert_recognition/20261014-recognition-v1 --validate-data
python experiments/expert_recognition_report.py --run-dir artifacts/expert_recognition/20261014-recognition-v1
.\scripts\run_expert_recognition.ps1 -RunId <new-id> -SkipBuild
```

The launcher refuses to overwrite an existing run. The full verifier recomputes source hashes, data exclusions, numerical checks, bootstrap intervals and the stop decision without loading a language model or using a GPU. Large activation caches remain locally retained/ignored with recorded hashes; raw scalar candidate matrices and state/prefix metadata are committed. Exact snapshot hashes remain authoritative; `--allow-current-newline-only` is available for a checkout with explicitly acknowledged CRLF/LF conversion.

This is a completed negative confirmation cycle, not evidence that no partition or future selector can work.
