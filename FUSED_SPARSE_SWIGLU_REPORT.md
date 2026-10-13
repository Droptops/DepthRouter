# Fused sparse SwiGLU on RTX 4070 SUPER

## Decision

**Do not promote the neuron-sparse path.** No deployable sub-full policy passed the development gate of mean KL strictly below 0.01 and top-1 agreement at least 99%. The best development near-miss retained 4,621/4,864 neurons (95.004%), reached KL 0.001595, but only 97.266% top-1 agreement. It was frozen before held-out access and then fell to KL 0.021670 and 95.508% agreement on 1,024 new held-out prompts.

The isolated 75% prepacked Triton kernel is a real GPU speed upper bound, not a deployable win. It is 1.37-1.95x faster than the strongest graph-captured dense operator across batches 1, 8, and 32, but the mask is non-deployable and the quality gate fails. The fastest dynamic predictor/top-k/gather mode is 7.6-14.2x slower than dense after all routing costs are included.

The separate full-width engineering result is the native BF16 dense SwiGLU under CUDA Graphs. It is faster than every full-width packed or Triton candidate tested and exactly reproduces the reference model.

## Claim boundary

- Branch: `codex/fused-sparse-swiglu`, forked from `research/photonic-parallel-minimal-inference` at `55aec303c7148c228defe9b7dedfc89936a58aa6`.
- PR #14 and `main` were not modified or merged.
- Model: Qwen2.5-0.5B-Instruct revision `7ae557604adf67be50417f59c2c2f167def9a775`, final decoder-layer SwiGLU, BF16.
- Hardware: NVIDIA GeForce RTX 4070 SUPER, 12,282 MiB, driver 610.88, CUDA 12.8, PyTorch 2.7.1, Triton 3.3.1 under Docker Desktop/WSL2.
- All reported latencies are measured CUDA execution. No optical or simulated-photonic number is used as GPU evidence.
- Fixed prepacked masks avoid runtime gathering but only become deployable when the mask is chosen without held-out information. Per-example contribution masks are explicitly oracle-only.

## Protocol and data isolation

The run generated 1,536 deterministic prompts: 256 calibration, 256 development, 512 held-out matched-task, and 512 held-out OOD. Tasks are balanced within each split, template families do not cross splits, and the manifest validator rejects overlap with the historical 512-example study. Manifest SHA-256: `f62fce2ddef0611254ebde442f7b5c05bf901b36c87dbcb443e6e0e1376ce8b9`.

Calibration alone determined fixed contribution-norm, output-sensitive, 64-neuron contiguous-page, and rank-8/rank-16 residual-correction policies. Development selected or rejected policies. The selection record was serialized before held-out activations were captured. Because no sub-full policy passed development, the held-out run is a frozen negative audit, not a tuning round.

Quality was measured by replacing the MLP inside the complete model forward pass. This avoids the reduction-order artifact found when final-token logits were reconstructed offline. Reference logits remain FP32, and reference/candidate top-1 both use `argmax` so tied logits cannot create a false mismatch.

## Isolated kernel result

Each row contains 500 post-warmup CUDA-event measurements. Both dense and fixed sparse paths were graph-captured at the same fixed shape. Confidence values are bootstrap 95% intervals over the measured samples; the conservative speedup divides the dense lower mean bound by the sparse upper mean bound.

| Batch | Strongest dense median | Best 75% sparse median | Sparse kernel | Median speedup | Conservative 95% speedup |
|---:|---:|---:|---|---:|---:|
| 1 | 0.037888 ms | 0.019456 ms | Triton m16/n32/w4 | 1.947x | 1.873x |
| 8 | 0.027648 ms | 0.017408 ms | Triton m16/n32/w4 | 1.588x | 1.546x |
| 32 | 0.041984 ms | 0.030720 ms | Triton m16/n64/w4 | 1.367x | 1.278x |

The fused Triton path combines gate projection, up projection, SiLU, and multiplication, then uses the optimized PyTorch/CUBLAS down projection. At batch 1 the Torch trace reports two kernels per invocation versus five for dense. The 75% packed weights occupy 19,611,648 bytes; analytic minimum operator traffic is 19,622,528 bytes at batch 1. Graph rows allocate no additional memory after capture; that zero is incremental allocation, not total model memory.

The dynamic route includes predictor inference, top-k, sorting, three weight gathers, and SwiGLU execution. Its best observed medians were 0.287520 ms at batch 1 (graph), 0.391168 ms at batch 8 (eager), and 0.472064 ms at batch 32 (graph), only 0.132x, 0.071x, and 0.089x the strongest dense speed. Its graph trace contains 21 CUDA kernels per invocation, and eager incremental peak memory is 19.7-20.8 MB. CUDA Graphs therefore do not rescue dynamic individual-neuron routing.

For full width, native dense CUDA Graph medians are 0.037888, 0.027648, and 0.041984 ms at batches 1, 8, and 32. They are 2.19x, 2.78x, and 1.83x faster than native dense eager medians and remain faster than every full-width prepacked/Triton arm.

## Development quality

| Retention | Per-example contribution oracle KL / top-1 | Fixed output-sensitive KL / top-1 | Contiguous-page KL / top-1 |
|---:|---:|---:|---:|
| 75% | 0.001013 / 97.266% | 0.035191 / 91.797% | 0.034128 / 83.203% |
| 80% | 0.000734 / 97.656% | 0.019876 / 93.750% | 0.018289 / 80.469% |
| 85% | 0.000549 / 98.047% | 0.004854 / 94.531% | 0.011358 / 88.281% |
| 90% | 0.000504 / 97.266% | 0.002899 / 97.266% | 0.005393 / 86.328% |
| 95% | 0.000402 / 98.438% | 0.001595 / 97.266% | 0.003009 / 91.406% |

No sub-full row passes. Even the non-deployable 95% oracle produces 252/256 matches and misses the required 254/256. FP32 down-projection variants do not close the gap. Rank-8/rank-16 residual corrections regress or remain below the gate; the best residual arm reaches only 96.094% top-1 at 95%. The 64-neuron structured-page pivot also fails, so the evidence does not support further individual-neuron routing complexity.

The calibration cancellation ratio is only 0.0352 on average (0.00260 at the fifth percentile), showing that large opposing neuron effects cancel. In the frozen held-out candidate, the 46 top-1 failures have mean reference margin 0.114 versus 1.314 for agreements; 20 failures are below margin 0.125. Small KL is therefore compatible with unacceptable decision flips.

## Frozen held-out audit

| Method | Matched KL / top-1 | OOD KL / top-1 | Combined KL / top-1 | Status |
|---|---:|---:|---:|---|
| Full-width BF16 repacked control | 0 / 100% | 0 / 100% | 0 / 100% | Exact control |
| 75% contribution oracle | 0.000778 / 99.023% | 0.000769 / 98.633% | 0.000774 / 98.828% | Non-deployable; fails top-1 |
| Frozen 95% PyTorch prepacked | 0.002164 / 95.898% | 0.041162 / 95.313% | 0.021663 / 95.605% | Fails |
| Frozen 95% actual Triton | 0.002211 / 95.703% | 0.041130 / 95.313% | 0.021670 / 95.508% | Fails |

For the actual Triton candidate, the bootstrap 95% intervals are [0.008053, 0.038920] for mean KL and [94.238%, 96.777%] for top-1 agreement. OOD generation and string-reversal prompts dominate failures. The historical 75% oracle result (KL 0.000983, 505/512 top-1) is retained only as historical context; it was neither reused for tuning nor included in these confidence intervals.

## Promotion decision and Phase 3

The development quality condition failed, so there is no deployable quality-preserving policy to combine with a kernel. Complete-model prefill/decode timing and free-running task-success evaluation were intentionally not run. Reporting them for a known-invalid policy would not support promotion and would spend held-out task evidence on a rejected design.

Promotion conditions therefore resolve as follows:

- Mean KL below 0.01: failed on the frozen held-out candidate.
- Top-1 agreement at least 99%: failed on development and held-out.
- No unacceptable held-out task degradation: not opened as a generation claim after the prerequisite failed.
- Credible end-to-end improvement: not tested because no valid policy exists.
- Predictor, movement, routing, kernel, and fallback overhead: not satisfied because no deployable path or fallback policy exists. The dynamic benchmark does include predictor, selection, and all three weight gathers; any fallback would add cost.

The recommended pivot is operator-level specialization: fixed contiguous blocks, whole projections, or layer-level bypass/routing with dense GEMM-compatible shapes. The first contiguous 64-neuron page experiment here is negative, so future work should change the operator boundary rather than merely grouping the same per-neuron score.

## Profiling limitation

Nsight Compute 2026.3.1 was installed in a separate reproducible image and invoked against the selected batch-1 Triton kernel. The host rejected performance-counter access with `ERR_NVGPUCTRPERM`. Consequently measured HBM traffic and achieved occupancy remain null; the report preserves the exact log and does not substitute estimates. Torch Chrome traces, kernel-launch counts, analytic minimum bytes/MACs, latency, throughput, and incremental-memory results are available.

## Reproduction and artifacts

Run the full experiment from this branch:

```powershell
.\scripts\run_fused_sparse_swiglu.ps1 -RunId <new-run-id> -Stage all
```

Attempt the Nsight capture after enabling NVIDIA performance counters on the Windows host:

```powershell
.\scripts\profile_fused_swiglu_ncu.ps1 -Method triton -Batch 1 -Fraction 0.75 -RunId <new-run-id>
```

Canonical machine-readable results are under `artifacts/fused_sparse_swiglu/20261013-validated-final-v2/`: `summary.json`, `kernel_benchmarks.json/csv`, development and held-out quality JSON/CSV, confidence intervals, low-margin failures, the manifest, research log, failure audit, environment snapshot, Torch profiler traces, and the Nsight permission-failure record. Large intermediate tensors are reproducible but git-ignored.
