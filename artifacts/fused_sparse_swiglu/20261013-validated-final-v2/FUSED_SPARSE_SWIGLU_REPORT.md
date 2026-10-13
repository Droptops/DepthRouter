# Fused sparse SwiGLU result

**Decision:** `NO_DEPLOYABLE_SPARSE_POLICY_PASSED_DEVELOPMENT`

No sub-full-width deployable policy passed the KL < 0.01 and top-1 >= 99% development gate. The frozen 95.004%-width near-miss produced KL 0.021670 and 95.508% top-1 agreement on 1,024 new held-out prompts. End-to-end and generation claims were therefore withheld.

The graph-captured 75% Triton operator is a non-deployable upper bound: 1.947x, 1.588x, and 1.367x median speedup at batches 1, 8, and 32. The fastest dynamic predictor/top-k/gather mode remains 7.6-14.2x slower than dense, and its graph trace launches 21 kernels per invocation. Native full-width BF16 dense under CUDA Graphs is the fastest full-width implementation and the full-width repacked control is exact.

Nsight Compute 2026.3.1 was installed and attempted, but the host returned `ERR_NVGPUCTRPERM`; HBM traffic and achieved occupancy remain explicitly null. See `profiler/ncu-triton-b1-f0.75.log` and `profiler/nsight_status.json`.

The complete reproducible analysis, limitations, tables, and commands are in the repository-root `FUSED_SPARSE_SWIGLU_REPORT.md`. Machine-readable details are in `summary.json`, `kernel_benchmarks.json/csv`, `development_quality.json/csv`, `heldout_quality.json/csv`, `low_margin_failures.json`, and `failure_audit.json`.
