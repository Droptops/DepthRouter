# Morning summary

- Docker startup was repaired by quarantining inaccessible transient AF_UNIX
  runtime directories after one graceful stop attempt and targeted termination of
  confirmed Docker-owned orphan processes. Persistent images, containers,
  volumes, VHDX files, and WSL registrations were preserved.
- The experiment ran in Docker with PyTorch 2.7.1+cu128 on an NVIDIA GeForce RTX
  4070 SUPER (12,282 MiB), using Qwen2.5-0.5B-Instruct revision
  `7ae557604adf67be50417f59c2c2f167def9a775` in BF16.
- Completed: eager/SDPA dense baselines; exact 2/4/8/16-way SwiGLU decomposition;
  105 held-out selection arms; serial, CUDA-stream, packed, and routed operator
  benchmarks; free-running checks; and a 1,215-point optical sweep.
- Best verified finding: independent SwiGLU group contributions reconstruct the
  model within BF16 tolerance and preserve top-1 plus deterministic generations.
- Strongest negative finding: every deployable policy needed 100% of selected
  layers. At 75% on layer 23, state-transition KL was 0.174 versus 0.054 for the
  best static selector; both failed the quality gate.
- Actual parallel GPU result: the best two-stream arm was 0.457x dense at batch 1
  (about 2.19x slower). A non-parallel 100%-width repacked control reached 1.042x
  end-to-end at batch 1 with a conservative 95% bound of 1.012x, but was
  approximately neutral at batches 8 and 32.
- Free-running candidate/dense agreement was 100%, but manifest exact-match task
  acceptance was only 25% for both, so task acceptance failed.
- Strict gates: decomposition PASS; relative distribution fidelity PASS only at
  100%; nontrivial subset FAIL; task acceptance FAIL; true parallel performance
  FAIL; overall FAIL.
- Optical simulation: optimistic 102.58x and central 10.978x beat the measured
  dense GPU MLP comparator; pessimistic 0.00495x failed. The central result assumes
  20 ns feedback, nine iterations, four lanes, 6.4 us readout, 2.5 us cooldown,
  and programming amortized over 256 inferences. It is simulated, not measured.
- Branch: `research/photonic-parallel-minimal-inference`. The recommended next
  experiment is a fused grouped GEMM with Nsight Compute before more selector work.

See [RESEARCH_REPORT.md](RESEARCH_REPORT.md) and the
[raw run artifacts](artifacts/photonic_parallel/20261012-final).
