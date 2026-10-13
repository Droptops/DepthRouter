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

## Predictive working-set follow-up

- The follow-up used 256 calibration, 128 development, and 512 held-out prompts;
  predictor selection was complete before held-out activations were captured.
- The 100%-width control reconstructed dense output exactly.
- The non-deployable 75% contribution-norm oracle passed KL (0.000983) but
  narrowly missed top-1 agreement: 98.633%, or 505/512 instead of the required
  507/512.
- The selected rank-64 state-plus-transition predictor recovered 78.892% of the
  oracle set, with KL 0.103771 and 68.945% top-1 agreement. It failed decisively
  and did not outperform a static set on output quality.
- The automatic 16-contiguous-page pivot also failed at 75% width; even its
  oracle reached only 66.602% agreement. The raw physical layout is therefore
  not a useful specialization unit.
- The fused benchmark was correctly skipped because quality did not pass. The
  next candidate should first construct meaningful operator units (permuted
  co-active blocks, experts with residuals, or block/low-rank components) and
  establish oracle sufficiency before router or kernel work.

See [PREDICTIVE_WORKING_SET_REPORT.md](PREDICTIVE_WORKING_SET_REPORT.md) and the
[validated artifacts](artifacts/predictive_working_set/20261013-validated-final).
