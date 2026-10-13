# Morning summary

- Docker: repaired separately by quarantining inaccessible transient AF_UNIX
  runtime directories; no persistent Docker data was deleted.
- CUDA environment: PyTorch `2.7.1+cu128`, CUDA `12.8`,
  `NVIDIA GeForce RTX 4070 SUPER` with `11.99 GiB`.
- Experiments completed: dense baseline, exact 2/4/8/16-way decomposition,
  held-out retention sweep, sequential/concurrent CUDA benchmarks, deterministic
  free-running comparison, and optical sensitivity sweep.
- Best verified finding: exact 2/4/8/16-way SwiGLU decomposition preserved top-1
  logits and deterministic generated tokens within the declared BF16 tolerance.
- Unconstrained fastest operator: `packed_fraction_sweep` reached
  `1.187x`, but this is not a
  quality-preserving architecture result. Quality-arm end-to-end batch-1 was
  `1.042x` with conservative bound
  `1.012x`.
- Strongest negative finding: No deployable sub-100% neuron set passed KL < 0.01 and top-1 >= 99%; state-aware routing also trailed the best matched static selector.
- Strict gates: decomposition `PASS`,
  distribution quality `PASS`, nontrivial subset
  `FAIL`, free-running
  dense agreement `PASS`, task exact-match acceptance
  `FAIL`
  (`25.00%`), packed end-to-end
  `PASS`, parallel GPU performance
  `FAIL`;
  overall `FAIL`.
- Optical central case: `10.978x` modeled, with
  20 ns feedback, nine iterations, 6.4 us readout, 2.5 us cooldown, four lanes,
  and programming amortized over 256 inferences. This is simulated, not measured.
- Next experiment: profile a fused grouped GEMM with Nsight Compute, preserving the
  same selected union, before expanding selector complexity.
