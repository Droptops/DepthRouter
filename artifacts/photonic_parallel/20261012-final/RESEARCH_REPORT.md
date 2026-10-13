# Photonic parallel minimum-inference research report

## Scope and claim boundary

This run used a real `NVIDIA GeForce RTX 4070 SUPER` for CUDA measurements. Optical
numbers are simulations only. Analytic weight-byte counts are not physical HBM
traffic measurements. The model was pinned to `Qwen/Qwen2.5-0.5B-Instruct` revision
`7ae557604adf67be50417f59c2c2f167def9a775` in `bfloat16` with seed `20261012`.

## Method

- Dataset: 24 locally generated nonsensitive records, with disjoint calibration,
  development, and held-out prompts across four task families.
- Dense baseline: synchronized warmups and repeated prefill, one-token decode,
  time-to-first-token, and deterministic generation at batches 1, 8, and 32.
- Correctness control: the middle Qwen SwiGLU layer was decomposed into 2, 4, 8,
  and 16 disjoint neuron groups and summed before the residual connection.
- Selection: random, static activation, weight magnitude, calibration contribution,
  task-conditioned, state-transition-conditioned, and a non-deployable held-out
  oracle were tested at 10%, 25%, 50%, 75%, and 100% union retention.
- CUDA operators: dense, one packed sparse GEMM, serial groups, concurrent CUDA
  streams, and a state-routed packed path include host dispatch, synchronization,
  aggregation, and routing in measured wall time.
- Photonics: elapsed latency, device occupancy, throughput, energy, area, lane
  utilization, programming amortization, conversion, readout, cooldown, controller,
  and aggregation were modeled separately.

## Results

- Dense SDPA batch-1 prefill median: `13.7688 ms`.
- Dense SDPA batch-1 decode median: `13.5047 ms`.
- Exact grouped reconstruction gate: `PASS`.
- Selected deployable arm: `calibration_activation`, layers `23`, retention
  `100%`, KL `0`, top-1
  `100.00%`.
- Free-running exact agreement with dense: `100.00%`.
- Fastest unconstrained CUDA operator arm: `1.187x`
  (`packed_fraction_sweep`); this is not a valid quality-preserving
  speedup when its fraction fails the fidelity gate.
- Quality-arm packed operator speedup: `1.171x`
  (conservative 95% bound `1.049x`).
- Measured end-to-end batch-1 speedup: `1.042x`
  (conservative 95% bound `1.012x`).
- Best true parallel batch-1 operator result: `0.457x`
  versus dense (values below 1 are slowdowns).
- Central optical model: `10.978x` versus the
  measured dense GPU MLP comparator, elapsed `7477.7 ns`.
- Optical sweep break-even fraction: `93.91%`
  of explicitly modeled points.
- Strict combined gate: `FAIL`.

## Strongest negative result

No deployable sub-100% neuron set passed KL < 0.01 and top-1 >= 99%; state-aware routing also trailed the best matched static selector.

## Caveats

The dataset is intentionally small and synthetic. The packed PyTorch kernels are
real measured kernels but are not custom fused sparse kernels. Nsight Compute was
not present, so physical off-chip bytes and achieved occupancy are unavailable;
the report does not infer them from neuron count. CUDA streams may overlap kernel
timelines, but they are not independent hardware accelerators. The oracle uses
held-out activation information and is excluded from deployable comparisons.

## Optical falsification boundary

The central cost model includes 20 ns feedback, nine iterations, 25 us matrix
programming amortized over 256 inferences, 6.4 us readout, 2.5 us cooldown, and
controller/aggregation costs. The hardware characteristics with the largest
one-at-a-time modeled latency sensitivity are:
lanes, residency_lifetime_inferences, readout_ns, aggregation_ns.
These require independent hardware measurement before any hybrid speedup claim.

## Reproduction

```powershell
./scripts/run_photonic_docker.ps1 -Config configs/photonic_parallel_quick.json -RunId reproduced
```

Artifacts: `accuracy_latency_pareto.png, photonic_break_even_heatmap.png` plus JSON/CSV tables and `cuda_stream_trace.json`.
