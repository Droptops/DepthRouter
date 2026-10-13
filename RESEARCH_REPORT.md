# Photonic parallel minimum-inference research report

## Executive result

The tested architecture did **not** pass the combined promotion gate.

- Exact SwiGLU neuron-group decomposition worked within BF16 reduction-order
  tolerance for 2, 4, 8, and 16 groups.
- No deployable 10%, 25%, 50%, or 75% retained-neuron policy passed both held-out
  KL `< 0.01` and top-1 agreement `>= 99%`. The minimum sufficient fraction was
  100% for every deployable method, tested layer set, and task family.
- State-transition conditioning did not beat the best matched static selector.
- The fastest true CUDA-stream parallel arm was 0.457x dense at batch 1, or about
  2.19x slower after dispatch, synchronization, and aggregation.
- A 100%-width repacked control was 1.171x faster at the isolated MLP operator and
  1.042x faster end to end at batch 1 (conservative 95% bound 1.012x), but this is
  a layout/control result, not a minimum-sufficient or parallel-architecture win.
- The central optical simulation beat the measured dense GPU MLP comparator by
  10.978x, but the pessimistic joint assumptions lost badly. No physical optical
  hardware was measured.

## Environment and reproducibility

The run used:

- NVIDIA GeForce RTX 4070 SUPER, 12,282 MiB VRAM, compute capability 8.9;
- NVIDIA Windows driver 610.88;
- Docker Desktop 4.82.0 and Docker Engine 29.6.1 on WSL2;
- PyTorch 2.7.1+cu128, CUDA runtime 12.8, cuDNN 9.7.1;
- Transformers 4.57.1 and NumPy 2.2.6;
- Qwen/Qwen2.5-0.5B-Instruct at immutable revision
  `7ae557604adf67be50417f59c2c2f167def9a775`;
- BF16 inference and seed `20261012`.

VRAM headroom after loading the measured model was 80.1% in the fresh run, well
above the requested 15% minimum. The Docker worker used a 26 GiB RAM cap, a
persistent host Hugging Face cache, host-mounted checkpoints, and a scoped Windows
execution-state request that is automatically released when the worker exits.

Docker initially failed on inaccessible stale AF_UNIX reparse points. The repair
stopped confirmed Docker-owned orphan processes, terminated only the Docker WSL
distribution, and renamed the two disposable runtime IPC directories. No reset,
prune, VHDX change, volume deletion, reinstall, WSL unregister, or reboot occurred.
The full evidence is in
[`artifacts/docker_diagnostics/DIAGNOSTIC.md`](artifacts/docker_diagnostics/DIAGNOSTIC.md).

Reproduce the measured run from PowerShell:

```powershell
./scripts/run_photonic_docker.ps1 `
  -Config configs/photonic_parallel_quick.json `
  -RunId reproduced
```

Use `scripts/start_overnight_worker.ps1` for a detached, resumable host worker.

## Dataset and evaluation

The manifest contains 24 locally generated nonsensitive records: eight
calibration, eight development, and eight held-out examples, balanced across
classification, extraction, numerical reasoning, and short generation. Prompts,
entities, numbers, and wording differ across splits. The oracle may inspect the
held-out activation but is excluded from all deployable comparisons.

Teacher-forced metrics cover answer tokens. Free-running evaluation used
deterministic greedy decoding. The 100%-width selected control reproduced the
dense generated tokens exactly on all four quick-run examples, but both dense and
candidate outputs achieved only 25% exact match against the manifest answers.
Task-level acceptance therefore failed even though relative preservation passed.

## Phase 1 — dense baselines

All timings include warmup, CUDA synchronization, repeated measurements, and
fixed shapes. Values below are medians.

| Backend | Batch | Prefill ms | Decode ms/token | TTFT ms | 8-token generation ms |
|---|---:|---:|---:|---:|---:|
| eager | 1 | 17.29 | 16.68 | 19.41 | 143.77 |
| eager | 8 | 18.70 | 16.04 | 19.46 | 148.17 |
| eager | 32 | 38.95 | 16.09 | 35.23 | 164.09 |
| SDPA | 1 | 13.77 | 13.50 | 14.51 | 108.17 |
| SDPA | 8 | 17.66 | 13.27 | 18.91 | 142.65 |
| SDPA | 32 | 37.89 | 14.05 | 34.13 | 159.77 |

The SDPA held-out response-token cross-entropy was 2.0587 nats. Peak incremental
VRAM was 12.83 MiB, 135.26 MiB, and 541.38 MiB at batches 1, 8, and 32.

## Phase 2 — exact decomposition control

Layer 12's 4,864 intermediate neurons were partitioned exhaustively and
disjointly. Contributions were summed before the original residual connection.

| Groups | Local max abs | Local relative L2 | Full-logit mean KL | Top-1 | Generated tokens |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.003906 | 0.003125 | 0.002662 | 100% | identical |
| 4 | 0.007812 | 0.003798 | 0.001983 | 100% | identical |
| 8 | 0.007812 | 0.004783 | 0.002834 | 100% | identical |
| 16 | 0.007812 | 0.005729 | 0.003334 | 100% | identical |

The outputs are not bitwise identical because BF16 reduction order changed. They
pass the declared tolerance: local relative L2 below 1%, full-logit KL below
0.01, unchanged top-1, and identical deterministic generation.

## Phase 3 — minimum-sufficient sets

The sweep tested random, static activation, static weight, calibration
contribution, task-conditioned, state-transition-conditioned, and oracle
selection at 10%, 25%, 50%, 75%, and 100% union retention. Layer sets were
`[23]`, `[12, 23]`, and `[0, 6, 12, 18, 23]`.

No deployable sub-100% arm passed. On layer 23 at 75% retention:

| Selector | Deployable | Mean KL | Top-1 agreement |
|---|---:|---:|---:|
| static activation | yes | 0.0543 | 88.46% |
| calibration contribution | yes | 0.0713 | 86.54% |
| task-conditioned | yes | 0.0737 | 88.46% |
| state-transition-conditioned | yes | 0.1743 | 86.54% |
| oracle | no | 0.00054 | 100% |

The oracle shows that a sufficient 75% set exists when test activation
information is allowed, but the deployable selectors did not identify it. The
state-aware hypothesis is falsified for this implementation and calibration set;
it is not evidence that all learned selectors must fail.

## Phase 4 — measured CUDA parallelism

The selected deployable control retained 100% of layer 23. Measured batch-1 MLP
latencies were 0.082 ms dense, 0.070 ms repacked, and 0.316 ms for the routed
packed path. The 100%-width repacked control's small gain did not persist as a
clear win at batches 8 and 32.

For exactly the same selected union, splitting into parallel CUDA-stream groups
lost to both dense and serial execution:

| Batch | Best branches | Parallel speedup vs dense | Interpretation |
|---:|---:|---:|---|
| 1 | 2 | 0.457x | 2.19x slower |
| 8 | 2 | 0.478x | 2.09x slower |
| 32 | 2 | 0.520x | 1.92x slower |

The profiler trace showed launches on multiple streams, but summed stream time
never overcame host dispatch and aggregation. Nsight Compute was unavailable, so
physical HBM traffic and achieved occupancy are explicitly unmeasured. Analytic
weight bytes are retained only as estimates and are not used as proof of reduced
off-chip traffic.

End-to-end 100%-width repacked medians were 1.042x, approximately 1.00x, and
approximately 1.00x dense at batches 1, 8, and 32. Because there was no
nontrivial sufficient subset and true grouped parallelism slowed down, H4 fails.

## Phase 5 — photonic simulator

The simulator distinguishes feedback time, elapsed latency, device occupancy,
throughput, energy, area, and lane utilization. It models programming
amortization, input conversion, readout, cooldown, controller, aggregation, lane
waves, and explicitly configured overlap.

The GPU comparator is the measured dense batch-1 MLP median of 82,089 ns.

| Scenario | Modeled speedup | Break-even |
|---|---:|---:|
| optimistic | 102.58x | yes |
| central | 10.978x | yes |
| pessimistic | 0.00495x | no |

The central case uses 20 ns feedback, nine iterations, four lanes, 25 us matrix
programming amortized across 256 inferences, 250 ns conversion, 6.4 us readout,
2.5 us cooldown, 300 ns aggregation, and 250 ns controller latency. It yields
7,477.7 ns elapsed latency, 9,427.7 ns device occupancy, 105.3 microjoules per
inference under the assumed powers, 8 mm2 area, and full lane utilization.

Of 1,215 grid points varying feedback, iterations, lanes, programming, readout,
and cooldown, 93.91% beat this small GPU-operator comparator. This percentage is
not a hardware probability. The joint pessimistic case fails, so the modeled win
is conditional. The smallest provisional measurement set is optical lane count,
operator residency lifetime, readout latency, and aggregation latency; matrix
programming and conversion remain required secondary checks.

## Hypothesis decisions

| Hypothesis | Decision | Evidence |
|---|---|---|
| H1: small sufficient MLP subset | fail in tested scope | every deployable arm required 100% |
| H2: independent exact decomposition | pass within BF16 tolerance | all 2/4/8/16 controls passed |
| H3: state-aware beats matched baselines | fail | state-conditioned KL was worse at matched fractions |
| H4: structured GPU parallelism wins | fail | best parallel arm was 0.457x dense at batch 1 |
| H5: optical break-even region exists | modeled conditional pass | central/optimistic pass; pessimistic fails |

## Caveats and next falsification experiment

This is a small synthetic task suite, one model revision, BF16, and selected layer
sets on one consumer GPU. It does not establish general impossibility. The next
high-information experiment is a fused grouped GEMM or contiguous block-sparse
kernel profiled with Nsight Compute, using the same preregistered masks and
including routing. Selector complexity should not be expanded until that kernel
either removes the launch/aggregation loss or confirms it.

Raw measurements, confidence intervals, failure examples, environment metadata,
plots, and the CUDA trace are under
[`artifacts/photonic_parallel/20261012-final`](artifacts/photonic_parallel/20261012-final).

## Predictive working-set follow-up (2026-10-13)

A larger, split-isolated follow-up tested whether a lightweight causal predictor
could recover the layer-23 contribution-norm oracle's 75% neuron set. It used 256
calibration, 128 development, and 512 held-out prompts. Predictor feature form
and rank were fixed before the held-out split was captured.

The 100%-width reconstruction control produced effectively zero KL and 100%
top-1 agreement. The non-deployable 75% oracle reached KL 0.000983 but only
98.633% top-1 agreement, narrowly missing the 99% gate. The selected rank-64
state-plus-transition predictor recovered 78.892% of oracle members but reached
KL 0.103771 and 68.945% agreement. A calibration-static set was slightly better
on output quality, so causal prediction did not justify its added machinery.

The failure triggered a structured 16-page, 12-selected-page pivot. Even its
page oracle failed (KL 0.208599, 66.602% agreement), establishing that the raw
contiguous weight layout is not a sufficient operator specialization. The fused
performance test was skipped under the preregistered decision rule.

See [`PREDICTIVE_WORKING_SET_REPORT.md`](PREDICTIVE_WORKING_SET_REPORT.md) and the
[`validated raw artifacts`](artifacts/predictive_working_set/20261013-validated-final).
