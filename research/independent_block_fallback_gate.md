# Independent-distribution causal block fallback gate

## Result

The final-layer causal block path clears the preregistered token gate on a public prompt distribution independent of the synthetic enterprise generator. Across three pairwise-disjoint 200-prompt Stanford Alpaca test sets (1,800 autoregressive decode states), the frozen probability-margin policy reached `0.99944` top-1 agreement and `0.00820` mean KL at `0.84389` average exact-MLP execution. The combined 95% Wilson lower bound for top-1 agreement is `0.99686`.

This is a behavioral and analytic result, not a physical speed or bandwidth result.

## Frozen protocol

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Model/tokenizer revision: `7ae557604adf67be50417f59c2c2f167def9a775`
- Precision/device: FP16 on RTX 4070 SUPER
- External source: Stanford Alpaca `alpaca_data.json` at commit `761dc5bfbdeeffa89b8bff5d038781a4055f796a`
- Source after exact-text deduplication: 52,002 prompts; SHA-256 `a6867486baba6e273959daae7c1adcb29ccd222fb5f88257d955c4b99237eefb`
- Split per replication: deterministic shuffled 200-prompt calibration set plus disjoint 200-prompt test set; three greedy decode states per prompt
- Test split seeds: `20261013`, `20261014`, and post-freeze confirmation `20261015`
- Pairwise test-set overlap: zero prompts
- Structured unit: 16 contiguous final-layer MLP neurons; 228/304 blocks selected (`75%` exact sparse execution)
- Causal address: rowwise 2-bit gate/up projections with FP16 per-row scales and FP16 down-column norms
- Block score: summed approximate `|SiLU(gate) * up| * ||down column||`
- Confidence: sparse candidate probability margin
- Calibration guardrail: top-1 at least `0.998` and mean KL below `0.0075`
- Fallback accounting: accepted states execute 75% of exact MLP neurons; rejected states are charged 100% because omitted blocks must be completed

The packed address is 2,208,256 bytes against 26,148,864 FP16 bytes in the final-layer MLP (`8.44494%`). Metadata is added to, not hidden inside, the average exact-execution fraction.

## Held-out results

| Split | Sparse acceptance | Avg. exact execution | Metadata + execution | Top-1 | 95% Wilson lower | Mean KL | Exact free-running prompts | Raw sparse-only top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 20261013 | 0.63667 | 0.84083 | 0.92528 | 1.00000 | 0.99364 | 0.00824 | 1.000 | 0.940 |
| 20261014 | 0.61500 | 0.84625 | 0.93070 | 1.00000 | 0.99364 | 0.00831 | 1.000 | 0.935 |
| 20261015, post-freeze | 0.62167 | 0.84458 | 0.92903 | 0.99833 | 0.99062 | 0.00806 | 0.995 | 0.925 |
| Aggregate | 0.62444 | 0.84389 | 0.92834 | 0.99944 | 0.99686 | 0.00820 | 0.9983 | — |

The raw 75% sparse path fails badly on every split. The positive frontier comes from a causal low-bit block address plus a calibrated fallback, not from an easy test distribution or an unreported dense path.

Exact result payloads:

- `research/results/block_lowbit_fallback_alpaca_split_20261013.json`
- `research/results/block_lowbit_fallback_alpaca_split_20261014.json`
- `research/results/block_lowbit_fallback_alpaca_split_20261015.json`

## Claim boundary

- PyTorch forward hooks still execute the dense MLP before replacing its final-token output. No physical selective execution occurs in this harness.
- The diagnostic reconstructs packed 2-bit weights into FP32 tensors. No packed low-bit projection kernel has been implemented or timed.
- Confidence requires a full-vocabulary candidate projection before fallback. A rejected state would need the omitted MLP blocks and another final normalization/output projection; this overhead is not included in the final-layer MLP fraction.
- The result applies to one MLP layer. The average metadata-inclusive analytic margin is `7.1662%` of that layer, roughly `0.19%` of whole-model FP16 parameter bytes before dispatch, candidate-projection, and kernel overhead.
- Physical bytes/token and latency are therefore `null`.

## Next gate

The quality objective is achieved. The remaining engineering question is whether a fused packed-address/block-sparse implementation can preserve the behavior while producing any measurable end-to-end benefit. A kernel should be attempted only as a measurement experiment, with the candidate projection and fallback recomputation charged in full.
