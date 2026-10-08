# Temporal tangent address head

## Result so far

On the held-out 20-prompt, 10-step Qwen2.5-0.5B-Instruct run
(`experiments/hf_temporal_tangent_distilled.py`), the rank-16 learned address
head fails the quality gate:

| active rows | mean KL (nats) | top-1 agreement | oracle-set recall |
|---|---|---|---|
| 30% | 0.139 | 78% | 57.4% |
| 40% | 0.105 | 83% | — |

The oracle logit-tangent frontier still looks promising, but nothing deployable
has been shown yet.

## Suspected causes

1. **Data bottleneck.** The head has 111,361 trainable parameters but sees only
   160 training states (20 prompts x 8 decode steps).
2. **Exposure bias.** Training feeds the previous *oracle* mask as history;
   held-out evaluation feeds the head's own previous *prediction*.

## Diagnostic

`experiments/hf_temporal_tangent_exposure_diagnostic.py` trains the same head
and scores every held-out state under five arms: `student_history`
(deployable), `oracle_history` (teacher forced), `no_history`, `copy_previous`
(persistence-only: reuse the last oracle set) and `oracle_selection` (ceiling).
It reports first-step and steady-state metrics separately. It runs as the
`temporal-tangent-exposure-diagnostic` CI job.

How to read it:

- `oracle_history` well above `student_history` points to exposure bias. Next
  step is to train on predicted-history rollouts (scheduled sampling or DAgger).
- `student_history` close to `no_history` means the history input is not
  helping.
- `copy_previous` matching or beating the head means the learned part is not
  earning its metadata cost.
- If all head arms are well below `oracle_selection` and close to each other,
  scale training data before trying higher-rank sweeps.
