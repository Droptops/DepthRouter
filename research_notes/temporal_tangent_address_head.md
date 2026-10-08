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

## First diagnostic result

Qwen2.5-0.5B-Instruct, last layer, 30% active rows, rank 16, 160 training
states. Steady-state steps only (180 held-out states):

| arm | mean KL (nats) | p95 KL | top-1 | oracle-set recall |
|---|---|---|---|---|
| `student_history` | 0.149 | 0.67 | 77.8% | 56.3% |
| `oracle_history` | 0.144 | 0.66 | 78.9% | 56.7% |
| `no_history` | 0.168 | 0.81 | 77.2% | 55.3% |
| `copy_previous` | 0.245 | 0.81 | 76.7% | 48.5% |
| `oracle_selection` | 0.002 | 0.008 | 97.2% | 100% |

- Exposure bias is negligible: teacher forcing the history gains only 0.005
  nats over the deployable arm.
- The head beats both no-history and persistence-only, so history and learning
  each add a little.
- The gap to the oracle (~56% recall, ~60x the KL) is the real problem. This
  points at the data bottleneck: scale training states before trying scheduled
  sampling or higher ranks.
