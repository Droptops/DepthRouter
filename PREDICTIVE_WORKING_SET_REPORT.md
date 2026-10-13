# Predictive working-set feasibility

## Decision

**The individual-neuron causal predictor failed the preregistered quality gate.
The contiguous operator-page pivot also failed. The fused-performance test was
therefore not run.**

This is a retained negative result, not a reason to add a more complicated
neuron selector. It redirects the research toward operator-level structures that
are learned or laid out as units, rather than arbitrary neurons gathered at
runtime.

## Protocol

The run used Qwen2.5-0.5B-Instruct revision
`7ae557604adf67be50417f59c2c2f167def9a775`, BF16, and layer 23 on an NVIDIA
GeForce RTX 4070 SUPER. A deterministic 896-prompt manifest contains:

- 256 calibration prompts for fitting predictor parameters;
- 128 development prompts for choosing feature form and predictor rank;
- 512 held-out prompts opened only after that choice;
- equal classification, extraction, numerical, and short-generation strata;
- split-specific templates with no prompt or template-family overlap.

Candidate predictors used rank 16 or 64 and either the current MLP input or the
current input plus its previous-token transition. The target was the per-token
75% set induced by absolute SwiGLU activation times the corresponding down-projection
column norm. This contribution-norm oracle is a diagnostic upper bound, not a
deployable selector and not a combinatorial proof of the globally optimal set.

Quality was evaluated with dense-shape counterfactual masking to isolate
selection error from packed-kernel BF16 roundoff. The full-width control exactly
reconstructed the dense result. Promotion required mean KL strictly below 0.01
and at least 99% dense top-1 agreement.

## Held-out result

| Method | Width | Mean KL (95% CI) | Top-1 agreement (95% CI) | Oracle recall | Gate |
|---|---:|---:|---:|---:|---:|
| full-width control | 100% | ~0 | 100.000% (100%, 100%) | n/a | pass |
| contribution-norm oracle | 75% | 0.000983 (0.000896, 0.001074) | 98.633% (97.461%, 99.609%) | 100% | fail |
| rank-64 causal predictor | 75% | 0.103771 (0.091931, 0.115886) | 68.945% (65.039%, 72.852%) | 78.892% | fail |
| calibration-static control | 75% | 0.085294 (0.072767, 0.097571) | 71.875% (68.164%, 75.586%) | 78.735% | fail |

The oracle passed the KL threshold but produced 505/512 matching top-1 tokens.
The gate required at least 507/512, so it missed by two examples. Its interval
crosses 99%; that makes the oracle result borderline rather than evidence of a
large deficit. The learned predictor is not borderline: it missed 159 top-1
tokens and its entire reported agreement interval lies far below 99%.

Development selection chose the rank-64 state-plus-transition predictor. It has
430,912 parameters (about 0.86 MB in BF16) and an analytic 425,984 MACs/token,
3.26% of the target SwiGLU's dense MAC count. Oracle-set recall fell from 80.990%
on development to 78.892% held out. The static control slightly outperformed it
on output quality, so the extra causal model did not buy useful generalization.

## Structured operator pivot

The failed neuron gate automatically triggered a hardware-regular alternative:
4,864 intermediate neurons were divided into 16 contiguous pages of 304 neurons,
and exactly 12 pages (75%) were selected per token. A rank-16 state predictor was
chosen on development data.

| Method | Mean KL (95% CI) | Top-1 agreement (95% CI) | Oracle-page recall | Gate |
|---|---:|---:|---:|---:|
| page oracle | 0.208599 (0.178173, 0.241866) | 66.602% (62.500%, 70.508%) | 100% | fail |
| causal page predictor | 0.432600 (0.380302, 0.490320) | 50.781% (46.484%, 55.078%) | 86.214% | fail |

The page oracle's failure is decisive for this naive contiguous layout: better
page prediction cannot make those particular 12-of-16 units sufficient. A useful
operator-level follow-up must first create meaningful units—for example through a
function-preserving neuron permutation that packs co-active channels, trained
expert blocks with a residual path, or a block/low-rank decomposition—then test
the unit oracle before training a router.

## Fused performance decision

The fused implementation was intentionally skipped because the individual-neuron
quality prerequisite failed. Benchmarking it would answer the wrong question:
even infinite kernel speed would not repair the observed quality loss. The next
performance gate belongs after a structured unit oracle passes the same held-out
thresholds.

Raw per-example metrics, failure cases, selected predictor checkpoints,
environment metadata, the resolved manifest, and the research log are in
[`artifacts/predictive_working_set/20261013-validated-final`](artifacts/predictive_working_set/20261013-validated-final).
