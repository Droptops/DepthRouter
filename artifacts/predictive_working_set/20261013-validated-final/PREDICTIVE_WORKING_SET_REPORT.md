# Predictive working-set feasibility

**Decision: INDIVIDUAL_NEURON_FAILED_OPERATOR_PAGE_PIVOT_ALSO_FAILED**

The predictor was selected using calibration/development data before the 512-prompt held-out split was captured. The quality gate required mean KL < 0.01 and dense top-1 agreement >= 99%.

## Individual-neuron result

| Method | Held-out prompts | Mean KL | Top-1 agreement | Oracle-set recall | Pass |
|---|---:|---:|---:|---:|---:|
| Full-width reconstruction control | 512 | 0.000000 | 100.000% | n/a | True |
| Non-deployable oracle 75% | 512 | 0.000983 | 98.633% | 100.000% | False |
| Causal predictor 75% | 512 | 0.103771 | 68.945% | 78.892% | False |

## Structured operator-page pivot

The failed neuron gate triggered a 16-page contiguous layout; exactly 12 pages (75% of 4,864 neurons) execute per token.

| Method | Mean KL | Top-1 agreement | Oracle-page recall | Pass |
|---|---:|---:|---:|---:|
| Page oracle | 0.208599 | 66.602% | 100.000% | False |
| Causal page predictor | 0.432600 | 50.781% | 86.214% | False |

## Fused performance gate

Skipped: preregistered individual-neuron quality gate failed

## Claim boundary

Oracle masks inspect the current dense intermediate activation and are not deployable. The learned predictor sees only the current causal MLP input (and, for the transition variant, its previous-token difference). No optical timing is inferred in this test. Quality uses dense-shape counterfactual masking to isolate selection error from packed-kernel roundoff.
