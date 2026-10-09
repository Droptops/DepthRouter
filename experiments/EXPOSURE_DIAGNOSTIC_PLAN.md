# Temporal address exposure diagnostic

## Hypothesis

The learned address head fails because teacher-forced previous oracle masks at training time differ from previous predicted masks at inference time. Test this separately from low sample complexity.

## Controlled comparison

On held-out prompts with the same dense-generated prefixes and same trained address head, compare: (1) student-history masks, (2) oracle-history masks, (3) zero-history masks, and (4) oracle-selected sparse MLP. Report first-step and all-step mean KL, top-1 agreement, oracle-set recall, and complete-trajectory exact agreement. Keep fractions and token counts identical.

## Gate

Deployable student-history arm must satisfy mean KL < 0.01 nats, top-1 >= 99%, and measured metadata plus newly transferred cold payload <= 25% of dense MLP bytes. Oracle-history is a diagnostic upper bound, not a deployable result. Byte estimates assuming ideal neuron-granular transfers are not physical-cache measurements. Require profiler-backed traffic and latency before making hardware claims.

## Decision

If oracle-history materially improves quality over student-history, prioritize scheduled sampling / on-policy history training. If both fail similarly, prioritize sample complexity, feature adequacy, and temporal target entropy. If oracle-selected quality fails, revisit sparse working-set oracle and budget before training a router. Do not claim breakthrough without all three simultaneous gates on held-out free-running sparse trajectories.
