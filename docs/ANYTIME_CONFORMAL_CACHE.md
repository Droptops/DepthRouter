# Anytime Conformal Cache

The raw "cache the top posterior mass" idea is not safe enough for recurrent
inference.

A recurrent model can become **more confident and more wrong** at an intermediate
spin. Lower entropy is therefore not evidence that the physical working set can
safely shrink.

The cache should be a calibrated prediction set over future computation.

## 1. Page prediction as a statistical set

Let (J_t) be the cold page that an exact/full computation would require for a
token or solver state.

At spin (k), the resident solver emits scores over pages:

```text
p_k(j) = predicted probability that page j will be needed
```

Instead of taking a fixed top-k or top-95%-mass set, use a held-out calibration
split to construct a conformal set:

```text
C_k(x) = calibrated candidate pages at spin k
```

with target miss risk `alpha`.

Under the split-conformal exchangeability assumption:

```text
P(J not in C_k) <= alpha
```

up to the standard finite-sample conformal guarantee.

The physical cache is then a materialized uncertainty set, not a heuristic list
of likely weights.

## 2. Adaptive depth requires simultaneous validity

Marginal coverage at each spin is not enough if the runtime is allowed to inspect
the sets and stop adaptively.

For a finite horizon of K spins, the first implementation uses a conservative
Bonferroni allocation:

```text
alpha_k = alpha / K
```

and calibrates one conformal set per spin.

By the union bound, the probability that the true page is excluded at *any* of
the K spins is at most alpha, provided the conformal assumptions hold.

The runtime can therefore intersect the sets:

```text
C'_k = intersection(C_1, ..., C_k)
```

so the candidate cache is monotone:

```text
C'_1 superset C'_2 superset ... superset C'_K
```

A spin is useful when it safely removes future memory obligations.

This is the systems interpretation of an anytime confidence sequence.

## 3. New hardware-facing quantity: conformal working set

Define the conformal working-set size:

```text
W_alpha(k) = E[bytes(C'_k)]
```

This is more operational than parameter count.

It asks:

> How many bytes must remain physically available after k inference refinements
> to keep the probability of evicting the page we will actually need below alpha?

The relevant model curve is therefore not only loss versus FLOPs.

It is:

```text
quality
vs
compute already spent
vs
W_alpha(k)
vs
measured page-transfer bytes
```

A useful spin can pay for itself twice: it can improve the prediction and reduce
the size of the future resident working set.

## 4. Memory-distortion frontier

For a page-set policy C, define:

```text
miss_risk(C) = P(J not in C)
memory(C)    = E[bytes(C)]
distortion(C)= predictive regret from missing/approximating cold work
```

The systems problem becomes a physical rate-distortion problem:

```text
min memory(C)
subject to miss_risk(C) <= alpha
           distortion(C) <= D
```

This connects causal-state compression to a concrete memory hierarchy.

The cache does not need to preserve every latent distinction. It needs to
preserve the distinctions whose loss changes the future computation or output.

## 5. Co-posterior page packing and conformal paging solve different problems

Co-posterior affinity learns **where states should live physically**:

```text
A_ij = E[q_i q_j]
```

Conformal calibration learns **how much of that physical state must be resident**
for a chosen miss risk.

These are complementary:

1. pack hypotheses that are often jointly plausible onto the same page;
2. calibrate the smallest page set that can be trusted;
3. group tokens by predicted page fault;
4. load each required page once per scheduling window.

The compound metric is bytes at fixed coverage, not cache hit rate alone.

## 6. Important dependence caveat

Language-model traffic is not i.i.d. Tokens within a conversation and repeated
requests from the same user are dependent.

A token-random calibration split can therefore produce a misleading conformal
guarantee.

Real experiments must split calibration and test data at the **conversation or
session level**, and should report both:

- marginal page coverage;
- worst-group / session-stratified coverage when enough data exists.

If dependence invalidates the conformal assumptions, the guarantee must be
weakened rather than silently retained.

## 7. First experiment

`experiments/anytime_conformal_cache.py` uses the recurrent pointer-chasing
model as a controlled proxy.

It compares:

- a model trained only for its final spin;
- a model trained so per-example predictive loss is discouraged from worsening
  across spins.

For every spin it reports:

- cross-entropy;
- accuracy;
- posterior entropy;
- uncalibrated top-mass cache size and coverage;
- marginal conformal cache size and coverage;
- finite-horizon nested anytime cache size and coverage.

This experiment does not establish a hardware speedup. Its purpose is narrower:

> Determine whether recurrent inference can produce a probability trajectory
> whose statistically defensible physical working set shrinks with compute.

Only after that passes should page identities be replaced with real cold operator
slices and measured on the target GPU.
