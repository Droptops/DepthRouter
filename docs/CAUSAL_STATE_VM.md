# Causal-State Virtual Memory

This experiment treats recurrent inference as approximate belief refinement over
future-equivalence classes, with physical memory placement derived from that
belief.

## 1. Predictive state

Two histories are considered equivalent when they induce approximately the same
distribution over futures:

```text
x_<=t ~ x'_<=t  iff  P(X_>t | x_<=t) ~= P(X_>t | x'_<=t)
```

The solver maintains an approximate posterior over these predictive states:

```text
q_(t,k)(z) ~= P(Z_t = z | x_<=t)
```

where `t` is interaction/token time and `k` is internal solver time.

A recurrent spin is useful only insofar as it improves the posterior predictive;
it does not create new external evidence.

## 2. The cache is the materialized posterior

Let each predictive state live on a physical page. The posterior induces a page
mass distribution. For fixed-size pages, define the predictive working set

```text
W_epsilon(q) =
    minimum bytes of pages needed to cover at least (1 - epsilon)
    posterior mass
```

This is the quantity the project should try to drive down while preserving
prediction quality.

The important placement objective is not semantic similarity. It is
**co-posterior affinity**:

```text
A_ij = E[q_i q_j]
```

States that are simultaneously plausible are expensive when spread across many
pages. Packing high-affinity states together makes uncertainty itself cheaper.

The repository implements a first deterministic greedy partitioner using this
matrix.

### Smooth address entropy

For equal-size physical pages, define

```text
N_epsilon(q) =
    minimum number of pages covering at least (1 - epsilon) posterior mass

H0_epsilon(q) = log N_epsilon(q)
```

`H0_epsilon` is the smooth Renyi-0 entropy of the page-demand support.

The physical working set is therefore exactly

```text
W_epsilon(q) = page_bytes * exp(H0_epsilon(q))
```

when natural logarithms are used.

This exposes a useful compute-memory duality: reducing address uncertainty by
one nat divides the required fixed-page working set by approximately `e`; one
bit divides it by two.

The target of a resident solver spin is therefore not generic hidden-state
convergence. It is useful **address information**: information that collapses the
set of cold computations that could still be required.

This suggests a hardware-facing yield metric for the eventual profiler run:

```text
address_information_yield =
    reduction in smooth address entropy
    / measured bytes spent on the spin
```

The project should only promote this metric if a lower address entropy actually
predicts fewer measured cold transfers on held-out real traces.

## 3. Why this is different from ordinary caching

Ordinary cache policy asks which already-defined pages should remain resident.

This proposal also changes the page layout:

```text
learned posterior traces
        ->
co-posterior affinity
        ->
physical page partition
        ->
smaller posterior working set
```

In a future trained system, model learning and memory layout could be optimized
together.

## 4. Fault scheduling

Token-major execution can destroy the gain even with a good layout. Requests
should be grouped by predicted semantic page fault.

The first benchmark therefore reports both:

- token-major LRU page faults;
- fault-grouped page loads in the same scheduling window.

The compound systems target is:

```text
posterior concentration
    -> smaller predictive working set
    -> fewer semantic page faults
    -> coalesced transfers
    -> fewer bytes moved
```

## 5. First falsification

`experiments/belief_page_packing.py` learns the physical layout from training
posterior traces and evaluates it on held-out posterior traces.

It compares:

1. sequential physical layout;
2. random physical layout;
3. posterior-affinity layout learned from training traces;
4. oracle family layout, available only in the synthetic generator.

The learned layout has not succeeded merely because it wins on synthetic data.
The synthetic test only establishes that the objective and scheduler can recover
known hidden structure without being told the hidden family labels.

The next meaningful gate is real posterior traces from an actual recurrent model.
If held-out model traces do not show that lower posterior entropy predicts a
smaller `W_epsilon`, then the causal-state caching thesis should be weakened or
discarded.

## 6. Strong claim boundary

No result in the synthetic benchmark establishes a new language-model scaling
law or a hardware speedup.

A result becomes systems-relevant only when a real model shows:

```text
posterior entropy down
    AND predictive working set down
    AND measured transferred bytes down
    AND quality preserved
```

on the target hardware.
