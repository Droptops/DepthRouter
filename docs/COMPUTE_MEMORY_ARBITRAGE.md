# Compute-for-memory arbitrage

The most important decision rule in this branch is not an early-exit threshold.

It is a break-even condition.

Let the calibrated future operator working set before a resident solver spin be

```text
W_k = bytes(C_k)
```

and let the same quantity after the spin be `W_(k+1)`. Let `B_spin` be the
measured bytes moved to execute that resident spin.

Define

```text
MROI_k = E[W_k - W_(k+1) | state_k] / B_spin
```

The spin is bandwidth-profitable when

```text
MROI_k > 1.
```

This is the first-order **compute-for-memory arbitrage** condition: spend cheap
resident computation now only when it is expected to retire more expensive
future memory traffic than it consumes.

The exact runtime policy must also include latency, overlap, cache eviction and
quality effects, but those do not change the falsification test. If measured
future bytes retired do not exceed bytes spent, the spin is overhead.

## Address-information view

For equal-size pages,

```text
W_epsilon = page_bytes * exp(H0_epsilon)
```

where `H0_epsilon` is smooth Renyi-0 entropy of the calibrated page-demand
support.

Therefore a spin that reduces smooth address entropy by `Delta H` changes the
idealized working set multiplicatively:

```text
W_after / W_before = exp(-Delta H).
```

One nat of useful address information divides the candidate working set by
approximately `e`. One bit divides it by two.

The useful yield of a spin is therefore

```text
address_information_yield =
    Delta H0_epsilon / measured_spin_bytes
```

provided that lower address entropy actually produces fewer measured transfers.

## This changes what "reasoning" can buy

A recurrent step can be economically useful even when it barely moves the final
logits.

If it sharply reduces uncertainty about which cold operator pages will be
needed, it can pay for itself by preventing future memory movement.

Conversely, a step that improves logits but does not justify its memory traffic
may be a bad systems action.

This separates two values of computation:

```text
answer value  = predictive loss reduced now
address value = future bytes made unnecessary
```

The runtime should eventually price both.

## Decisive profiler test

On target hardware, record for every spin:

- calibrated operator-set bytes before the spin;
- calibrated operator-set bytes after the spin;
- profiler-measured bytes moved by the spin;
- eventual cold-page transfers;
- cross-entropy / task quality.

Then report

```text
realized_MROI =
    measured cold bytes avoided / measured spin bytes
```

not a parameter-count estimate.

A sustained `realized_MROI > 1` at preserved quality is the point where this
stops being an interesting cache metaphor and becomes a systems result.
