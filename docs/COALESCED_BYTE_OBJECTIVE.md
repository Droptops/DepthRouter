# Coalesced unique-byte objective

The real Qwen gate falsified the strongest version of the static paging thesis:
a modern dense SwiGLU block did not expose enough strict output sparsity merely
by reordering neurons. At a 1% relative MLP-output error target, essentially all
pages remained necessary in the first Qwen gate.

That changes the next question from:

> Can a runtime predictor discover sparsity that the dense model never learned?

to:

> Can training make behaviorally interchangeable computation share physical
> pages so the union of bytes requested by a batch becomes small?

## Physical batch cost

Let q_ij be the probability token i requests physical page j, b_j the page
payload bytes, and c_j in [0, 1] the residency probability.

The probability that page j must be faulted by at least one token is:

```text
P_fault(j) = 1 - product_i (1 - q_ij)
```

so expected new coalesced traffic is:

```text
C_union(B, c)
  = sum_j b_j (1 - c_j) [1 - product_i (1 - q_ij)]
```

Compare that with a token-separable byte objective:

```text
C_independent(B, c)
  = sum_i sum_j q_ij b_j (1 - c_j)
```

The independent objective rewards sparsity per token. It is blind to whether
two tokens request the same page. The union objective rewards physical agreement
across the batch.

## Computational sparsity is not I/O sparsity

Two routers can each request one page per token and have identical token-level
compute while producing radically different transfer traffic:

```text
64 tokens -> 64 different pages -> 64 transfers
64 tokens ->  1 shared page     ->  1 transfer
```

The expected-union formula is differentiable with respect to q, so coalescing
pressure can be present during training instead of patched onto inference.

## Controlled gate

`experiments/coalesced_union_training.py` isolates the optimization geometry.

Each semantic family has two exactly behavior-equivalent implementations:

- one page shared by every family;
- one private page for that family.

The router starts by preferring private pages. A per-token byte objective has no
economic reason to move from private to shared because both cost one page per
token. The union-byte objective does.

The gate passes when both objectives preserve the task while union-byte training
materially reduces unique pages and coalesced bytes per mixed batch.

Passing this gate proves only that the loss has the intended geometry.

## Next gate

If the controlled gate passes, upcycle a small dense SwiGLU teacher into a
page-gated student:

```text
L = L_distill
    + lambda * C_union(B, c)
    + calibration / stability terms
```

and compare against an otherwise identical student trained with
`C_independent`.

The decisive comparison is at matched teacher distortion:

```text
unique physical pages per batch
coalesced payload bytes
measured HBM bytes
```

Only the final measurement establishes a systems result.
