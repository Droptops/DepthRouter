# Memory-State Solver Formulation

DepthRouter should not be described as a transformer with a stopping rule bolted
onto recurrence. The useful abstraction is a bandwidth-constrained solver whose
state includes the memory hierarchy.

This document intentionally adds no new mechanism. It changes the state variable,
the control problem, and the falsification test.

## 1. State

Use

```text
s_k = (h_k, c_k, p_k)
```

where:

- `h_k` is the residual / solver state;
- `c_k` is the cache-placement state, including which cold operator slices are
  resident and whether the one-shot KV write has already occurred;
- `p_k` is the current next-token logit distribution.

The point of including `c_k` is that byte cost should be a property of a state
transition, not a penalty computed after the model has already chosen a path that
moved the bytes.

If two executions with the same `(h_k, p_k)` incur different memory cost because
their residency differs, they are not the same state.

## 2. Actions

At each step the policy selects one move from a finite set:

```text
spin
fault(j)
write
halt
```

Semantics:

- `spin`: advance the resident solver core without fetching a cold operator slice;
- `fault(j)`: fetch cold slice `j`, update `c_k`, then advance the solver;
- `write`: emit the KV update; this is legal at most once per token position;
- `halt`: commit `p_k`.

Depth is therefore not a configured layer count. It is the number of `spin`
moves the policy purchases before `halt`.

## 3. Transition and cost

Let

```text
s_(k+1) = T(s_k, a_k)
b_k     = b(s_k, a_k)
```

where `b_k` is the measured byte cost on the target memory hierarchy.

The offline training/evaluation objective should use task loss and measured
traffic:

```text
J(pi) = E[L_task(p_T)] + lambda * E[sum_k b_k]
```

Latency can be reported separately or included as another measured transition
cost. Do not substitute parameter count for bytes moved.

At inference, labels are unavailable. The cheap observable for whether another
step changes the prediction is logit movement:

```text
q_k = KL(p_(k+1) || p_k)
```

The decision is a value-of-information problem: buy another move only when the
expected predictive return exceeds its measured byte price.

## 4. Coalesced misses are the routing target

The policy is not only per-token. Scheduling should group tokens by predicted
`fault(j)`.

If thirty tokens need the same cold slice, the desired execution is one coalesced
transfer followed by work on those thirty tokens, not thirty independent faults.

This means the systems objective is:

```text
minimize unique/coalesced cold transfers
```

subject to the quality target.

Per-token adaptive depth that ignores coalescing can reduce arithmetic work while
losing on bandwidth.

## 5. Hot core and cold program memory

The resident component should be treated as a solver/interpreter rather than as
a miniature full transformer.

The working interpretation is:

```text
resident core  = interpreter / state transition engine
h_k            = program counter + scratch state
cold slices    = program / knowledge memory
spin           = advance resident computation
fault(j)       = load the next required program fragment
```

This is a hypothesis to falsify, not a claim that hidden states literally encode
a conventional instruction pointer.

## 6. Belief state: the vector is an anytime posterior, not just a hidden state

There are two different notions of time and they should not be conflated:

```text
t = user / token / interaction time
k = internal solver iteration
```

Let `z_t` be the latent task or trajectory state at interaction time `t`: the
"where are we on the timeline?" variable. The observable history is `x_(1:t)`.

The solver should be interpreted as maintaining an approximate belief state

```text
q_(t,k)(z) ~= P(z_t = z | x_(1:t))
```

and spending compute to refine that approximation.

A spin does not create new external evidence. It performs another inference step
over the same evidence:

```text
q_(t,k+1) = Phi(q_(t,k), x_(1:t), c_(t,k))
```

The output distribution is the posterior predictive:

```text
p_(t,k)(y) = sum_z P(y | z) q_(t,k)(z)
```

This makes the recurring vector interpretable as an **anytime probability
trajectory**. More FLOPs are useful only if they move the approximate posterior
toward a better predictive distribution. They are not assumed to help
monotonically; that has to be measured.

The previous `h_k` remains the implementation state, but conceptually it should
encode only what is needed to update the belief: a compact program counter,
scratchpad, and sufficient statistics for `q`.

### Cache the posterior support

The residency state should cache the operator / knowledge slices most likely to
be needed under the current belief.

For cold slice `j`, let

```text
r_j = P(slice j will be needed | q_(t,k))
```

and let `g_j` be the expected predictive gain if that slice is available.

The cache set is a capacity-constrained placement problem:

```text
C* = argmax_C sum_(j in C) r_j * g_j * reuse_j
     subject to sum_(j in C) size_j <= cache_capacity
```

So "cache the top percentage" should not mean top posterior probability alone.
It should mean top **expected value per resident byte**, including batch reuse.

A useful score is

```text
score_j = r_j * E[Delta CE_j] * reuse_j / size_j
```

or the corresponding logit-information estimate when labels are unavailable.

This is where repeated user behavior becomes useful: if many requests collapse
onto a small set of latent trajectories, the posterior concentrates and the hot
working set becomes small. That is an empirical hypothesis, not an assumption.

### Halt and fault are both posterior decisions

The same belief state determines both whether to continue and what to load.

A `spin` is worth buying when the expected change in the predictive
distribution per resident byte is high enough.

A `fault(j)` is worth buying when the expected gain from resolving probability
mass associated with slice `j` exceeds the transfer cost.

A `halt` is optimal when no available spin or fault has positive enough value:

```text
max_a E[ predictive_gain(a) | q_(t,k), c_(t,k) ] / bytes(a) <= eta
```

The observable proxy can be the KL movement of the output distribution:

```text
Delta I_k = KL(p_(t,k+1) || p_(t,k))
```

This turns adaptive depth into approximate Bayesian inference under a memory
budget. Depth is simply how many inference refinements were worth purchasing.

### Batch by posterior fault

For a batch of tokens, the scheduler should group tokens by their highest-value
predicted cold slice:

```text
token -> q_(t,k) -> predicted fault j -> coalesced transfer
```

The unit of optimization is therefore not "passes per token." It is
**posterior mass resolved per coalesced byte transfer**.

This is the formulation to test before adding any new architecture.

## 7. The useful mathematical import from anomaly / inflow ideas

There is no claim that DepthRouter is a topological field theory. The useful
structural idea is **global consistency of a local description**.

For a trajectory

```text
gamma = (s_0, a_0, s_1, ..., a_(T-1), s_T)
```

define the accounting residual

```text
A(gamma) = B_hw(gamma) - sum_k b(s_k, a_k)
```

where `B_hw` is the byte count reported by the hardware profiler.

If `A(gamma)` is systematically non-zero beyond profiler noise, the local state
is incomplete. In practice this means `c_k` is missing a variable that changes
traffic, such as a residency, eviction, coalescing, or graph-capture effect.

This is the computational analogue of an anomaly test: a locally specified model
fails to compose into the observed global quantity. The fix is not another loss
term. The fix is to enlarge the state until the accounting discrepancy vanishes.

The resulting criterion is:

```text
memory state is sufficient  <=>  A(gamma) ~= 0 across held-out schedules
```

This is a state-identification test, not a physics analogy presented as evidence.

## 8. One decisive experiment

Use one pinned mid-stack block on one small model and the same token stream for
all schedules.

Measure four schedules:

1. stock forward;
2. naive recurrence, K=4;
3. K=4 with KV written once and later spins read-only;
4. schedule 3 plus a cold MLP slice that is faulted only when the previous
   spin's logit movement exceeds a fixed threshold.

Measure with a hardware profiler:

- HBM / memory-controller bytes read and written;
- L2 hit rate;
- cross-entropy in nats;
- wall-clock decode latency as a secondary metric.

The primary metric is incremental cross-entropy improvement per incremental byte
after the first useful pass.

For schedule `sigma`:

```text
eta_sigma =
    (CE_after_pass1 - CE_sigma)
    / max(B_sigma - B_after_pass1, epsilon)
```

Units are nats / byte.

The residency thesis survives only if schedule 4 produces a better
quality-per-byte frontier than schedule 3 after including the gate's own traffic.

If it does not, stop. The cache-aware residency story has been falsified on that
hardware/model pair.

If it does, the next training target is the policy over
`{spin, fault(j), write, halt}`, with coalesced faults. Do not add another
adapter, accelerator, or residual penalty before that measurement.

## 9. Claim boundary

A successful result would establish only this:

> On the tested model and hardware, making placement part of the solver state
> improved predictive gain per byte moved.

It would not establish that all looped transformers benefit, that L2 residency
generalizes across devices, or that the interpreter/program-memory view is the
unique explanation.
