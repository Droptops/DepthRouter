# State-transition residency: first feasibility checkpoint (synthetic only)

This checkpoint is **not** a Qwen result and is **not** measured HBM traffic. It records a falsifiable failure mode discovered while designing the replay harness.

## Controlled synthetic sanity case
Two disjoint training and two disjoint held-out sequences alternate between stable page-demand regimes {1,2} and {3,4}. Capacity is 2 pages, assumed page payload is 128 bytes, and the transition detector is assigned an **illustrative** 16 bytes per token of metadata cost. Cold starts are excluded from the steady-state denominator.

| Policy | Steady logical faults (14 held-out tokens) | Modeled steady bytes/token |
| --- | ---: | ---: |
| Static training-frequency residency | 14 | 128.00 |
| Previous-token reuse | 8 | 73.14 |
| Causal transition-conditioned retention | 8 | 89.14 |
| Clairvoyant next-demand oracle (diagnostic) | 8 | 73.14 |

**Result:** transition-conditioned retention **does not beat** previous-token reuse here. With detector overhead it is worse. The synthetic test is a negative control, not a finding about model behavior.

## Implication
The proposed transition detector is only useful if it avoids enough *future* faults to pay for its own execution and metadata traffic. Detecting a transition *after* the first fault cannot retroactively avoid that fault. A transition policy must improve retention/eviction for subsequent tokens; prefetching pages that are inevitably required does not, by itself, reduce total bytes transferred.

The next Qwen experiment should therefore measure per-token *page-demand persistence*, conditional next-page probability and **incremental faults avoided over previous-token reuse**, before introducing a learned detector. This is a necessary feasibility test, not sufficient evidence for physical bandwidth savings.

## Evidence requirements
- Export per-sequence train/test page-demand traces without mixing prompt boundaries. Freeze page size, page mapping and held-out split.
- Never use current/future demand to decide residency *before* serving the current token. A policy may update its cache using current demand **after** serving the token.
- Keep cold-start, steady-state, assumed detector metadata, page payload, and static preloading separate.
- Report quality KL, top-1 and free-running agreement as **null** until actual model evaluation; logical replay has no quality measurement.
- Report physical off-chip bytes and latency as **null** until profiled on actual selective/offload execution. A stock dense GPU kernel with all weights resident is not a physical-cache savings test.
- Gate remains: KL <0.01, top-1 >=99%, plus **measured** quality-adjusted off-chip bytes/token improvement over previous-token reuse with overhead included.

No merge or promotion follows from this checkpoint.
