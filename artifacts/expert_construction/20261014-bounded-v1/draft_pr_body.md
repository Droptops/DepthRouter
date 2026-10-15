## Result

One bounded final-MLP expert-construction cycle completed on the RTX 4070 SUPER
in 449.735 seconds (7.5 minutes), against a four-GPU-hour ceiling. This is audit
wall time, not an inference latency measurement.

**Partial capability success, no deployment promotion.** Four individual
partitions (B-17, B-29, C-17, C-29) passed the >=80% cheap coverage and >=5 pp
portfolio-advantage gates on development, matched held-out, and OOD held-out.
Their context-bootstrap lower bounds also passed. No construction family passed
across all three seeds. The development-selected C family missed OOD coverage
at seed 43: 78.620% (95% CI 76.735–80.587), versus the 80% criterion.

The **additional preregistered all-three-seed continuation rule** therefore
stopped router training. This is not a claim that every individual portfolio
failed or that cheap candidates do not exist.

Output-KL construction D did not improve the combined fidelity criterion under
this bounded recipe. Compared with hidden reconstruction C, paired seed-mean
acceptable-candidate coverage changed by:

| Split | D minus C | 95% context-bootstrap CI |
| --- | ---: | ---: |
| Matched held-out | -18.982 pp | [-20.527, -17.509] |
| OOD held-out | -21.037 pp | [-22.522, -19.618] |

Greedy agreement is reported separately: the cheap-only oracle's matched greedy
agreement improved slightly (+0.098 pp); OOD difference was unresolved. That does
not compensate for the full-distribution KL/coverage result.

No deployment router, runtime benchmark, physical-traffic counter attempt, or
free-generation task evaluation was run. Those costs and task-success results
remain unknown. Oracles are nondeployable; active-width fractions are arithmetic
proxies, not measured savings. Findings are limited to the tested partitions,
data, and 100-step fitting recipe, not a global partition optimum.

## Implementation and evidence

- Branch base: `ec696fddabc59c51292ba1e8f500c9000a50c79b`, PR #16.
  This draft is stacked on `research/output-gap-completion`; PRs #15 and #16
  are unchanged. No merge is requested or performed.
- Pinned Qwen/Qwen2.5-0.5B-Instruct revision
  `7ae557604adf67be50417f59c2c2f167def9a775`, BF16, original weights frozen.
- Eight equal 608-neuron experts; all 28 six-of-eight subsets; random,
  co-activation, hidden-reconstruction and full-vocabulary-KL constructions;
  seeds 17/29/43. No correction network.
- Paired C/D initialization, state/candidate schedules, capacities, optimizer,
  training budget and final-iterate rule. Actual packed hard forwards with a
  documented assignment-gradient surrogate, informed by DOT-MoE rather than
  claimed as its reproduction.
- Fresh 1,024/256/512/512 Dolly contexts with source/license provenance and
  lexical family separation; 6,163 historical prompts excluded. OOD reserves
  two task categories; enterprise-related keyword stratum is not a production
  benchmark. Article-level semantic independence is not established.
- Actual dense-prefix continuation positions 1/3/7/11, stopping at EOS:
  9,178 evaluated states, 48 matrices, **3,083,808 candidate measurements**.
- Preregistration before model access; fixed development choices sealed before
  held-out captures; exact full-model/cached numerical checks and measured
  original-order plus permuted all-eight controls.
- All raw scalar results, including failed arms, are versioned at
  `artifacts/expert_construction/20261014-bounded-v1/`. Large deterministic
  activation caches stay local; their hashes and reconstruction inputs remain.
- `EXPERT_CONSTRUCTION_REPORT.md` is the concise report; the run's `REPORT.md`
  has every arm/seed/split, task/margin strata, numerical controls and raw hashes.
  `promotion_decision.json` and `integrity_audit.json` record the decisions and
  independent evidence verification.

## Validation and reproduction

Repository-wide Ruff and all 157 tests passed in the pinned container after final
report refinements. `validation.json` records the verification count.
The offline integrity audit passed all 17 source snapshots, 15 sealed files,
12 balanced partitions, paired construction schedules, 48 matrices and 348
complete-model replay checks (16 states per check).

```powershell
# Uses the existing local image and Hugging Face cache; a new run ID is mandatory.
./scripts/run_expert_construction.ps1 -SkipBuild -RunId reproduction-001
```

```bash
# Offline evidence checks: no model loading or training.
PYTHONPATH=src python scripts/verify_expert_audit.py \
  --run-dir artifacts/expert_construction/20261014-bounded-v1
PYTHONPATH=src python experiments/expert_construction_report.py \
  --run-dir artifacts/expert_construction/20261014-bounded-v1
python -m ruff check .
PYTHONPATH=src python -m pytest
```

For a checkout that converts source line endings, the verifier's explicit
`--allow-current-newline-only` option reports that distinction; saved source
snapshots must still match their exact byte hashes.
