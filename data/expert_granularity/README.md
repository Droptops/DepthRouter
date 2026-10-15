# Unopened-partition derivative for the granularity audit

This dataset retains exactly the previously unopened partitions from the frozen
recognition manifest: calibration 768, development 128, held-out matched 512 and
held-out OOD 192 (1,600 contexts total). IDs, prompt strings, split assignments,
source metadata, quotas, original row order and serialized JSONL line bytes are
unchanged. There is no new sampling, answer use or source download.

The entire failed 128-context confirmation split is quarantined. Its IDs and
prompt hashes, plus hashes of the inherited confirmation rows, are recorded in
`quarantined_confirmation.json`. None is part of the new evaluation manifest.

The derivative pins commit `1be53fefd37c344dbc4cdc44bb708e3e226ae0e9` and the
specific prior run `artifacts/expert_recognition/20261014-recognition-v1`.
The preparation command verifies its failed terminal confirmation, frozen C-29
partition/fixed candidate 8/all 28-candidate audit, registered source snapshots,
and confirmation-only captures, matrix, state metadata and event log. It does not
search unrelated or future run folders. The prior run's no-training/no-runtime
negative result remains unchanged.

Freshness means **the pinned prior run produced no model outputs on these retained
contexts**. It does not establish that nobody has ever used these source texts,
or prove semantic distribution independence. The command also reruns the strict
full 1,728-row recognition exclusion checks, then validates the filtered 1,600-row
four-split manifest against historical prompts, previous expert contexts, and
all quarantined confirmation prompts. Source-document/instruction-family and
lexical separation from confirmation are inherited from the fully checked split.

Source and derivative license: [Databricks Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k/tree/bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a),
copyright 2023 Databricks, Inc., with passages attributed to Wikipedia editors and
contributors; [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/), not the
repository code license. Every split retains a 12.5% enterprise-instruction-keyword
stratum, not a production-enterprise benchmark. Creative-writing/summarization
are reserved OOD task categories; article-level and semantic-family independence
are not established. OOD enterprise examples are creative-writing-heavy.

Reproduce with `python scripts/prepare_granularity_data.py`. The command is CPU-only,
uses no language model, and refuses to replace different frozen outputs. Exact
lineage hashes, inherited source metadata, validation results and scope limitations
are in `data_audit.json` and `source_provenance.json`.
