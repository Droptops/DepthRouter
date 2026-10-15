# Data provenance and licenses

`evaluation_manifest.jsonl` is a selected/reformatted derivative of
[Databricks Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k/tree/bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a),
copyright 2023 Databricks, Inc., with source passages attributed to Wikipedia
editors and contributors. The dataset and this derivative are licensed under
[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/), not the repository's
source-code license. Transformations: exclude historical/duplicate/similar prompts,
select instruction/context rows, discard responses, add provenance identifiers and
deterministic split/task/stratum fields. See `source_provenance.json` and
`data_audit.json` for pinned revisions, hashes, exact filters, and limitations.

`historical_alpaca_prompts.jsonl` contains exclusion-only prompts reconstructed
from [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca/tree/761dc5bfbdeeffa89b8bff5d038781a4055f796a),
attributed to Stanford Alpaca / tatsu-lab, under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). It is not training
or evaluation data for this cycle; answers are omitted. Its provenance file
documents historical split reconstruction and verified prompt hashes.

`scripts/prepare_expert_audit_data.py` downloads the exact public source versions,
reconstructs historical exclusions, and reproduces the manifest. Raw source
caches are ignored by Git. Exact source documents and normalized instruction
families are separated, with additional lexical-similarity exclusions. The source
does not include article IDs; article-level or broad semantic-family independence
is not guaranteed.

Run-level state files retain source-derived prompt prefix token IDs and dense
model continuation IDs for reproduction. Dataset licensing/attribution continues
to apply to the source-derived prompt material. Numerical candidate measurements
are diagnostic results, not task-success labels.
