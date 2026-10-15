# Fresh recognition-study data

This manifest is a selected/reformatted derivative of pinned
[Databricks Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k/tree/bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a),
copyright 2023 Databricks, Inc., with source passages attributed to Wikipedia
editors and contributors. The data derivative is licensed under
[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/), not the code license.

Responses are omitted. Source instructions and contexts are unmodified apart
from their separator. All 2,304 contexts from the prior expert-construction
study and all 6,163 unique registered historical prompts are excluded.
New source-document and normalized instruction-family IDs are globally unique.
Prompt/instruction token Jaccard and source-document five-token-shingle screening
separate every new split and exclude earlier experiment source material.
This establishes lexical separation, not broad semantic-family or article-ID
independence; the source provides no article IDs.

Confirmation has equal matched and OOD task-category strata and is not fitting
data. Creative-writing and summarization categories occur only in confirmation
OOD or held-out OOD, never calibration/development. Every category stratum has
exactly one-eighth instructions meeting the frozen enterprise-topic keyword
screen. This is not a production-enterprise benchmark.

The frozen recipe is 128 confirmation contexts (64 matched plus 64 OOD),
768 calibration, 128 development, 512 held-out matched, and 192 held-out OOD:
1,728 total. These smaller targets were fixed before accessing model outputs.
The originally proposed 2,816 contexts could not satisfy the unchanged source,
exclusions and one-eighth enterprise quota. The pinned length-filtered source
has only 106 OOD enterprise rows total, 64 already used previously. Strict
freshness screening leaves 41 OOD enterprise rows and 212 matched enterprise
rows before allocating the new splits. No filter, source or quota was loosened.

Run `scripts/prepare_recognition_data.py` to reproduce the pinned manifest and
audit. The command refuses to replace different frozen outputs. It accepts
`--source-path` to reuse the prior ignored raw-source cache. No model outputs or
source answers are consulted for data selection. Exact provenance, quotas,
exclusions, limitations and hashes are in the adjacent audit/provenance files.
