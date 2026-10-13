# Predictive working-set evaluation manifest

`make_evaluation_manifest()` in `src/depth_router/predictive_working_set.py`
deterministically generates the preregistered evaluation set:

- 256 calibration prompts used to fit predictor parameters;
- 128 development prompts used only to select feature form and predictor rank;
- 512 held-out prompts opened once after model selection;
- four equally represented task families: classification, extraction, numerical,
  and short generation.

Each split uses distinct prompt templates and task variants. IDs, prompts, and
template families are validated for uniqueness and cross-split isolation. The
resolved JSONL manifest is written into every run artifact directory before any
model inference.
