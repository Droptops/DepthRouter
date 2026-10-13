# Dataset and split manifest

`photonic_parallel_manifest.jsonl` is a locally generated, nonsensitive workload
for a systems falsification experiment. It has 24 records: eight calibration,
eight development, and eight held-out records, balanced across classification,
structured extraction, numerical reasoning, and short-form generation.

The splits use different instructions, entities, numbers, and surface forms.
Calibration records may define static, task-conditioned, or state-conditioned
selectors. Development records choose among preregistered variants. Held-out
records are used only for final metrics. The oracle diagnostic may inspect a
held-out activation, but it is always excluded from deployable comparisons.

The answers support teacher-forced response-token cross-entropy. Free-running
quality is also scored against both the manifest answer and the dense model's
deterministic generation. This small synthetic suite is a controlled smoke gate,
not a claim of broad language-model capability.
