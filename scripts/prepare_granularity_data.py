"""Freeze a byte-preserving derivative of the prior run's unopened partitions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from depth_router.granularity_data import (
    BASE_COMMIT,
    PRIOR_MANIFEST,
    PRIOR_RUN,
    derive_granularity_manifest,
    serialized_manifest_bytes,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output_dir or repo / "data/expert_granularity"
    rows, audit, quarantine = derive_granularity_manifest(repo)
    inherited_provenance = json.loads((repo / "data/expert_recognition/source_provenance.json").read_text())
    provenance = {
        **inherited_provenance, "base_commit": BASE_COMMIT,
        "derived_from_manifest": PRIOR_MANIFEST, "prior_execution": PRIOR_RUN,
        "transformations": "Remove all 128 prior confirmation rows; retain all 1600 unopened rows"
            " byte-for-byte in inherited order, without new sampling or prompt/provenance changes.",
        "frozen_split_sizes": audit["split_sizes"], "quarantined_confirmation_contexts": 128,
        "responses_used": False,
        "model_outputs_accessed": True,
        "model_output_access_scope": "Read-only prior confirmation stop/identity metadata for lineage;"
            " no retained-context outputs accessed and no model inference.",
        "freshness_scope": audit["freshness_scope"],
        "limitations": "Category/lexical holdout is not semantic-family or source-article independence;"
            " enterprise is a 12.5% instruction-keyword topic screen, not a production benchmark.",
    }
    artifacts = {
        "evaluation_manifest.jsonl": serialized_manifest_bytes(repo, rows),
        "data_audit.json": (json.dumps(audit, indent=2, sort_keys=True) + "\n").encode(),
        "source_provenance.json": (json.dumps(provenance, indent=2, sort_keys=True) + "\n").encode(),
        "quarantined_confirmation.json": (json.dumps(quarantine, indent=2, sort_keys=True) + "\n").encode(),
    }
    for name, raw in artifacts.items():
        path = output / name
        if path.exists() and path.read_bytes() != raw:
            raise ValueError(f"refusing to overwrite different frozen data: {path}")
    output.mkdir(parents=True, exist_ok=True)
    for name, raw in artifacts.items():
        (output / name).write_bytes(raw)
    print(json.dumps({"manifest_file_sha256": audit["manifest_file_sha256"],
                      "canonical_manifest_sha256": audit["manifest_sha256"],
                      "split_sizes": audit["split_sizes"], "rows": len(rows),
                      "quarantined_ids_sha256": quarantine["ids_sha256"],
                      "freshness_scope": audit["freshness_scope"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
