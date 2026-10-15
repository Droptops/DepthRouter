"""Freeze fresh prompt-only confirmation/training/evaluation data before model use."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from depth_router import expert_audit_data as source
from depth_router.recognition_data import (
    SPLIT_SIZES,
    audit_enterprise_source_capacity,
    load_recognition_exclusions,
    make_recognition_manifest,
    validate_recognition_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or args.repo / "data/expert_recognition"
    source_path = args.source_path or output / "databricks-dolly-15k.jsonl"
    if not source_path.exists():
        with urlopen(source.SOURCE_URL, timeout=60) as response:
            raw = response.read(20_000_001)
        if len(raw) > 20_000_000 or hashlib.sha256(raw).hexdigest() != source.SOURCE_SHA256:
            raise ValueError("source download exceeds size limit or fails pinned SHA256")
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_bytes(raw)
    historical, prior, exclusions = load_recognition_exclusions(args.repo)
    rows = make_recognition_manifest(source_path, historical_prompts=historical,
                                    prior_expert_rows=prior)
    audit = validate_recognition_manifest(rows, historical_prompts=historical,
                                         prior_expert_rows=prior)
    audit["exclusion_sources"] = exclusions
    capacity = audit_enterprise_source_capacity(source_path, historical_prompts=historical,
                                               prior_expert_rows=prior)
    manifest_bytes = ("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n").encode()
    audit["manifest_file_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    provenance = {
        "dataset": source.SOURCE_ID, "revision": source.SOURCE_REVISION,
        "url": source.SOURCE_URL, "sha256": source.SOURCE_SHA256,
        "license": source.SOURCE_LICENSE,
        "attribution": "Copyright (2023) Databricks, Inc.; Wikipedia editors and contributors",
        "dataset_card": f"https://huggingface.co/datasets/{source.SOURCE_ID}/blob/"
                        f"{source.SOURCE_REVISION}/README.md",
        "transformations": "Select human-written instruction/context; omit responses; exclude all"
            " earlier experiment contexts and lexical families; freeze new disjoint splits with"
            " length, task-category, source-document, lexical and enterprise-keyword constraints.",
        "model_outputs_accessed": False, "frozen_split_sizes": SPLIT_SIZES,
    }
    artifacts = {
        "evaluation_manifest.jsonl": manifest_bytes,
        "data_audit.json": (json.dumps(audit, indent=2, sort_keys=True) + "\n").encode(),
        "source_provenance.json": (json.dumps(provenance, indent=2, sort_keys=True) + "\n").encode(),
        "source_capacity_audit.json": (json.dumps(capacity, indent=2, sort_keys=True) + "\n").encode(),
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, raw in artifacts.items():
        path = output / name
        if path.exists() and path.read_bytes() != raw:
            raise ValueError(f"refusing to replace different frozen data: {path}")
    for name, raw in artifacts.items():
        (output / name).write_bytes(raw)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
