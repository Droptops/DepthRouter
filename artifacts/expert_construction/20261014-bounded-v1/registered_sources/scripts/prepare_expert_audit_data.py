"""Download and freeze prompt-only data before the expert audit loads a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from depth_router.expert_audit_data import (
    ALPACA_SOURCE_SHA256,
    ALPACA_SOURCE_URL,
    SOURCE_ID,
    SOURCE_LICENSE,
    SOURCE_REVISION,
    SOURCE_SHA256,
    SOURCE_URL,
    load_historical_prompts,
    make_expert_manifest,
    make_historical_alpaca_manifest,
    validate_expert_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source-path", type=Path)
    parser.add_argument("--historical-alpaca-source", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    source_path = args.source_path or args.repo / "data/expert_audit/databricks-dolly-15k.jsonl"
    output = args.output_dir or args.repo / "data/expert_audit"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    if not source_path.exists():
        with urlopen(SOURCE_URL, timeout=60) as response:
            raw = response.read(20_000_001)
        if len(raw) > 20_000_000 or hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
            raise ValueError("download exceeds size limit or fails pinned SHA256")
        source_path.write_bytes(raw)
    historical_manifest = args.repo / "data/expert_audit/historical_alpaca_prompts.jsonl"
    if not historical_manifest.exists():
        historical_source = args.historical_alpaca_source or args.repo / "data/expert_audit/alpaca_data.json"
        if not historical_source.exists():
            with urlopen(ALPACA_SOURCE_URL, timeout=60) as response:
                raw = response.read(30_000_001)
            if len(raw) > 30_000_000 or hashlib.sha256(raw).hexdigest() != ALPACA_SOURCE_SHA256:
                raise ValueError("historical source exceeds size limit or fails pinned SHA256")
            historical_source.write_bytes(raw)
        prior_rows, prior_provenance = make_historical_alpaca_manifest(historical_source)
        historical_manifest.write_bytes(
            ("\n".join(json.dumps(row, sort_keys=True) for row in prior_rows) + "\n").encode()
        )
        historical_manifest.with_name("historical_alpaca_provenance.json").write_bytes(
            (json.dumps(prior_provenance, indent=2, sort_keys=True) + "\n").encode()
        )
    historical, historical_sources = load_historical_prompts(args.repo)
    rows = make_expert_manifest(source_path, historical_prompts=historical)
    audit = validate_expert_manifest(rows, historical_prompts=historical)
    audit["historical_sources"] = historical_sources
    output.mkdir(parents=True, exist_ok=True)
    manifest_bytes = ("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n").encode()
    audit["manifest_file_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    artifacts = {
        "evaluation_manifest.jsonl": manifest_bytes,
        "data_audit.json": (json.dumps(audit, indent=2, sort_keys=True) + "\n").encode(),
        "source_provenance.json": (json.dumps({
            "dataset": SOURCE_ID, "revision": SOURCE_REVISION, "url": SOURCE_URL,
            "sha256": SOURCE_SHA256, "license": SOURCE_LICENSE,
            "attribution": "Copyright (2023) Databricks, Inc.; Wikipedia editors and contributors",
            "dataset_card": f"https://huggingface.co/datasets/{SOURCE_ID}/blob/{SOURCE_REVISION}/README.md",
            "transformations": "Select human-written instruction/context rows; omit responses;"
                " filter length and lexical duplicates; deterministic split and task allocation.",
            "model_outputs_accessed": False,
        }, indent=2, sort_keys=True) + "\n").encode(),
    }
    for name, raw in artifacts.items():
        path = output / name
        if path.exists() and path.read_bytes() != raw:
            raise ValueError(f"refusing to replace a different frozen artifact: {path}")
        path.write_bytes(raw)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
