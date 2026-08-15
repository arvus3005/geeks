#!/usr/bin/env python3
"""Discover ai4bharat/MSMARCO-XI dataset structure and write manifest."""

import argparse
import dataclasses
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Discover MSMARCO-XI dataset structure")
    parser.add_argument("--revision", default=None, help="Dataset revision/commit to pin")
    parser.add_argument("--manifest-dir", default="artifacts/manifests", type=Path)
    parser.add_argument("--output-json", action="store_true", help="Print manifest as JSON")
    args = parser.parse_args()

    from hhgoa_rag.dataset.discovery import discover_dataset

    manifest = discover_dataset(revision=args.revision, manifest_dir=args.manifest_dir)

    if manifest.missing_configs:
        print(f"WARNING: Missing expected configs: {manifest.missing_configs}", file=sys.stderr)

    if args.output_json:
        print(json.dumps(dataclasses.asdict(manifest), indent=2))
    else:
        print(f"Dataset: {manifest.dataset_repo}")
        print(f"Observed configs: {manifest.observed_configs}")
        print(f"Missing configs: {manifest.missing_configs}")
        print(f"Manifest written: {manifest.manifest_path}")

    sys.exit(1 if manifest.missing_configs else 0)


if __name__ == "__main__":
    main()
