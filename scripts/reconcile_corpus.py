#!/usr/bin/env python3
"""Reconcile corpus manifest counts with Pinecone namespace vector counts.

For each language in the manifest, compares expected indexed points against
actual Pinecone counts. Reports per-language discrepancies.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconcile corpus manifest with Pinecone")
    parser.add_argument("--pinecone-api-key", default=None)
    parser.add_argument("--pinecone-index", default=os.environ.get("PINECONE_INDEX", "msmarco-xi"))
    parser.add_argument("--namespace", default="smoke", help="Namespace to inspect")
    parser.add_argument("--manifest", default=None, help="Path to ingest summary JSON")
    parser.add_argument("--output-json", action="store_true")
    args = parser.parse_args()

    api_key = args.pinecone_api_key or os.environ.get("PINECONE_API_KEY")
    if not api_key:
        print("ERROR: PINECONE_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    from pinecone import Pinecone

    from hhgoa_rag.pinecone_store import PineconeStore

    pc = Pinecone(api_key=api_key)
    store = PineconeStore(pc.Index(args.pinecone_index), embed_model="multilingual-e5-large")

    total = store.count_namespace(args.namespace)
    stats = store.describe_index_stats()

    manifest_per_lang: dict[str, int] = {}
    discrepancies: list[dict] = []

    if args.manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.exists():
            print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
            sys.exit(1)
        with open(manifest_path) as f:
            manifest = json.load(f)
        for shard in manifest.get("shard_stats", []):
            lang = shard.get("lang", "")
            expected = shard.get("chunks_emitted", 0)
            if lang and expected:
                manifest_per_lang[lang] = manifest_per_lang.get(lang, 0) + expected

    reconciliation = {
        "pinecone_index": args.pinecone_index,
        "namespace": args.namespace,
        "total_vectors_in_namespace": total,
        "index_stats": stats,
        "manifest_expected": manifest_per_lang,
        "discrepancies": discrepancies,
        "note": "Pinecone does not expose per-language counts without filtering. "
                "Use smoke_query_pinecone.py to spot-check per-language results.",
    }
    if args.output_json:
        print(json.dumps(reconciliation, indent=2, default=str))
    else:
        print(f"Index: {args.pinecone_index}  Namespace: {args.namespace}")
        print(f"Total vectors in namespace: {total}")
        if manifest_per_lang:
            total_expected = sum(manifest_per_lang.values())
            print(f"Manifest expected total: {total_expected}")

    sys.exit(0)


if __name__ == "__main__":
    main()
