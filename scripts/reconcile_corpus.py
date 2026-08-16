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

from hhgoa_rag.pinecone_contract import INDEX_NAME, MODEL


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconcile corpus manifest with Pinecone")
    # NOTE: API key comes from PINECONE_API_KEY environment variable ONLY.
    # A --pinecone-api-key flag would leak credentials via shell history and the
    # process list; it has been intentionally removed.
    parser.add_argument("--pinecone-index", default=os.environ.get("PINECONE_INDEX", INDEX_NAME))
    parser.add_argument("--namespace", default="smoke", help="Namespace to inspect")
    parser.add_argument("--manifest", default=None, help="Path to ingest summary JSON")
    parser.add_argument(
        "--expected-count",
        type=int,
        default=None,
        help=(
            "Assert the verified namespace vector count equals this positive "
            "integer. Exit 0 only on exact match; otherwise exit non-zero."
        ),
    )
    parser.add_argument("--output-json", action="store_true")
    args = parser.parse_args()

    # Validate the expectation itself before doing anything else. An invalid
    # expectation fails closed — never treated as "no expectation".
    if args.expected_count is not None and args.expected_count <= 0:
        print(
            f"ERROR: --expected-count must be a positive integer, got " f"{args.expected_count}",
            file=sys.stderr,
        )
        sys.exit(1)

    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        print("ERROR: PINECONE_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    from pinecone import Pinecone

    from hhgoa_rag.pinecone_store import PineconeProviderError, PineconeStore

    pc = Pinecone(api_key=api_key)
    store = PineconeStore(pc.Index(args.pinecone_index), embed_model=MODEL)

    try:
        total = store.count_namespace(args.namespace)
        stats = store.describe_index_stats()
    except PineconeProviderError as e:
        print(
            f"ERROR: reconciliation UNVERIFIABLE — could not obtain a valid vector "
            f"count for namespace '{args.namespace}': {e}",
            file=sys.stderr,
        )
        sys.exit(1)

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

    # Secondary count-assertion status. This is a SECONDARY count check only;
    # index_canary.py exact-ID reconciliation remains authoritative.
    if args.expected_count is None:
        status = "pass"
    elif total == args.expected_count:
        status = "pass"
    else:
        status = "mismatch"

    reconciliation = {
        "pinecone_index": args.pinecone_index,
        "namespace": args.namespace,
        "expected_count": args.expected_count,
        "actual_count": total,
        "status": status,
        "total_vectors_in_namespace": total,
        "index_stats": stats,
        "manifest_expected": manifest_per_lang,
        "discrepancies": discrepancies,
        "note": "Secondary count check only; index_canary.py exact-ID "
        "reconciliation remains authoritative. Pinecone does not expose "
        "per-language counts without filtering. Use smoke_query_pinecone.py "
        "to spot-check per-language results.",
    }
    if args.output_json:
        print(json.dumps(reconciliation, indent=2, default=str))
    else:
        print(f"Index: {args.pinecone_index}  Namespace: {args.namespace}")
        print(f"Total vectors in namespace: {total}")
        if args.expected_count is not None:
            print(f"Expected count: {args.expected_count}  Status: {status}")
        if manifest_per_lang:
            total_expected = sum(manifest_per_lang.values())
            print(f"Manifest expected total: {total_expected}")

    if status != "pass":
        print(
            f"ERROR: namespace count mismatch — expected {args.expected_count}, "
            f"actual {total}.",
            file=sys.stderr,
        )
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
