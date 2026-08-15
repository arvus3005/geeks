#!/usr/bin/env python3
"""Reconcile corpus manifest counts with Qdrant point counts.

For each language in the manifest, compares expected chunks against actual
Qdrant point counts filtered by language. Reports per-language discrepancies.
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Reconcile corpus manifest with Qdrant")
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--manifest", default=None, help="Path to ingest summary JSON")
    parser.add_argument("--output-json", action="store_true")
    args = parser.parse_args()

    import yaml

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}

    collection = cfg.get("qdrant_collection_physical", "msmarco_xi_passages_smoke_v001")

    from qdrant_client import QdrantClient
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    from hhgoa_rag.qdrant_lifecycle import validate_collection

    client = QdrantClient(url=args.qdrant_url)
    result = validate_collection(client, collection)

    per_lang: dict[str, int] = {}
    manifest_per_lang: dict[str, int] = {}

    # Query Qdrant point counts per language
    from hhgoa_rag.dataset.models import ALL_LANGUAGE_CODES

    for lang in ALL_LANGUAGE_CODES:
        try:
            r = client.count(
                collection_name=collection,
                count_filter=Filter(
                    must=[FieldCondition(key="language", match=MatchValue(value=lang))]
                ),
            )
            per_lang[lang] = r.count
        except Exception as e:
            per_lang[lang] = -1
            print(f"WARN: could not count lang={lang}: {e}", file=sys.stderr)

    # Load manifest if provided
    discrepancies: list[dict] = []
    if args.manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.exists():
            print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
            sys.exit(1)
        with open(manifest_path) as f:
            manifest = json.load(f)

        shard_stats = manifest.get("shard_stats", [])
        for shard in shard_stats:
            lang = shard.get("lang", "")
            expected_chunks = shard.get("chunks_emitted", 0)
            if lang and expected_chunks:
                manifest_per_lang[lang] = manifest_per_lang.get(lang, 0) + expected_chunks

        for lang, expected in manifest_per_lang.items():
            actual = per_lang.get(lang, 0)
            if actual < expected:
                discrepancies.append(
                    {"lang": lang, "expected_min": expected, "actual": actual, "gap": expected - actual}
                )

    reconciliation = {
        "collection": collection,
        "qdrant_total_points": result["points"],
        "status": result["status"],
        "valid": result["valid"],
        "issues": result["issues"],
        "per_language_counts": per_lang,
        "manifest_expected": manifest_per_lang,
        "discrepancies": discrepancies,
        "reconciliation_equation": (
            "source_occurrences = valid + rejected + unique + duplicate (see ingest summary)"
        ),
    }
    if args.output_json:
        print(json.dumps(reconciliation, indent=2))
    else:
        print(f"Collection: {collection}")
        print(f"Total points: {result['points']}  Status: {result['status']}")
        if discrepancies:
            print(f"DISCREPANCIES ({len(discrepancies)}):")
            for d in discrepancies:
                print(f"  lang={d['lang']} expected>={d['expected_min']} actual={d['actual']} gap={d['gap']}")
        else:
            print("No discrepancies (or no manifest provided for comparison)")
        print("Per-language counts:", json.dumps(per_lang))

    sys.exit(0 if result["valid"] and not discrepancies else 1)


if __name__ == "__main__":
    main()
