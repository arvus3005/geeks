#!/usr/bin/env python3
"""Audit point ID determinism and Qdrant count consistency."""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="Audit Qdrant point IDs")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--fixtures", default="tests/fixtures/smoke_passages.json")
    args = parser.parse_args()

    from qdrant_client import QdrantClient

    from hhgoa_rag.ingestion.normalizer import content_hash
    from hhgoa_rag.ingestion.passage_ids import make_point_id

    client = QdrantClient(url=args.qdrant_url)
    count = client.count(args.collection).count

    with open(args.fixtures) as f:
        fixtures = json.load(f)

    DATASET_REVISION = "smoke-fixture-v1"
    CHUNK_STRATEGY_VERSION = "passage_native_v1"
    ids = set()
    for p in fixtures:
        chash = content_hash(p["text"])
        pid = make_point_id(
            dataset_revision=DATASET_REVISION,
            language=p["language"],
            content_hash=chash,
            chunk_strategy_version=CHUNK_STRATEGY_VERSION,
            chunk_ordinal=0,
        )
        ids.add(pid)

    result = {
        "collection": args.collection,
        "qdrant_point_count": count,
        "fixture_unique_ids": len(ids),
        "ids_match_count": count == len(ids),
        "determinism": "PASS" if len(ids) == len(fixtures) else "COLLISION_DETECTED",
    }
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["ids_match_count"] else 1)


if __name__ == "__main__":
    main()
