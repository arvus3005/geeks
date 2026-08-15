#!/usr/bin/env python3
"""Audit point ID determinism — verify UUIDv5 IDs are stable across calls."""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="Audit Pinecone point ID determinism")
    parser.add_argument("--fixtures", default="tests/fixtures/smoke_passages.json")
    parser.add_argument("--output-json", action="store_true")
    args = parser.parse_args()

    from hhgoa_rag.ingestion.normalizer import content_hash
    from hhgoa_rag.ingestion.passage_ids import make_point_id

    with open(args.fixtures) as f:
        fixtures = json.load(f)

    DATASET_REVISION = "smoke-fixture-v1"
    CHUNK_STRATEGY_VERSION = "passage_native_v1"
    ids: set[str] = set()
    collisions = 0
    for p in fixtures:
        chash = content_hash(p["text"])
        pid = make_point_id(
            dataset_revision=DATASET_REVISION,
            language=p["language"],
            content_hash=chash,
            chunk_strategy_version=CHUNK_STRATEGY_VERSION,
            chunk_ordinal=0,
        )
        if pid in ids:
            collisions += 1
        ids.add(pid)

    result = {
        "fixture_count": len(fixtures),
        "unique_ids": len(ids),
        "collisions": collisions,
        "determinism": "PASS" if collisions == 0 else "COLLISION_DETECTED",
    }
    if args.output_json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Fixtures: {len(fixtures)}  Unique IDs: {len(ids)}  Collisions: {collisions}")
        print(f"Determinism: {result['determinism']}")
    sys.exit(0 if collisions == 0 else 1)


if __name__ == "__main__":
    main()
