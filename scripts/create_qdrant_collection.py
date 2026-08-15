#!/usr/bin/env python3
"""Create a versioned Qdrant collection. Never destructive on production names."""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="Create Qdrant collection")
    parser.add_argument("--config", default="configs/smoke.yaml", help="YAML config file")
    parser.add_argument("--collection", default=None, help="Override collection name")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and recreate (smoke/pilot only)",
    )
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--output-json", action="store_true")
    args = parser.parse_args()

    import yaml

    cfg = {}
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}

    collection = args.collection or cfg.get(
        "qdrant_collection_physical", "msmarco_xi_passages_smoke_v001"
    )
    qdrant_url = args.qdrant_url or cfg.get("qdrant_url", "http://localhost:6333")

    from qdrant_client import QdrantClient

    from hhgoa_rag.qdrant_lifecycle import create_collection, validate_collection

    client = QdrantClient(url=qdrant_url)
    try:
        create_collection(client, collection, force=args.force)
        result = validate_collection(client, collection)
        if args.output_json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Created and validated collection: {collection}")
            print(f"Status: {result['status']}, Points: {result['points']}")
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
