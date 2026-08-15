#!/usr/bin/env python3
"""Atomically switch the serving alias. Refuses smoke/pilot for production alias."""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="Switch Qdrant collection alias atomically")
    parser.add_argument("--target-collection", required=True)
    parser.add_argument("--alias", default="msmarco_xi_passages_current")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument(
        "--smoke-ok",
        action="store_true",
        help="Allow smoke collections (for testing only)",
    )
    args = parser.parse_args()

    from qdrant_client import QdrantClient

    from hhgoa_rag.qdrant_lifecycle import switch_alias

    client = QdrantClient(url=args.qdrant_url)
    try:
        switch_alias(client, args.target_collection, alias=args.alias, smoke_ok=args.smoke_ok)
        print(json.dumps({"success": True, "alias": args.alias, "target": args.target_collection}))
        sys.exit(0)
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
