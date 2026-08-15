#!/usr/bin/env python3
"""Create a Qdrant snapshot of a collection."""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="Create Qdrant collection snapshot")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    args = parser.parse_args()

    from qdrant_client import QdrantClient

    client = QdrantClient(url=args.qdrant_url)
    try:
        result = client.create_snapshot(collection_name=args.collection)
        print(json.dumps({"success": True, "snapshot": str(result)}))
        sys.exit(0)
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
