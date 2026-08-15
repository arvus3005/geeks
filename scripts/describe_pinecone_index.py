#!/usr/bin/env python3
"""Describe Pinecone index statistics — total vectors, per-namespace counts."""

import argparse
import json
import os
import sys


def main() -> None:
    p = argparse.ArgumentParser(description="Describe Pinecone index stats")
    p.add_argument("--pinecone-api-key", default=None)
    p.add_argument("--pinecone-index", default=os.environ.get("PINECONE_INDEX", "msmarco-xi"))
    p.add_argument("--output-json", action="store_true")
    args = p.parse_args()

    api_key = args.pinecone_api_key or os.environ.get("PINECONE_API_KEY")
    if not api_key:
        print("ERROR: PINECONE_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    from pinecone import Pinecone

    from hhgoa_rag.pinecone_lifecycle import get_index_info
    from hhgoa_rag.pinecone_store import PineconeStore

    pc = Pinecone(api_key=api_key)
    index = pc.Index(args.pinecone_index)
    store = PineconeStore(index, embed_model="multilingual-e5-large")

    info = get_index_info(pc, args.pinecone_index)
    stats = store.describe_index_stats()

    result = {"index": args.pinecone_index, "info": info, "stats": stats}
    if args.output_json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"Index: {args.pinecone_index}")
        print(f"Embed model: {info.get('embed_model')}")
        print(f"Stats: {stats}")


if __name__ == "__main__":
    main()
