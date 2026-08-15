#!/usr/bin/env python3
"""Create (or validate an existing) Pinecone integrated-embedding index.

Never deletes or recreates an existing index. If the index already exists,
validates its config and prints actionable errors if incompatible.

IMPORTANT: This script creates a real cloud resource. Run only after confirming
index name, cloud, and region with the operator.
"""

import argparse
import json
import os
import sys


def main() -> None:
    p = argparse.ArgumentParser(description="Create Pinecone integrated-embedding index")
    p.add_argument("--pinecone-api-key", default=None)
    p.add_argument("--pinecone-index", required=True, help="Index name (e.g. msmarco-xi)")
    p.add_argument("--cloud", default=os.environ.get("PINECONE_CLOUD", "aws"))
    p.add_argument("--region", default=os.environ.get("PINECONE_REGION", "us-east-1"))
    p.add_argument("--embed-model", default="multilingual-e5-large")
    p.add_argument("--output-json", action="store_true")
    args = p.parse_args()

    api_key = args.pinecone_api_key or os.environ.get("PINECONE_API_KEY")
    if not api_key:
        print("ERROR: PINECONE_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    from pinecone import Pinecone

    from hhgoa_rag.pinecone_lifecycle import create_index_idempotent, get_index_info

    pc = Pinecone(api_key=api_key)

    try:
        create_index_idempotent(
            pc,
            name=args.pinecone_index,
            cloud=args.cloud,
            region=args.region,
            embed_model=args.embed_model,
            tags={"project": "hhgoa-rag", "phase": "3a"},
        )
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    info = get_index_info(pc, args.pinecone_index)
    result = {"status": "ok", "index": args.pinecone_index, "info": info}

    if args.output_json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Index '{args.pinecone_index}' ready. embed_model='{info.get('embed_model')}'")


if __name__ == "__main__":
    main()
