#!/usr/bin/env python3
"""Validate Pinecone configuration — API key, index existence, embed model, field map."""

import argparse
import json
import os
import sys

from hhgoa_rag.pinecone_contract import CLOUD, INDEX_NAME, MODEL, REGION


def main() -> None:
    p = argparse.ArgumentParser(description="Validate Pinecone config")
    # NOTE: API key must come from PINECONE_API_KEY environment variable only.
    # Passing credentials on the command line is a security risk (shell history,
    # process list). The --pinecone-api-key flag has been intentionally removed.
    p.add_argument("--pinecone-index", default=os.environ.get("PINECONE_INDEX", INDEX_NAME))
    p.add_argument("--embed-model", default=MODEL)
    p.add_argument("--cloud", default=os.environ.get("PINECONE_CLOUD", CLOUD))
    p.add_argument("--region", default=os.environ.get("PINECONE_REGION", REGION))
    p.add_argument("--output-json", action="store_true")
    args = p.parse_args()

    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        print("ERROR: PINECONE_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    from pinecone import Pinecone

    from hhgoa_rag.pinecone_lifecycle import get_index_info, validate_index

    pc = Pinecone(api_key=api_key)
    indexes = {idx.name for idx in pc.list_indexes().indexes}

    if args.pinecone_index not in indexes:
        result = {
            "status": "index_not_found",
            "index": args.pinecone_index,
            "available_indexes": list(indexes),
        }
        if args.output_json:
            print(json.dumps(result, indent=2))
        else:
            print(f"ERROR: Index '{args.pinecone_index}' not found. Available: {list(indexes)}")
        sys.exit(1)

    errors = validate_index(pc, args.pinecone_index)
    info = get_index_info(pc, args.pinecone_index)

    result = {
        "status": "ok" if not errors else "config_mismatch",
        "index": args.pinecone_index,
        "info": info,
        "errors": errors,
    }
    if args.output_json:
        print(json.dumps(result, indent=2))
    else:
        if errors:
            print("CONFIG ERRORS:")
            for e in errors:
                print(f"  • {e}")
            sys.exit(1)
        else:
            print(f"OK: index='{args.pinecone_index}' embed_model='{info.get('embed_model')}'")


if __name__ == "__main__":
    main()
