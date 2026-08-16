#!/usr/bin/env python3
"""Create (or validate an existing) Pinecone integrated-embedding index.

DEFAULT: dry-run — prints the proposed plan without creating anything.

To create a real index both conditions MUST be satisfied:
  1. Pass --execute
  2. Set CONFIRM_PINECONE_CREATE=1 in the environment

An API key alone NEVER creates an index.
Never deletes or recreates an existing index.
"""

import argparse
import json
import os
import sys


def _print_plan(args: argparse.Namespace) -> None:
    plan = {
        "action": "CREATE_INDEX (dry-run)",
        "index_name": args.pinecone_index,
        "cloud": args.cloud,
        "region": args.region,
        "embed_model": args.embed_model,
        "text_field": "chunk_text",
        "field_map": {"text": "chunk_text"},
        "deletion_protection": False,
        "tags": {"project": "hhgoa-rag", "phase": "smoke"},
        "existing_index_action": "validate_and_refuse_if_incompatible",
        "note": ("Re-run with --execute and CONFIRM_PINECONE_CREATE=1 to create the real index."),
    }
    print(json.dumps(plan, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(
        description="Create Pinecone integrated-embedding index (dry-run by default)"
    )
    p.add_argument("--pinecone-index", required=True, help="Index name (e.g. msmarco-xi)")
    p.add_argument("--cloud", default=os.environ.get("PINECONE_CLOUD", "aws"))
    p.add_argument("--region", default=os.environ.get("PINECONE_REGION", "us-east-1"))
    p.add_argument("--embed-model", default="multilingual-e5-large")
    p.add_argument("--output-json", action="store_true")
    p.add_argument(
        "--execute",
        action="store_true",
        help="Attempt real index creation (also requires CONFIRM_PINECONE_CREATE=1)",
    )
    args = p.parse_args()

    # Dry-run if either guard is absent
    confirmed = os.environ.get("CONFIRM_PINECONE_CREATE", "").strip() == "1"
    if not args.execute or not confirmed:
        _print_plan(args)
        missing = []
        if not args.execute:
            missing.append("--execute flag")
        if not confirmed:
            missing.append("CONFIRM_PINECONE_CREATE=1 env var")
        print(
            f"\nDRY-RUN: no index created. Missing: {', '.join(missing)}",
            file=sys.stderr,
        )
        sys.exit(0)

    api_key = os.environ.get("PINECONE_API_KEY")
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
            tags={"project": "hhgoa-rag", "phase": "smoke"},
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
