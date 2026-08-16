#!/usr/bin/env python3
"""Create (or validate an existing) Pinecone integrated-embedding index.

DEFAULT: dry-run — prints the proposed plan without creating anything.
No Pinecone import, no API key read on the dry-run path.

To create a real index BOTH conditions MUST be satisfied:
  1. Pass --execute
  2. Set CONFIRM_PINECONE_CREATE=1 in the environment

An API key alone NEVER creates an index.
--execute alone (without CONFIRM_PINECONE_CREATE=1) exits 2 (fail-closed).
Never deletes or recreates an existing index.
"""

import argparse
import json
import os
import sys


def _print_plan(args: argparse.Namespace) -> None:
    # Import canonical contract only on dry-run path (no Pinecone needed)
    from hhgoa_rag.pinecone_contract import canonical_contract, contract_fingerprint

    contract = canonical_contract()
    plan = {
        "action": "CREATE_INDEX (dry-run)",
        "index_name": args.pinecone_index,
        "cloud": args.cloud,
        "region": args.region,
        "embed_model": args.embed_model,
        "canonical_contract": contract,
        "contract_fingerprint": contract_fingerprint(),
        "existing_index_action": "validate_and_refuse_if_incompatible",
        "note": (
            "Re-run with --execute AND CONFIRM_PINECONE_CREATE=1 to create the real index. "
            "--execute alone exits 2 (fail-closed)."
        ),
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

    confirmed = os.environ.get("CONFIRM_PINECONE_CREATE", "").strip() == "1"

    # --execute without the confirmation env var is an explicit error (exit 2).
    # This is fail-closed: --execute alone must never silently downgrade to dry-run.
    if args.execute and not confirmed:
        print(
            "ERROR: --execute requires CONFIRM_PINECONE_CREATE=1 to be set. "
            "Refusing to proceed. Exit 2.",
            file=sys.stderr,
        )
        sys.exit(2)

    # No --execute: dry-run (no Pinecone import, no API key read)
    if not args.execute:
        _print_plan(args)
        print(
            "\nDRY-RUN: no index created. Pass --execute AND CONFIRM_PINECONE_CREATE=1 to create.",
            file=sys.stderr,
        )
        sys.exit(0)

    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        print("ERROR: PINECONE_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    from pinecone import Pinecone

    from hhgoa_rag.pinecone_contract import CLOUD, MODEL, REGION, contract_fingerprint
    from hhgoa_rag.pinecone_lifecycle import create_index_idempotent, get_index_info

    # Validate CLI args against canonical contract before any API call
    if args.cloud != CLOUD:
        print(
            f"ERROR: --cloud '{args.cloud}' does not match canonical contract '{CLOUD}'.",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.region != REGION:
        print(
            f"ERROR: --region '{args.region}' does not match canonical contract '{REGION}'.",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.embed_model != MODEL:
        print(
            f"ERROR: --embed-model '{args.embed_model}' does not match canonical contract '{MODEL}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    pc = Pinecone(api_key=api_key)

    try:
        create_index_idempotent(
            pc,
            name=args.pinecone_index,
            cloud=args.cloud,
            region=args.region,
            embed_model=args.embed_model,
            tags={"project": "hhgoa-rag", "phase": "smoke", "contract_fp": contract_fingerprint()},
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
