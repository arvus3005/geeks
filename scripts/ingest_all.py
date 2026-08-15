#!/usr/bin/env python3
"""Ingest all passages (smoke/pilot/full modes)."""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="Ingest MSMARCO-XI passages into Qdrant")
    parser.add_argument("--mode", choices=["smoke", "pilot", "full"], default="smoke")
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument(
        "--confirm-full-ingest",
        action="store_true",
        help="Required for full mode",
    )
    args = parser.parse_args()

    if args.mode == "full" and not args.confirm_full_ingest:
        print(
            "ERROR: Full ingestion requires --confirm-full-ingest. "
            "DO NOT RUN UNTIL INFRASTRUCTURE IS APPROVED.",
            file=sys.stderr,
        )
        sys.exit(1)

    import yaml

    cfg = {}
    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}

    collection = cfg.get("qdrant_collection_physical", f"msmarco_xi_passages_{args.mode}_v001")
    qdrant_url = args.qdrant_url or cfg.get("qdrant_url", "http://localhost:6333")

    if args.mode == "smoke":
        from hhgoa_rag.ingestion.smoke_ingest import run_smoke_ingest

        result = run_smoke_ingest(qdrant_url=qdrant_url, collection=collection)
        print(json.dumps(result, indent=2))
        sys.exit(0 if result.get("success") else 1)
    elif args.mode == "pilot":
        print("WARNING: PILOT MODE — NOT FULL CORPUS", file=sys.stderr)
        print("Pilot ingestion not yet implemented (requires dataset access)", file=sys.stderr)
        sys.exit(1)
    else:
        print(
            "Full ingestion not launched in this phase. See docs/INGESTION_RUNBOOK.md",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
