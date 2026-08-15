#!/usr/bin/env python3
"""Ingest a single shard from MSMARCO-XI into Qdrant."""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Ingest one shard from MSMARCO-XI")
    parser.add_argument("--config-lang", required=True, help="Language config (e.g. bn)")
    parser.add_argument("--split", required=True, choices=["train", "validation"])
    parser.add_argument("--shard", type=int, default=0, help="Shard index")
    parser.add_argument("--mode", choices=["smoke", "pilot", "full"], default="pilot")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--checkpoint-dir", default="artifacts/checkpoints", type=Path)
    args = parser.parse_args()

    if args.mode == "full":
        print(
            "ERROR: Full shard ingestion not authorized in this session. "
            "See INGESTION_RUNBOOK.md",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"Shard ingestion for {args.config_lang}/{args.split}/shard{args.shard} "
        f"in {args.mode} mode"
    )
    print("NOTE: Requires Hugging Face dataset access. Not run in this CI session.")
    sys.exit(0)


if __name__ == "__main__":
    main()
