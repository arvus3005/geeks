#!/usr/bin/env python3
"""Resume an interrupted ingestion from checkpoint."""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="Resume ingestion from checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint JSON")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    args = parser.parse_args()

    with open(args.checkpoint) as f:
        ckpt = json.load(f)

    if ckpt.get("mode") == "full":
        print("ERROR: Full ingestion resume not authorized in this session.", file=sys.stderr)
        sys.exit(1)

    print(f"Checkpoint: {ckpt}")
    print("Resume not implemented for pilot/full in this phase. See docs/INGESTION_RUNBOOK.md")
    sys.exit(0)


if __name__ == "__main__":
    main()
