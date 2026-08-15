#!/usr/bin/env python3
"""Resume an interrupted ingestion from checkpoint.

Reads an existing checkpoint, validates compatibility with current config,
then continues from the last acknowledged row.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Resume interrupted MSMARCO-XI ingestion")
    p.add_argument("--checkpoint", required=True, type=Path, help="Path to checkpoint JSON")
    p.add_argument("--pinecone-api-key", default=None)
    p.add_argument("--dedup-db-dir", type=Path, default=Path("artifacts/dedup"))
    p.add_argument(
        "--confirm-full-ingest",
        action="store_true",
        help="Required to resume a full-mode ingestion",
    )
    p.add_argument("--output-json", action="store_true")
    args = p.parse_args()

    from pinecone import Pinecone

    from hhgoa_rag.ingestion.checkpoint import IngestCheckpoint
    from hhgoa_rag.ingestion.dedup import ContentDeduplicator
    from hhgoa_rag.ingestion.engine import IngestionConfig, ingest_shard
    from hhgoa_rag.pinecone_store import PineconeStore

    ckpt = IngestCheckpoint.load(args.checkpoint)

    if ckpt.mode == "full" and not args.confirm_full_ingest:
        print(
            "ERROR: --confirm-full-ingest required to resume full-mode ingestion.\n"
            "DO NOT RUN UNTIL INFRASTRUCTURE AND COST ARE APPROVED.",
            file=sys.stderr,
        )
        sys.exit(1)

    if ckpt.status == "complete":
        result = {"status": "already_complete", "checkpoint": str(args.checkpoint)}
        if args.output_json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Checkpoint already complete: {args.checkpoint}")
        sys.exit(0)

    api_key = args.pinecone_api_key or os.environ.get("PINECONE_API_KEY")
    if not api_key:
        print("ERROR: PINECONE_API_KEY must be set", file=sys.stderr)
        sys.exit(1)

    cfg = IngestionConfig(
        mode=ckpt.mode,
        pinecone_index=ckpt.pinecone_index,
        pinecone_namespace=ckpt.pinecone_namespace,
        embed_model=ckpt.embed_model,
        chunk_strategy=ckpt.chunk_strategy,
        chunk_strategy_version=ckpt.chunk_strategy_version,
        dataset_revision=ckpt.dataset_revision,
        dedup_db_dir=args.dedup_db_dir,
        checkpoint_dir=args.checkpoint.parent,
    )

    pc = Pinecone(api_key=api_key)
    store = PineconeStore(pc.Index(ckpt.pinecone_index), embed_model=ckpt.embed_model)

    dedup_en = ContentDeduplicator(cfg.dedup_db_dir / f"{ckpt.run_id}_en_global.db")
    dedup_lang = ContentDeduplicator(cfg.dedup_db_dir / f"{ckpt.run_id}_{ckpt.config_language}.db")

    try:
        stats = ingest_shard(
            config_language=ckpt.config_language,
            split=ckpt.split,
            shard_idx=ckpt.source_shard,
            cfg=cfg,
            store=store,
            dedup_en=dedup_en,
            dedup_lang=dedup_lang,
            run_id=ckpt.run_id,
        )
        result = {
            "run_id": ckpt.run_id,
            "resumed_from_row": ckpt.last_acknowledged_row,
            "source_rows_total": stats.source_rows,
            "indexed_points_total": stats.indexed_points,
        }
        if args.output_json:
            print(json.dumps(result, indent=2))
        else:
            print(
                f"Resumed {ckpt.config_language}/{ckpt.split}/shard{ckpt.source_shard}: "
                f"{stats.indexed_points} total points"
            )
        sys.exit(0)
    except Exception as e:
        print(f"ERROR during resume: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        dedup_en.close()
        dedup_lang.close()


if __name__ == "__main__":
    main()
