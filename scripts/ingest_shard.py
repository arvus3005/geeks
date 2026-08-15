#!/usr/bin/env python3
"""Ingest a single shard from MSMARCO-XI into Qdrant.

Pilot mode: bounded by --pilot-rows. Full mode requires --confirm-full-ingest.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    p = argparse.ArgumentParser(description="Ingest one MSMARCO-XI shard into Qdrant")
    p.add_argument("--config-lang", required=True, help="Language config code (e.g. bn)")
    p.add_argument("--split", required=True, choices=["train", "validation"])
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--mode", choices=["pilot", "full"], required=True)
    p.add_argument("--collection", required=True, help="Physical collection name")
    p.add_argument("--qdrant-url", default="http://localhost:6333")
    p.add_argument("--chunk-strategy", default="passage_native")
    p.add_argument("--dataset-revision", default=None)
    p.add_argument("--pilot-rows", type=int, default=1000, help="Max rows in pilot mode")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/checkpoints"))
    p.add_argument("--dedup-db-dir", type=Path, default=Path("artifacts/dedup"))
    p.add_argument("--run-id", default=None, help="Use existing run ID to resume")
    p.add_argument("--confirm-full-ingest", action="store_true")
    p.add_argument("--output-json", action="store_true")
    args = p.parse_args()

    if args.mode == "full" and not args.confirm_full_ingest:
        print(
            "ERROR: --confirm-full-ingest required for full mode.\n"
            "DO NOT RUN UNTIL INFRASTRUCTURE AND COST ARE APPROVED.",
            file=sys.stderr,
        )
        sys.exit(1)

    from qdrant_client import QdrantClient

    from hhgoa_rag.ingestion.dedup import ContentDeduplicator
    from hhgoa_rag.ingestion.engine import IngestionConfig, ingest_shard
    from hhgoa_rag.qdrant_lifecycle import create_collection
    from hhgoa_rag.retrieval.embedder import FakeEmbedder
    from hhgoa_rag.retrieval.sparse_encoder import BM25SparseEncoder

    run_id = args.run_id or str(uuid.uuid4())[:8]

    cfg = IngestionConfig(
        mode=args.mode,
        physical_collection=args.collection,
        chunk_strategy=args.chunk_strategy,
        dataset_revision=args.dataset_revision,
        batch_size=args.batch_size,
        pilot_rows_per_shard=args.pilot_rows,
        checkpoint_dir=args.checkpoint_dir,
        dedup_db_dir=args.dedup_db_dir,
    )
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg.dedup_db_dir.mkdir(parents=True, exist_ok=True)

    client = QdrantClient(url=args.qdrant_url, timeout=60)
    existing = {c.name for c in client.get_collections().collections}
    if args.collection not in existing:
        create_collection(client, args.collection, force=False)

    dense_embedder = FakeEmbedder()  # swap for E5MultilingualEmbedder in production
    sparse_encoder = BM25SparseEncoder()

    dedup_en = ContentDeduplicator(cfg.dedup_db_dir / f"{run_id}_en_global.db")
    dedup_lang = ContentDeduplicator(cfg.dedup_db_dir / f"{run_id}_{args.config_lang}.db")

    try:
        stats = ingest_shard(
            config_language=args.config_lang,
            split=args.split,
            shard_idx=args.shard,
            cfg=cfg,
            dense_embedder=dense_embedder,
            sparse_encoder=sparse_encoder,
            client=client,
            dedup_en=dedup_en,
            dedup_lang=dedup_lang,
            run_id=run_id,
        )
        result = {
            "run_id": run_id,
            "mode": args.mode,
            "lang": args.config_lang,
            "split": args.split,
            "shard": args.shard,
            "source_rows": stats.source_rows,
            "qdrant_points": stats.qdrant_points_uploaded,
            "chunks_emitted": stats.chunks_emitted,
            "elapsed_s": round(stats.elapsed_seconds, 1),
            "note": "EXPERIMENT SUBSET — NOT FULL CORPUS"
            if args.mode == "pilot"
            else "FULL CORPUS",
        }
        if args.output_json:
            print(json.dumps(result, indent=2))
        else:
            print(
                f"Shard {args.config_lang}/{args.split}/{args.shard}: {stats.qdrant_points_uploaded} points uploaded"
            )
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        dedup_en.close()
        dedup_lang.close()


if __name__ == "__main__":
    main()
