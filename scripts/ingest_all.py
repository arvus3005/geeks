#!/usr/bin/env python3
"""Ingest all passages from MSMARCO-XI into Qdrant.

Modes:
  smoke  — deterministic local fixtures, no network needed (default)
  pilot  — bounded real-data subset from HuggingFace (NOT FULL CORPUS)
  full   — entire dataset; requires --confirm-full-ingest
           DO NOT RUN UNTIL INFRASTRUCTURE AND COST ARE APPROVED
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ingest MSMARCO-XI passages into Qdrant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--mode", choices=["smoke", "pilot", "full"], default="smoke")
    p.add_argument("--config", default="configs/smoke.yaml")
    p.add_argument("--qdrant-url", default=None)
    p.add_argument("--collection", default=None, help="Override physical collection name")
    p.add_argument("--chunk-strategy", default="passage_native")
    p.add_argument("--dataset-revision", default=None, help="Pin dataset revision")
    p.add_argument(
        "--pilot-rows-per-shard",
        type=int,
        default=1000,
        help="Rows per language/split in pilot mode",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/checkpoints"))
    p.add_argument("--dedup-db-dir", type=Path, default=Path("artifacts/dedup"))
    p.add_argument("--run-id", default=None, help="Resume with existing run ID")
    p.add_argument(
        "--confirm-full-ingest", action="store_true", help="Required acknowledgement for full mode"
    )
    p.add_argument("--output-json", action="store_true")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    if args.mode == "full" and not args.confirm_full_ingest:
        print(
            "ERROR: --confirm-full-ingest is required for full mode.\n"
            "DO NOT RUN UNTIL INFRASTRUCTURE AND COST ARE APPROVED.\n"
            "See docs/INGESTION_RUNBOOK.md for prerequisites.",
            file=sys.stderr,
        )
        sys.exit(1)

    import yaml

    cfg: dict = {}
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}

    qdrant_url = args.qdrant_url or cfg.get("qdrant_url", "http://localhost:6333")
    collection = args.collection or cfg.get(
        "qdrant_collection_physical", f"msmarco_xi_passages_{args.mode}_v001"
    )

    # ── Smoke mode ────────────────────────────────────────────────────────────
    if args.mode == "smoke":
        from hhgoa_rag.ingestion.smoke_ingest import run_smoke_ingest

        result = run_smoke_ingest(qdrant_url=qdrant_url, collection=collection)
        if args.output_json:
            print(json.dumps(result, indent=2))
        else:
            status = "SUCCESS" if result.get("success") else "FAILED"
            print(f"{status}: {result.get('points_ingested', 0)} points in '{collection}'")
        sys.exit(0 if result.get("success") else 1)

    # ── Pilot / Full modes ────────────────────────────────────────────────────
    print(
        f"WARNING: {args.mode.upper()} MODE — "
        + ("NOT FULL CORPUS" if args.mode == "pilot" else "FULL CORPUS INGEST"),
        file=sys.stderr,
    )

    from qdrant_client import QdrantClient

    from hhgoa_rag.dataset.models import INDIC_LANGUAGE_CODES
    from hhgoa_rag.ingestion.dedup import ContentDeduplicator
    from hhgoa_rag.ingestion.engine import IngestionConfig, ingest_shard
    from hhgoa_rag.qdrant_lifecycle import create_collection, validate_collection
    from hhgoa_rag.retrieval.embedder import FakeEmbedder
    from hhgoa_rag.retrieval.sparse_encoder import BM25SparseEncoder

    run_id = args.run_id or str(uuid.uuid4())[:8]

    ingest_cfg = IngestionConfig(
        mode=args.mode,
        physical_collection=collection,
        chunk_strategy=args.chunk_strategy,
        dataset_revision=args.dataset_revision,
        batch_size=args.batch_size,
        pilot_rows_per_shard=args.pilot_rows_per_shard,
        checkpoint_dir=args.checkpoint_dir,
        dedup_db_dir=args.dedup_db_dir,
    )
    ingest_cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ingest_cfg.dedup_db_dir.mkdir(parents=True, exist_ok=True)

    # Full-mode safety gates
    if args.mode == "full":
        if collection == "msmarco_xi_passages_current":
            print("ERROR: Cannot ingest directly into the serving alias.", file=sys.stderr)
            sys.exit(1)
        # Capacity preflight placeholder — in production, check disk/RAM here
        logger.info("Full ingest preflight: collection=%s run_id=%s", collection, run_id)

    client = QdrantClient(url=qdrant_url, timeout=60)

    # Create physical collection if needed
    existing = {c.name for c in client.get_collections().collections}
    if collection not in existing:
        create_collection(client, collection, force=False)
        logger.info("Created collection %s", collection)

    # Load encoders once
    dense_embedder = FakeEmbedder()  # swap for E5MultilingualEmbedder in production
    sparse_encoder = BM25SparseEncoder()

    # Global English dedup (shared across all 14 Indic configs)
    dedup_en = ContentDeduplicator(ingest_cfg.dedup_db_dir / f"{run_id}_en_global.db")

    all_stats = []
    configs_to_run = INDIC_LANGUAGE_CODES  # 14 Indic languages
    splits = ["train", "validation"]

    for lang in configs_to_run:
        dedup_lang = ContentDeduplicator(ingest_cfg.dedup_db_dir / f"{run_id}_{lang}.db")
        for split in splits:
            logger.info("Ingesting %s/%s (mode=%s, shard=0)", lang, split, args.mode)
            try:
                stats = ingest_shard(
                    config_language=lang,
                    split=split,
                    shard_idx=0,
                    cfg=ingest_cfg,
                    dense_embedder=dense_embedder,
                    sparse_encoder=sparse_encoder,
                    client=client,
                    dedup_en=dedup_en,
                    dedup_lang=dedup_lang,
                    run_id=run_id,
                )
                all_stats.append(
                    {
                        "lang": lang,
                        "split": split,
                        "source_rows": stats.source_rows,
                        "valid_occurrences": stats.valid_occurrences,
                        "duplicate_occurrences": stats.duplicate_occurrences,
                        "rejected_occurrences": stats.rejected_occurrences,
                        "chunks_emitted": stats.chunks_emitted,
                        "qdrant_points": stats.qdrant_points_uploaded,
                        "elapsed_s": round(stats.elapsed_seconds, 1),
                    }
                )
            except Exception as e:
                logger.error("Shard %s/%s failed: %s", lang, split, e)
                all_stats.append({"lang": lang, "split": split, "error": str(e)})
        dedup_lang.close()

    dedup_en.close()

    validation = validate_collection(client, collection)
    summary = {
        "mode": args.mode,
        "run_id": run_id,
        "collection": collection,
        "shard_stats": all_stats,
        "validation": validation,
        "note": "EXPERIMENT SUBSET — NOT FULL CORPUS" if args.mode == "pilot" else "FULL CORPUS",
    }

    if args.output_json:
        print(json.dumps(summary, indent=2))
    else:
        total_points = sum(s.get("qdrant_points", 0) for s in all_stats if isinstance(s, dict))
        print(f"Run {run_id}: {total_points} points in '{collection}'")
        print(f"Validation: {validation['status']}, points={validation['points']}")
        if args.mode == "pilot":
            print("NOTE: EXPERIMENT SUBSET — NOT FULL CORPUS")

    sys.exit(0 if validation.get("valid") else 1)


if __name__ == "__main__":
    main()
