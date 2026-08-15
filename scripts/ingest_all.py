#!/usr/bin/env python3
"""Ingest all passages from MSMARCO-XI into Pinecone.

Modes:
  smoke  — deterministic local fixtures, no network HuggingFace needed (default)
  pilot  — bounded real-data subset from HuggingFace (NOT FULL CORPUS)
  full   — entire dataset; requires --confirm-full-ingest
           DO NOT RUN UNTIL INFRASTRUCTURE AND COST ARE APPROVED

Pinecone integrated multilingual embedding (multilingual-e5-large) is used
server-side. No local dense model is loaded during ingestion.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import uuid
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ingest MSMARCO-XI passages into Pinecone",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--mode", choices=["smoke", "pilot", "full"], default="smoke")
    p.add_argument("--config", default="configs/smoke.yaml")
    p.add_argument("--pinecone-api-key", default=None, help="Override PINECONE_API_KEY env var")
    p.add_argument("--pinecone-index", default=None, help="Override Pinecone index name")
    p.add_argument("--pinecone-namespace", default=None, help="Override namespace")
    p.add_argument("--chunk-strategy", default="passage_native")
    p.add_argument("--dataset-revision", default=None, help="Pin dataset revision")
    p.add_argument("--pilot-rows-per-shard", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=96)
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

    api_key = args.pinecone_api_key or os.environ.get("PINECONE_API_KEY") or cfg.get("pinecone_api_key")
    if not api_key:
        print("ERROR: PINECONE_API_KEY must be set (env var or --pinecone-api-key)", file=sys.stderr)
        sys.exit(1)

    index_name = args.pinecone_index or cfg.get("pinecone_index", "msmarco-xi")
    embed_model = cfg.get("pinecone_embed_model", "multilingual-e5-large")

    # ── Smoke mode ────────────────────────────────────────────────────────────
    if args.mode == "smoke":
        from pinecone import Pinecone

        from hhgoa_rag.ingestion.smoke_ingest import run_smoke_ingest
        from hhgoa_rag.pinecone_store import SMOKE_NAMESPACE, PineconeStore

        pc = Pinecone(api_key=api_key)
        store = PineconeStore(pc.Index(index_name), embed_model=embed_model)
        result = run_smoke_ingest(store=store, namespace=SMOKE_NAMESPACE)
        if args.output_json:
            print(json.dumps(result, indent=2))
        else:
            status = "SUCCESS" if result.get("success") else "FAILED"
            print(f"{status}: {result.get('records_submitted', 0)} records in '{SMOKE_NAMESPACE}'")
        sys.exit(0 if result.get("success") else 1)

    # ── Pilot / Full modes ────────────────────────────────────────────────────
    print(
        f"WARNING: {args.mode.upper()} MODE — "
        + ("EXPERIMENT SUBSET — NOT FULL CORPUS" if args.mode == "pilot" else "FULL CORPUS INGEST"),
        file=sys.stderr,
    )

    from pinecone import Pinecone

    from hhgoa_rag.dataset.models import INDIC_LANGUAGE_CODES
    from hhgoa_rag.ingestion.dedup import ContentDeduplicator
    from hhgoa_rag.ingestion.engine import IngestionConfig, ingest_shard
    from hhgoa_rag.pinecone_store import FULL_NAMESPACE, PILOT_NAMESPACE_PREFIX, PineconeStore

    run_id = args.run_id or str(uuid.uuid4())[:8]

    if args.mode == "pilot":
        namespace = args.pinecone_namespace or f"{PILOT_NAMESPACE_PREFIX}{run_id}"
    else:
        namespace = FULL_NAMESPACE

    ingest_cfg = IngestionConfig(
        mode=args.mode,
        pinecone_index=index_name,
        pinecone_namespace=namespace,
        embed_model=embed_model,
        chunk_strategy=args.chunk_strategy,
        dataset_revision=args.dataset_revision,
        batch_size=args.batch_size,
        pilot_rows_per_shard=args.pilot_rows_per_shard,
        checkpoint_dir=args.checkpoint_dir,
        dedup_db_dir=args.dedup_db_dir,
    )
    ingest_cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ingest_cfg.dedup_db_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "full":
        if namespace != FULL_NAMESPACE:
            print(f"ERROR: Full mode must use namespace '{FULL_NAMESPACE}'", file=sys.stderr)
            sys.exit(1)

    pc = Pinecone(api_key=api_key)
    store = PineconeStore(pc.Index(index_name), embed_model=embed_model)

    dedup_en = ContentDeduplicator(ingest_cfg.dedup_db_dir / f"{run_id}_en_global.db")

    all_stats = []
    splits = ["train", "validation"]

    for lang in INDIC_LANGUAGE_CODES:
        dedup_lang = ContentDeduplicator(ingest_cfg.dedup_db_dir / f"{run_id}_{lang}.db")
        for split in splits:
            logger.info("Ingesting %s/%s (mode=%s, ns=%s)", lang, split, args.mode, namespace)
            try:
                stats = ingest_shard(
                    config_language=lang,
                    split=split,
                    shard_idx=0,
                    cfg=ingest_cfg,
                    store=store,
                    dedup_en=dedup_en,
                    dedup_lang=dedup_lang,
                    run_id=run_id,
                )
                all_stats.append({
                    "lang": lang,
                    "split": split,
                    "source_rows": stats.source_rows,
                    "valid_occurrences": stats.valid_occurrences,
                    "duplicate_occurrences": stats.duplicate_occurrences,
                    "rejected_occurrences": stats.rejected_occurrences,
                    "chunks_emitted": stats.chunks_emitted,
                    "indexed_points": stats.indexed_points,
                    "elapsed_s": round(stats.elapsed_seconds, 1),
                })
            except Exception as e:
                logger.error("Shard %s/%s failed: %s", lang, split, e)
                all_stats.append({"lang": lang, "split": split, "error": str(e)})
        dedup_lang.close()

    dedup_en.close()

    summary = {
        "mode": args.mode,
        "run_id": run_id,
        "pinecone_index": index_name,
        "namespace": namespace,
        "shard_stats": all_stats,
        "note": "EXPERIMENT SUBSET — NOT FULL CORPUS" if args.mode == "pilot" else "FULL CORPUS",
    }

    if args.output_json:
        print(json.dumps(summary, indent=2))
    else:
        total = sum(s.get("indexed_points", 0) for s in all_stats if isinstance(s, dict))
        print(f"Run {run_id}: {total} points in index='{index_name}' ns='{namespace}'")
        if args.mode == "pilot":
            print("NOTE: EXPERIMENT SUBSET — NOT FULL CORPUS")

    sys.exit(0)


if __name__ == "__main__":
    main()
