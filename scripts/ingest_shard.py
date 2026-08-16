#!/usr/bin/env python3
"""Ingest a single shard from MSMARCO-XI into Pinecone.

Pilot mode: bounded by --pilot-rows. Full mode requires --confirm-full-ingest
AND a non-Starter PINECONE_PLAN.
Pinecone handles server-side embedding; no local model is loaded.
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


def main() -> None:
    p = argparse.ArgumentParser(description="Ingest one MSMARCO-XI shard into Pinecone")
    p.add_argument("--config-lang", required=True)
    p.add_argument("--split", required=True, choices=["train", "validation"])
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--mode", choices=["pilot", "full"], required=True)
    p.add_argument("--pinecone-index", required=True)
    p.add_argument("--pinecone-namespace", required=True)
    p.add_argument("--chunk-strategy", default="passage_native")
    p.add_argument("--dataset-revision", default=None)
    p.add_argument("--pilot-rows", type=int, default=1000)
    p.add_argument(
        "--batch-size",
        type=int,
        default=96,
        help="Records per Pinecone request (1-96; Pinecone hard limit is 96)",
    )
    p.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/checkpoints"))
    p.add_argument("--dedup-db-dir", type=Path, default=Path("artifacts/dedup"))
    p.add_argument("--run-id", default=None)
    p.add_argument("--confirm-full-ingest", action="store_true")
    p.add_argument(
        "--execute",
        action="store_true",
        help="Perform real Pinecone writes (also requires CONFIRM_PINECONE_WRITE=1)",
    )
    p.add_argument("--output-json", action="store_true")
    args = p.parse_args()

    # ── Block raw-HF pilot ingestion — redirect to ingest_prepared.py ──────
    if args.mode == "pilot":
        print(
            f"ERROR: Raw HuggingFace pilot ingestion via {__file__} is blocked.\n"
            "Use the prepared-data path instead:\n"
            "  uv run python scripts/prepare_canary.py --dataset-revision <sha> "
            "--tokenizer-revision <sha>\n"
            "  uv run python scripts/ingest_prepared.py "
            "--manifest artifacts/prepared/<id>_manifest.json --execute",
            file=sys.stderr,
        )
        sys.exit(1)

    if not (1 <= args.batch_size <= 96):
        print(
            f"ERROR: --batch-size must be between 1 and 96, got {args.batch_size}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Starter full-mode block — BEFORE all other checks
    from hhgoa_rag.ingestion.budget import get_plan

    plan = get_plan()
    if args.mode == "full" and plan == "starter":
        print(
            "ERROR: Full-corpus ingestion is permanently blocked on Pinecone Starter plan.\n"
            f"PINECONE_PLAN={plan!r} is active. No flag or environment variable can override this.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.mode == "full" and not args.confirm_full_ingest:
        print(
            "ERROR: --confirm-full-ingest required for full mode.\n"
            "DO NOT RUN UNTIL INFRASTRUCTURE AND COST ARE APPROVED.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.mode == "full":
        full_env = os.environ.get("CONFIRM_FULL_INGEST", "").strip()
        if full_env != "YES_I_APPROVE_FULL_CORPUS":
            print(
                "ERROR: CONFIRM_FULL_INGEST=YES_I_APPROVE_FULL_CORPUS required for full mode.\n"
                "DO NOT RUN UNTIL INFRASTRUCTURE AND COST ARE APPROVED.",
                file=sys.stderr,
            )
            sys.exit(1)

    write_confirmed = os.environ.get("CONFIRM_PINECONE_WRITE", "").strip() == "1"
    if not args.execute or not write_confirmed:
        missing = []
        if not args.execute:
            missing.append("--execute flag")
        if not write_confirmed:
            missing.append("CONFIRM_PINECONE_WRITE=1 env var")
        print(
            f"DRY-RUN: no Pinecone writes. Missing: {', '.join(missing)}",
            file=sys.stderr,
        )
        sys.exit(0)

    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        print("ERROR: PINECONE_API_KEY must be set", file=sys.stderr)
        sys.exit(1)

    from pinecone import Pinecone

    from hhgoa_rag.ingestion.budget import make_default_guard
    from hhgoa_rag.ingestion.dedup import ContentDeduplicator
    from hhgoa_rag.ingestion.engine import IngestionConfig, ingest_shard
    from hhgoa_rag.pinecone_store import FULL_NAMESPACE, PineconeStore

    run_id = args.run_id or str(uuid.uuid4())[:8]

    if args.mode == "full" and args.pinecone_namespace != FULL_NAMESPACE:
        print(
            f"ERROR: Full mode must use namespace '{FULL_NAMESPACE}', got '{args.pinecone_namespace}'",
            file=sys.stderr,
        )
        sys.exit(1)

    cfg = IngestionConfig(
        mode=args.mode,
        pinecone_index=args.pinecone_index,
        pinecone_namespace=args.pinecone_namespace,
        chunk_strategy=args.chunk_strategy,
        dataset_revision=args.dataset_revision,
        batch_size=args.batch_size,
        pilot_rows_per_shard=args.pilot_rows,
        checkpoint_dir=args.checkpoint_dir,
        dedup_db_dir=args.dedup_db_dir,
        num_workers=1,
    )
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg.dedup_db_dir.mkdir(parents=True, exist_ok=True)

    pc = Pinecone(api_key=api_key)
    store = PineconeStore(pc.Index(args.pinecone_index), embed_model=cfg.embed_model)
    guard = make_default_guard()

    dedup_en = ContentDeduplicator(cfg.dedup_db_dir / f"{run_id}_en_global.db")
    dedup_lang = ContentDeduplicator(cfg.dedup_db_dir / f"{run_id}_{args.config_lang}.db")

    try:
        stats = ingest_shard(
            config_language=args.config_lang,
            split=args.split,
            shard_idx=args.shard,
            cfg=cfg,
            store=store,
            dedup_en=dedup_en,
            dedup_lang=dedup_lang,
            run_id=run_id,
            budget_guard=guard,
        )
        result = {
            "run_id": run_id,
            "mode": args.mode,
            "lang": args.config_lang,
            "split": args.split,
            "shard": args.shard,
            "source_rows": stats.source_rows,
            "indexed_points": stats.indexed_points,
            "chunks_emitted": stats.chunks_emitted,
            "elapsed_s": round(stats.elapsed_seconds, 1),
            "note": "EXPERIMENT SUBSET — NOT FULL CORPUS",
        }
        if args.output_json:
            print(json.dumps(result, indent=2))
        else:
            print(
                f"Shard {args.config_lang}/{args.split}/{args.shard}: {stats.indexed_points} points"
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
