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
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--dedup-db-dir", type=Path, default=Path("artifacts/dedup"))
    p.add_argument("--confirm-full-ingest", action="store_true")
    p.add_argument(
        "--execute",
        action="store_true",
        help="Perform real Pinecone writes (also requires CONFIRM_PINECONE_WRITE=1)",
    )
    p.add_argument("--output-json", action="store_true")
    args = p.parse_args()

    from hhgoa_rag.ingestion.checkpoint import IngestCheckpoint

    try:
        ckpt = IngestCheckpoint.load(args.checkpoint)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # ── Block raw-HF pilot/canary checkpoint resumption ──────────────────────
    if getattr(ckpt, "mode", None) in ("pilot", "canary"):
        print(
            f"ERROR: Resuming raw-HF pilot/canary checkpoints via {__file__} is blocked.\n"
            "Use the prepared-data path for pilot ingestion:\n"
            "  uv run python scripts/ingest_prepared.py "
            "--manifest artifacts/prepared/<id>_manifest.json --execute",
            file=sys.stderr,
        )
        sys.exit(1)

    # Starter full-mode block — BEFORE any other checks
    from hhgoa_rag.ingestion.budget import get_plan

    plan = get_plan()
    if ckpt.mode == "full" and plan == "starter":
        print(
            "ERROR: Full-corpus ingestion is permanently blocked on Pinecone Starter plan.\n"
            f"PINECONE_PLAN={plan!r} is active. Cannot resume a full-mode checkpoint on Starter.",
            file=sys.stderr,
        )
        sys.exit(1)

    if ckpt.mode == "full" and not args.confirm_full_ingest:
        print(
            "ERROR: --confirm-full-ingest required to resume full-mode ingestion.\n"
            "DO NOT RUN UNTIL INFRASTRUCTURE AND COST ARE APPROVED.",
            file=sys.stderr,
        )
        sys.exit(1)

    if ckpt.mode == "full":
        full_env = os.environ.get("CONFIRM_FULL_INGEST", "").strip()
        if full_env != "YES_I_APPROVE_FULL_CORPUS":
            print(
                "ERROR: CONFIRM_FULL_INGEST=YES_I_APPROVE_FULL_CORPUS required to resume full-mode.\n"
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

    from pinecone import Pinecone

    from hhgoa_rag.ingestion.budget import make_default_guard
    from hhgoa_rag.ingestion.dedup import ContentDeduplicator
    from hhgoa_rag.ingestion.engine import IngestionConfig, ingest_shard
    from hhgoa_rag.pinecone_store import PineconeStore

    if ckpt.status == "complete":
        result = {"status": "already_complete", "checkpoint": str(args.checkpoint)}
        if args.output_json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Checkpoint already complete: {args.checkpoint}")
        sys.exit(0)

    api_key = os.environ.get("PINECONE_API_KEY")
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
        num_workers=ckpt.num_workers,
    )

    pc = Pinecone(api_key=api_key)
    store = PineconeStore(pc.Index(ckpt.pinecone_index), embed_model=ckpt.embed_model)
    guard = make_default_guard()

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
            budget_guard=guard,
        )
        resumed_result = {
            "run_id": ckpt.run_id,
            "resumed_from_row": ckpt.last_acknowledged_row,
            "source_rows_total": stats.source_rows,
            "indexed_points_total": stats.indexed_points,
        }
        if args.output_json:
            print(json.dumps(resumed_result, indent=2))
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
