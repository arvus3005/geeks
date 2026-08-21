#!/usr/bin/env python3
"""Ingest all passages from MSMARCO-XI into Pinecone.

DEFAULT: offline preparation only — normalise, dedup, chunk, build manifests.
No Pinecone writes occur unless both --execute and CONFIRM_PINECONE_WRITE=1 are set.

Modes:
  smoke  — deterministic local fixtures, no HuggingFace network needed (default)
  pilot  — bounded real-data subset from HuggingFace (NOT FULL CORPUS)
  full   — entire dataset; blocked permanently on Starter plan regardless of flags.
           Also requires --confirm-full-ingest and
           CONFIRM_FULL_INGEST=YES_I_APPROVE_FULL_CORPUS on non-Starter.
           DO NOT RUN UNTIL INFRASTRUCTURE AND COST ARE APPROVED.

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

from hhgoa_rag.pinecone_contract import INDEX_NAME, MAX_BATCH_SIZE, MODEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_FULL_INGEST_ENV = "CONFIRM_FULL_INGEST"
_FULL_INGEST_VALUE = "YES_I_APPROVE_FULL_CORPUS"
_WRITE_ENV = "CONFIRM_PINECONE_WRITE"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ingest MSMARCO-XI passages into Pinecone",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--mode", choices=["smoke", "pilot", "full"], default="smoke")
    p.add_argument("--config", default="configs/smoke.yaml")
    p.add_argument("--pinecone-index", default=None)
    p.add_argument("--pinecone-namespace", default=None)
    p.add_argument("--chunk-strategy", default="passage_native")
    p.add_argument("--dataset-revision", default=None)
    p.add_argument("--pilot-rows-per-shard", type=int, default=1000)
    p.add_argument(
        "--batch-size",
        type=int,
        default=MAX_BATCH_SIZE,
        help=f"Records per Pinecone request (1-{MAX_BATCH_SIZE}; Pinecone hard limit is {MAX_BATCH_SIZE})",
    )
    p.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/checkpoints"))
    p.add_argument("--dedup-db-dir", type=Path, default=Path("artifacts/dedup"))
    p.add_argument("--run-id", default=None)
    p.add_argument(
        "--confirm-full-ingest",
        action="store_true",
        help="Required CLI acknowledgement for full mode on non-Starter (also needs env var)",
    )
    p.add_argument(
        "--execute",
        action="store_true",
        help="Perform real Pinecone writes (also requires CONFIRM_PINECONE_WRITE=1)",
    )
    p.add_argument("--output-json", action="store_true")
    return p


def _check_write_guards(mode: str, args: argparse.Namespace) -> bool:
    """Return True when writes are authorised. Print reason and return False otherwise."""
    write_confirmed = os.environ.get(_WRITE_ENV, "").strip() == "1"

    if mode == "full":
        full_confirmed = (
            args.confirm_full_ingest
            and os.environ.get(_FULL_INGEST_ENV, "").strip() == _FULL_INGEST_VALUE
        )
        if not full_confirmed:
            missing = []
            if not args.confirm_full_ingest:
                missing.append("--confirm-full-ingest flag")
            if os.environ.get(_FULL_INGEST_ENV, "").strip() != _FULL_INGEST_VALUE:
                missing.append(f"{_FULL_INGEST_ENV}={_FULL_INGEST_VALUE}")
            print(
                f"ERROR: Full mode refused. Missing: {', '.join(missing)}\n"
                "DO NOT RUN UNTIL INFRASTRUCTURE AND COST ARE APPROVED.\n"
                "This script targets the retired Pinecone pilot pipeline.",
                file=sys.stderr,
            )
            return False

    if not args.execute or not write_confirmed:
        missing = []
        if not args.execute:
            missing.append("--execute flag")
        if not write_confirmed:
            missing.append(f"{_WRITE_ENV}=1 env var")
        print(
            f"DRY-RUN: no Pinecone writes. Missing: {', '.join(missing)}",
            file=sys.stderr,
        )
        return False

    return True


def main() -> None:
    args = _build_parser().parse_args()

    # ── Block raw-HF pilot/canary ingestion — redirect to ingest_prepared.py ──
    if args.mode in ("pilot", "canary"):
        print(
            f"ERROR: Raw HuggingFace pilot/canary ingestion via {__file__} is blocked.\n"
            "Use the prepared-data path instead:\n"
            "  uv run python scripts/prepare_canary.py --dataset-revision <sha> "
            "--tokenizer-revision <sha>\n"
            "  uv run python scripts/ingest_prepared.py "
            "--manifest artifacts/prepared/<id>_manifest.json --dry-run\n"
            "  CONFIRM_PINECONE_WRITE=1 PINECONE_API_KEY=... \\\n"
            "    uv run python scripts/index_canary.py "
            "--manifest artifacts/prepared/<id>_manifest.json --execute --resume",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Validate batch size before any other processing ───────────────────────
    if not (1 <= args.batch_size <= MAX_BATCH_SIZE):
        print(
            f"ERROR: --batch-size must be between 1 and {MAX_BATCH_SIZE} (Pinecone limit), got {args.batch_size}",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Starter full-mode block — BEFORE API key, BEFORE HF data, BEFORE client ──
    from hhgoa_rag.ingestion.budget import get_plan

    plan = get_plan()
    if args.mode == "full" and plan == "starter":
        print(
            "ERROR: Full-corpus ingestion is permanently blocked on Pinecone Starter plan.\n"
            f"PINECONE_PLAN={plan!r} is active. No flag or environment variable can override this.\n"
            "Switch to a paid Pinecone plan to enable full-corpus ingestion.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Full mode non-Starter: requires CLI flag + env var ────────────────────
    if args.mode == "full":
        if not args.confirm_full_ingest:
            print(
                "ERROR: --confirm-full-ingest is required for full mode.\n"
                "DO NOT RUN UNTIL INFRASTRUCTURE AND COST ARE APPROVED.\n"
                "This script targets the retired Pinecone pilot pipeline.",
                file=sys.stderr,
            )
            sys.exit(1)
        full_env = os.environ.get(_FULL_INGEST_ENV, "").strip()
        if full_env != _FULL_INGEST_VALUE:
            print(
                f"ERROR: {_FULL_INGEST_ENV}={_FULL_INGEST_VALUE!r} required for full mode.\n"
                "DO NOT RUN UNTIL INFRASTRUCTURE AND COST ARE APPROVED.",
                file=sys.stderr,
            )
            sys.exit(1)

    write_ok = _check_write_guards(args.mode, args)

    if not write_ok:
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "status": "dry_run",
                    "plan": plan,
                    "note": "offline preparation only — no Pinecone writes",
                },
                indent=2,
            )
        )
        sys.exit(0)

    # ── Write path — API key required ─────────────────────────────────────────
    import yaml  # type: ignore[import-untyped]

    cfg: dict = {}
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}

    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        print("ERROR: PINECONE_API_KEY must be set as an environment variable", file=sys.stderr)
        sys.exit(1)

    index_name = args.pinecone_index or cfg.get("pinecone_index", INDEX_NAME)
    embed_model = cfg.get("pinecone_embed_model", MODEL)

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

    # ── Pilot mode ────────────────────────────────────────────────────────────
    print(
        f"WARNING: {args.mode.upper()} MODE — EXPERIMENT SUBSET — NOT FULL CORPUS",
        file=sys.stderr,
    )

    from pinecone import Pinecone

    from hhgoa_rag.dataset.models import INDIC_LANGUAGE_CODES
    from hhgoa_rag.ingestion.budget import make_default_guard
    from hhgoa_rag.ingestion.dedup import ContentDeduplicator
    from hhgoa_rag.ingestion.engine import IngestionConfig, ingest_shard
    from hhgoa_rag.pinecone_store import PILOT_NAMESPACE_PREFIX, PineconeStore

    run_id = args.run_id or str(uuid.uuid4())[:8]
    namespace = args.pinecone_namespace or f"{PILOT_NAMESPACE_PREFIX}{run_id}"

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
        num_workers=1,
    )
    ingest_cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ingest_cfg.dedup_db_dir.mkdir(parents=True, exist_ok=True)

    pc = Pinecone(api_key=api_key)
    store = PineconeStore(pc.Index(index_name), embed_model=embed_model)
    guard = make_default_guard()

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
                    budget_guard=guard,
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
                        "indexed_points": stats.indexed_points,
                        "elapsed_s": round(stats.elapsed_seconds, 1),
                    }
                )
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
        "note": "EXPERIMENT SUBSET — NOT FULL CORPUS",
    }

    if args.output_json:
        print(json.dumps(summary, indent=2))
    else:
        total = 0
        for s in all_stats:
            if isinstance(s, dict) and "indexed_points" in s:
                val = s["indexed_points"]
                if isinstance(val, int | float):
                    total += int(val)
        print(f"Run {run_id}: {total} points in index='{index_name}' ns='{namespace}'")
        print("NOTE: EXPERIMENT SUBSET — NOT FULL CORPUS")

    sys.exit(0)


if __name__ == "__main__":
    main()
