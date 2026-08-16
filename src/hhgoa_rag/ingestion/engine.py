"""Resumable ingestion engine for MSMARCO-XI → Pinecone.

Pipeline per shard:
  source rows → parser → normalisation → dedup → chunker
  → batch records → budget check → Pinecone upsert_records → ack
  → dedup commit → budget commit → checkpoint

Crash-consistency guarantees:
  - Dedup hashes are committed to SQLite ONLY after Pinecone acknowledges the batch.
  - Budget usage is committed ONLY after Pinecone acknowledges the batch.
  - The checkpoint is updated ONLY after dedup and budget are committed.
  - A crash at any point before acknowledgement allows safe replay.
  - Deterministic UUIDs make re-upsert idempotent.

Full ingestion is permanently blocked when PINECONE_PLAN=starter.
Single-worker mode is required for Starter canary/pilot ingestion.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ..dataset.models import PassageOccurrence
from ..dataset.parser import parse_record
from ..ingestion.budget import BudgetGuard, StarterFullModeError, get_plan, make_default_guard
from ..ingestion.checkpoint import IngestCheckpoint, make_schema_fingerprint
from ..ingestion.chunkers import BaseChunker, Chunk, get_chunker
from ..ingestion.dedup import ContentDeduplicator
from ..ingestion.passage_ids import make_point_id
from ..ingestion.schema import build_record, validate_record
from ..pinecone_contract import MAX_BATCH_SIZE, MODEL
from ..pinecone_store import PineconeStore

logger = logging.getLogger(__name__)

DATASET_REPO = "ai4bharat/MSMARCO-XI"
EXPECTED_SPLITS = ["train", "validation"]

# Conservative byte ceiling below Pinecone's 2 MB payload limit
MAX_REQUEST_BYTES = 1_800_000


@dataclass
class ShardStats:
    config_language: str
    split: str
    shard: int
    source_rows: int = 0
    valid_occurrences: int = 0
    duplicate_occurrences: int = 0
    rejected_occurrences: int = 0
    chunks_emitted: int = 0
    indexed_points: int = 0
    failed_batches: int = 0
    retried_batches: int = 0
    elapsed_seconds: float = 0.0


@dataclass
class IngestionConfig:
    mode: str  # "canary" | "pilot" | "full"
    pinecone_index: str
    pinecone_namespace: str
    embed_model: str = MODEL
    chunk_strategy: str = "passage_native"
    chunk_strategy_version: str = "v1"
    dataset_revision: str | None = None
    batch_size: int = MAX_BATCH_SIZE
    max_retries: int = 3
    retry_backoff_base: float = 2.0
    pilot_rows_per_shard: int = 1000
    checkpoint_dir: Path = Path("artifacts/checkpoints")
    dedup_db_dir: Path = Path("artifacts/dedup")
    manifest_dir: Path = Path("artifacts/manifests")
    num_workers: int = 1
    tokenizer_fingerprint: str = "unknown"

    def __post_init__(self) -> None:
        if not (1 <= self.batch_size <= MAX_BATCH_SIZE):
            raise ValueError(
                f"batch_size must be between 1 and {MAX_BATCH_SIZE} (Pinecone limit), got {self.batch_size}"
            )
        if self.num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {self.num_workers}")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _validate_single_worker_for_starter(num_workers: int, mode: str) -> None:
    """Enforce single-worker requirement for starter canary/pilot modes."""
    if mode in ("canary", "pilot") and num_workers != 1:
        raise ValueError(
            f"Starter canary/pilot ingestion requires exactly 1 worker (num_workers=1), "
            f"got {num_workers}. Multi-worker sharding for Starter is not implemented. "
            "Set num_workers=1 to proceed."
        )


def _check_starter_full_mode(mode: str) -> None:
    """Raise StarterFullModeError unconditionally when full mode attempted on Starter."""
    if mode == "full" and get_plan() == "starter":
        raise StarterFullModeError(
            "Full-corpus ingestion is permanently blocked on Pinecone Starter plan. "
            "PINECONE_PLAN=starter (default). No flag or environment variable can override this."
        )


def _stream_shard(
    config_language: str,
    split: str,
    worker_index: int,
    dataset_revision: str | None,
    start_row: int = 0,
    max_rows: int | None = None,
    num_workers: int = 1,
) -> Iterator[tuple[int, dict]]:
    """Stream the deterministic, non-overlapping partition for this worker.

    Uses local_row as the stable position within this worker's partition.
    Worker identity is encoded in make_point_id via the physical_shard field.
    """
    if num_workers != 1:
        raise ValueError(
            f"Multi-worker sharding is not yet safe for Starter ingestion. "
            f"num_workers must be 1, got {num_workers}."
        )

    from datasets import load_dataset

    ds = load_dataset(
        DATASET_REPO,
        config_language,
        split=split,
        streaming=True,
        revision=dataset_revision,
    )

    local_row = 0
    emitted = 0

    for record in ds:
        if local_row < start_row:
            local_row += 1
            continue
        yield local_row, record
        local_row += 1
        emitted += 1
        if max_rows is not None and emitted >= max_rows:
            break


def _estimate_batch_bytes(records: list[dict]) -> int:
    """Estimate serialised UTF-8 bytes of the records list."""
    import json as _json

    return len(_json.dumps(records).encode("utf-8"))


def ingest_shard(
    config_language: str,
    split: str,
    shard_idx: int,
    cfg: IngestionConfig,
    store: PineconeStore,
    dedup_en: ContentDeduplicator,
    dedup_lang: ContentDeduplicator,
    run_id: str,
    num_workers: int | None = None,
    budget_guard: BudgetGuard | None = None,
) -> ShardStats:
    """Ingest one worker partition. Resumes from checkpoint if one exists.

    Crash-consistency lifecycle per batch:
      1. Build records
      2. Check budgets (raises before any API call)
      3. Call Pinecone.upsert_records — retries on transient error
      4. On success: commit dedup hashes → commit budget → save checkpoint
    """
    # Fail closed for Starter full mode before any external call
    _check_starter_full_mode(cfg.mode)

    effective_num_workers = num_workers if num_workers is not None else cfg.num_workers
    _validate_single_worker_for_starter(effective_num_workers, cfg.mode)

    if effective_num_workers != 1:
        raise ValueError(
            f"num_workers must be 1 for this implementation; got {effective_num_workers}"
        )
    if shard_idx != 0:
        raise ValueError(f"shard_idx must be 0 for single-worker ingestion; got {shard_idx}")

    guard = budget_guard or make_default_guard()
    stats = ShardStats(config_language=config_language, split=split, shard=shard_idx)
    t0 = time.monotonic()

    schema_fp = make_schema_fingerprint(
        cfg.pinecone_index,
        cfg.pinecone_namespace,
        cfg.embed_model,
        cfg.chunk_strategy_version,
    )
    chunker: BaseChunker = get_chunker(cfg.chunk_strategy)

    ckpt_path = IngestCheckpoint.checkpoint_path(
        cfg.checkpoint_dir, run_id, config_language, split, shard_idx
    )

    start_row = 0
    existing_ckpt: IngestCheckpoint | None = None
    if ckpt_path.exists():
        try:
            existing_ckpt = IngestCheckpoint.load(ckpt_path)
            probe = IngestCheckpoint(
                run_id=run_id,
                dataset_repo=DATASET_REPO,
                dataset_revision=cfg.dataset_revision,
                config_language=config_language,
                split=split,
                source_shard=shard_idx,
                chunk_strategy=cfg.chunk_strategy,
                chunk_strategy_version=cfg.chunk_strategy_version,
                embed_model=cfg.embed_model,
                pinecone_index=cfg.pinecone_index,
                pinecone_namespace=cfg.pinecone_namespace,
                schema_fingerprint=schema_fp,
                last_acknowledged_row=0,
                cumulative_source_rows=0,
                cumulative_valid_occurrences=0,
                cumulative_duplicate_occurrences=0,
                cumulative_rejected_occurrences=0,
                cumulative_chunks_emitted=0,
                cumulative_indexed_points=0,
                started_at=_now_iso(),
                updated_at=_now_iso(),
                mode=cfg.mode,
                num_workers=effective_num_workers,
            )
            compatible, mismatches = existing_ckpt.is_compatible(probe)
            if not compatible:
                raise RuntimeError(f"Checkpoint incompatible: {mismatches}")
            if existing_ckpt.status == "complete":
                logger.info(
                    "Shard %s/%s/%d already complete, skipping", config_language, split, shard_idx
                )
                stats.source_rows = existing_ckpt.cumulative_source_rows
                stats.indexed_points = existing_ckpt.cumulative_indexed_points
                return stats
            start_row = existing_ckpt.last_acknowledged_row + 1
            stats.source_rows = existing_ckpt.cumulative_source_rows
            stats.valid_occurrences = existing_ckpt.cumulative_valid_occurrences
            stats.duplicate_occurrences = existing_ckpt.cumulative_duplicate_occurrences
            stats.rejected_occurrences = existing_ckpt.cumulative_rejected_occurrences
            stats.chunks_emitted = existing_ckpt.cumulative_chunks_emitted
            stats.indexed_points = existing_ckpt.cumulative_indexed_points
            logger.info(
                "Resuming shard %s/%s/%d from row %d", config_language, split, shard_idx, start_row
            )
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning("Corrupt checkpoint %s, starting fresh: %s", ckpt_path, e)

    max_rows = cfg.pilot_rows_per_shard if cfg.mode in ("pilot", "canary") else None

    # pending: list of (occ, chunk, point_id, token_length)
    pending: list[tuple[PassageOccurrence, Chunk, str, int]] = []
    last_row_seen = start_row - 1

    def _save_ckpt(last_row: int) -> None:
        ckpt = IngestCheckpoint(
            run_id=run_id,
            dataset_repo=DATASET_REPO,
            dataset_revision=cfg.dataset_revision,
            config_language=config_language,
            split=split,
            source_shard=shard_idx,
            chunk_strategy=cfg.chunk_strategy,
            chunk_strategy_version=cfg.chunk_strategy_version,
            embed_model=cfg.embed_model,
            pinecone_index=cfg.pinecone_index,
            pinecone_namespace=cfg.pinecone_namespace,
            schema_fingerprint=schema_fp,
            last_acknowledged_row=last_row,
            cumulative_source_rows=stats.source_rows,
            cumulative_valid_occurrences=stats.valid_occurrences,
            cumulative_duplicate_occurrences=stats.duplicate_occurrences,
            cumulative_rejected_occurrences=stats.rejected_occurrences,
            cumulative_chunks_emitted=stats.chunks_emitted,
            cumulative_indexed_points=stats.indexed_points,
            started_at=existing_ckpt.started_at if existing_ckpt else _now_iso(),
            updated_at=_now_iso(),
            mode=cfg.mode,
            num_workers=effective_num_workers,
        )
        ckpt.save(ckpt_path)

    def flush(last_row: int) -> None:
        nonlocal pending
        if not pending:
            return

        records = []
        for occ, chunk, point_id, token_length in pending:
            rec = build_record(
                record_id=point_id,
                chunk_text=chunk.text,
                language=occ.passage_language,
                config_language=occ.config_language,
                dataset_revision=cfg.dataset_revision or "unknown",
                split=split,
                physical_shard=str(shard_idx),
                local_source_row=occ.source_row,
                passage_position=occ.passage_position,
                parent_passage_id=occ.content_hash,
                content_hash=occ.content_hash,
                chunk_strategy=cfg.chunk_strategy,
                chunk_strategy_version=cfg.chunk_strategy_version,
                chunk_ordinal=chunk.chunk_ordinal,
                chunk_total=chunk.chunk_total,
                token_length=token_length,
                tokenizer_fingerprint=cfg.tokenizer_fingerprint,
                manifest_id=f"{cfg.mode}-{run_id[:8]}",
            )
            validate_record(rec)
            records.append(rec)

        # Enforce per-request limits before any provider call
        if len(records) > MAX_BATCH_SIZE:
            raise RuntimeError(
                f"Batch has {len(records)} records, exceeding Pinecone's {MAX_BATCH_SIZE}-record limit. "
                "This is a bug in the batch accumulation logic."
            )
        req_bytes = _estimate_batch_bytes(records)
        if req_bytes > MAX_REQUEST_BYTES:
            raise RuntimeError(
                f"Serialised request size {req_bytes} bytes exceeds safe ceiling "
                f"{MAX_REQUEST_BYTES} bytes."
            )

        # Aggregate token count for budget check
        total_tokens = sum(tok for _, _, _, tok in pending)
        primary_lang = pending[0][0].passage_language if pending else config_language

        # Budget check BEFORE provider call (includes retries)
        guard.check_upsert(
            language=primary_lang,
            record_count=len(records),
            token_count=total_tokens,
            byte_count=req_bytes,
        )

        for attempt in range(cfg.max_retries + 1):
            try:
                store.upsert_records(records, namespace=cfg.pinecone_namespace, context=cfg.mode)
                break
            except Exception as e:
                if attempt < cfg.max_retries:
                    wait = cfg.retry_backoff_base**attempt
                    logger.warning(
                        "Upsert attempt %d failed, retrying in %.1fs: %s", attempt + 1, wait, e
                    )
                    stats.retried_batches += 1
                    time.sleep(wait)
                else:
                    stats.failed_batches += 1
                    logger.error("Upsert permanently failed: %s", e)
                    raise

        # ── ONLY after successful Pinecone acknowledgement ──────────────────
        stats.indexed_points += len(records)

        # Commit dedup hashes to SQLite (in-memory buffer → DB)
        for occ, chunk, point_id, _ in pending:
            dedup = dedup_en if occ.is_original_english else dedup_lang
            dedup.mark_seen(occ.content_hash, point_id)
        dedup_en.flush()
        dedup_lang.flush()

        # Commit budget usage
        guard.commit_upsert(
            language=primary_lang,
            record_count=len(records),
            token_count=total_tokens,
            byte_count=req_bytes,
        )

        # Save checkpoint (after dedup + budget committed)
        _save_ckpt(last_row)

    for row_idx, record in _stream_shard(
        config_language,
        split,
        shard_idx,
        cfg.dataset_revision,
        start_row,
        max_rows,
        num_workers=effective_num_workers,
    ):
        stats.source_rows += 1
        last_row_seen = row_idx

        occurrences, rejected = parse_record(
            record,
            config_language=config_language,
            split=split,
            source_shard=str(shard_idx),
            source_row=row_idx,
            dataset_revision=cfg.dataset_revision or "unknown",
        )
        stats.rejected_occurrences += len(rejected)

        for occ in occurrences:
            stats.valid_occurrences += 1
            dedup = dedup_en if occ.is_original_english else dedup_lang
            if dedup.is_duplicate(occ.content_hash):
                stats.duplicate_occurrences += 1
                continue
            # Mark in dedup's pending buffer (not DB) to prevent intra-run duplicates.
            # DB commit happens after Pinecone ack in flush().
            dedup.mark_seen(occ.content_hash, occ.content_hash)

            chunks = chunker.chunk(occ.normalized_text, occ.content_hash)
            for chunk in chunks:
                stats.chunks_emitted += 1
                point_id = make_point_id(
                    dataset_revision=cfg.dataset_revision or "unknown",
                    language=occ.passage_language,
                    content_hash=occ.content_hash,
                    chunk_strategy_version=f"{cfg.chunk_strategy}_{cfg.chunk_strategy_version}",
                    chunk_ordinal=chunk.chunk_ordinal,
                )
                # Token length: use chunk char length as approximate if tokenizer not set
                token_length = (
                    chunk.token_length
                    if chunk.token_length is not None
                    else max(1, len(chunk.text) // 4)
                )
                pending.append((occ, chunk, point_id, token_length))

                # Flush immediately when batch is full (ensures no batch exceeds 96)
                if len(pending) >= cfg.batch_size:
                    flush(last_row_seen)
                    pending.clear()

    if pending:
        flush(last_row_seen)
        pending.clear()

    if ckpt_path.exists():
        final_ckpt = IngestCheckpoint.load(ckpt_path)
        final_ckpt.status = "complete"
        final_ckpt.updated_at = _now_iso()
        final_ckpt.save(ckpt_path)

    stats.elapsed_seconds = time.monotonic() - t0
    return stats
