"""Resumable ingestion engine for MSMARCO-XI → Pinecone.

Pipeline per shard:
  source rows → parser → normalization → dedup → chunker
  → batch records → Pinecone upsert_records → checkpoint

Pinecone handles server-side embedding (integrated multilingual-e5-large);
no local dense/sparse encoder is involved.

Crash-consistency: checkpoint only after Pinecone acknowledges a batch.
Replay is safe because point IDs are deterministic (UUIDv5).

Full ingestion is gated by --confirm-full-ingest and must not be started
in a development session. See docs/INGESTION_RUNBOOK.md.
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
from ..ingestion.checkpoint import IngestCheckpoint, make_schema_fingerprint
from ..ingestion.chunkers import BaseChunker, Chunk, get_chunker
from ..ingestion.dedup import ContentDeduplicator
from ..ingestion.passage_ids import make_point_id
from ..pinecone_store import TEXT_RECORD_FIELD, PineconeStore

logger = logging.getLogger(__name__)

DATASET_REPO = "ai4bharat/MSMARCO-XI"
EXPECTED_SPLITS = ["train", "validation"]


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
    mode: str  # "pilot" | "full"
    pinecone_index: str
    pinecone_namespace: str
    embed_model: str = "multilingual-e5-large"
    chunk_strategy: str = "passage_native"
    chunk_strategy_version: str = "v1"
    dataset_revision: str | None = None
    batch_size: int = 96  # Pinecone upsert_records supports up to 96 records per batch
    max_retries: int = 3
    retry_backoff_base: float = 2.0
    pilot_rows_per_shard: int = 1000  # rows per language/split in pilot mode
    checkpoint_dir: Path = Path("artifacts/checkpoints")
    dedup_db_dir: Path = Path("artifacts/dedup")
    manifest_dir: Path = Path("artifacts/manifests")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _stream_shard(
    config_language: str,
    split: str,
    shard_idx: int,
    dataset_revision: str | None,
    start_row: int = 0,
    max_rows: int | None = None,
    num_shards: int = 1,
) -> Iterator[tuple[int, dict]]:
    """Stream rows from one logical shard of MSMARCO-XI.

    When num_shards > 1, uses HuggingFace IterableDataset.shard() to divide
    the dataset across parallel workers without downloading the full split per
    worker. row_idx is always relative to the original (un-sharded) dataset so
    checkpoints remain comparable across shard counts.
    """
    from datasets import load_dataset

    ds = load_dataset(
        DATASET_REPO,
        config_language,
        split=split,
        streaming=True,
        revision=dataset_revision,
    )

    if num_shards > 1:
        ds = ds.shard(num_shards=num_shards, index=shard_idx, contiguous=True)

    row_idx = shard_idx  # first global row index for this shard
    step = num_shards if num_shards > 1 else 1
    emitted = 0

    for record in ds:
        if row_idx < start_row:
            row_idx += step
            continue
        yield row_idx, record
        row_idx += step
        emitted += 1
        if max_rows is not None and emitted >= max_rows:
            break


def ingest_shard(
    config_language: str,
    split: str,
    shard_idx: int,
    cfg: IngestionConfig,
    store: PineconeStore,
    dedup_en: ContentDeduplicator,  # global English dedup (shared across configs)
    dedup_lang: ContentDeduplicator,  # per-language dedup
    run_id: str,
) -> ShardStats:
    """Ingest one shard. Resumes from checkpoint if one exists."""
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
            )
            compatible, mismatches = existing_ckpt.is_compatible(probe)
            if not compatible:
                raise RuntimeError(f"Checkpoint incompatible: {mismatches}")
            if existing_ckpt.status == "complete":
                logger.info("Shard %s/%s/%d already complete, skipping", config_language, split, shard_idx)
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
            logger.info("Resuming shard %s/%s/%d from row %d", config_language, split, shard_idx, start_row)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning("Corrupt checkpoint %s, starting fresh: %s", ckpt_path, e)

    max_rows = cfg.pilot_rows_per_shard if cfg.mode == "pilot" else None

    pending: list[tuple[PassageOccurrence, Chunk, str]] = []
    last_row_seen = start_row - 1

    def flush(last_row: int) -> None:
        if not pending:
            return
        records = []
        for occ, chunk, point_id in pending:
            records.append(
                {
                    "id": point_id,
                    TEXT_RECORD_FIELD: chunk.text,
                    "language": occ.passage_language,
                    "content_hash": occ.content_hash,
                    "chunk_strategy": cfg.chunk_strategy,
                    "chunk_strategy_version": cfg.chunk_strategy_version,
                    "chunk_ordinal": chunk.chunk_ordinal,
                    "source_split": split,
                    "index_manifest_id": f"{cfg.mode}-{run_id[:8]}",
                }
            )

        for attempt in range(cfg.max_retries + 1):
            try:
                store.upsert_records(records, namespace=cfg.pinecone_namespace, context=cfg.mode)
                stats.indexed_points += len(records)
                _save_ckpt(last_row)
                return
            except Exception as e:
                if attempt < cfg.max_retries:
                    wait = cfg.retry_backoff_base**attempt
                    logger.warning("Upsert attempt %d failed, retrying in %.1fs: %s", attempt + 1, wait, e)
                    stats.retried_batches += 1
                    time.sleep(wait)
                else:
                    stats.failed_batches += 1
                    logger.error("Upsert permanently failed: %s", e)
                    raise

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
        )
        ckpt.save(ckpt_path)

    for row_idx, record in _stream_shard(
        config_language, split, shard_idx, cfg.dataset_revision, start_row, max_rows
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
                dedup.mark_seen(occ.content_hash, occ.content_hash)
                continue
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
                pending.append((occ, chunk, point_id))

        if len(pending) >= cfg.batch_size:
            flush(last_row_seen)
            pending.clear()

    if pending:
        flush(last_row_seen)
        pending.clear()

    dedup_en.flush()
    dedup_lang.flush()

    if ckpt_path.exists():
        final_ckpt = IngestCheckpoint.load(ckpt_path)
        final_ckpt.status = "complete"
        final_ckpt.updated_at = _now_iso()
        final_ckpt.save(ckpt_path)

    stats.elapsed_seconds = time.monotonic() - t0
    return stats
