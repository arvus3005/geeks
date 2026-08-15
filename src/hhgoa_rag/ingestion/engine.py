"""Resumable ingestion engine for MSMARCO-XI.

Pipeline per shard:
  source rows -> parser -> normalization -> dedup -> chunker
  -> dense embed batch -> sparse embed batch -> Qdrant upsert -> checkpoint

Crash-consistency: checkpoint only after Qdrant acknowledges a batch.
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

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from ..dataset.models import PassageOccurrence
from ..dataset.parser import parse_record
from ..ingestion.checkpoint import IngestCheckpoint, make_schema_fingerprint
from ..ingestion.chunkers import BaseChunker, Chunk, get_chunker
from ..ingestion.dedup import ContentDeduplicator
from ..ingestion.passage_ids import make_point_id
from ..qdrant_lifecycle import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from ..retrieval.embedder import BaseEmbedder
from ..retrieval.sparse_encoder import BM25SparseEncoder

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
    qdrant_points_uploaded: int = 0
    failed_batches: int = 0
    retried_batches: int = 0
    elapsed_seconds: float = 0.0


@dataclass
class IngestionConfig:
    mode: str  # "pilot" | "full"
    physical_collection: str
    chunk_strategy: str = "passage_native"
    chunk_strategy_version: str = "v1"
    dataset_revision: str | None = None
    dense_model_id: str = "intfloat/multilingual-e5-small"
    sparse_model_name: str = "Qdrant/bm25"
    batch_size: int = 64
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

    When num_shards > 1, uses HuggingFace IterableDataset.shard() to divide the
    dataset across parallel workers without downloading the full split per worker.
    row_idx is always relative to the original (un-sharded) dataset so checkpoints
    remain comparable across shard counts.
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
    dense_embedder: BaseEmbedder,
    sparse_encoder: BM25SparseEncoder,
    client: QdrantClient,
    dedup_en: ContentDeduplicator,  # global English dedup (shared across configs)
    dedup_lang: ContentDeduplicator,  # per-language dedup
    run_id: str,
) -> ShardStats:
    """Ingest one shard. Resumes from checkpoint if one exists."""
    stats = ShardStats(config_language=config_language, split=split, shard=shard_idx)
    t0 = time.monotonic()

    schema_fp = make_schema_fingerprint(
        cfg.physical_collection,
        dense_embedder.dimension,
        cfg.sparse_model_name,
        cfg.chunk_strategy_version,
    )
    chunker: BaseChunker = get_chunker(cfg.chunk_strategy)

    ckpt_path = IngestCheckpoint.checkpoint_path(
        cfg.checkpoint_dir, run_id, config_language, split, shard_idx
    )

    # Load existing checkpoint if present
    start_row = 0
    existing_ckpt: IngestCheckpoint | None = None
    if ckpt_path.exists():
        try:
            existing_ckpt = IngestCheckpoint.load(ckpt_path)
            new_ckpt_probe = IngestCheckpoint(
                run_id=run_id,
                dataset_repo=DATASET_REPO,
                dataset_revision=cfg.dataset_revision,
                config_language=config_language,
                split=split,
                source_shard=shard_idx,
                chunk_strategy=cfg.chunk_strategy,
                chunk_strategy_version=cfg.chunk_strategy_version,
                dense_model_id=cfg.dense_model_id,
                sparse_model_name=cfg.sparse_model_name,
                physical_collection=cfg.physical_collection,
                schema_fingerprint=schema_fp,
                last_acknowledged_row=0,
                cumulative_source_rows=0,
                cumulative_valid_occurrences=0,
                cumulative_duplicate_occurrences=0,
                cumulative_rejected_occurrences=0,
                cumulative_chunks_emitted=0,
                cumulative_qdrant_points=0,
                started_at=_now_iso(),
                updated_at=_now_iso(),
                mode=cfg.mode,
            )
            compatible, mismatches = existing_ckpt.is_compatible(new_ckpt_probe)
            if not compatible:
                raise RuntimeError(f"Checkpoint incompatible with current config: {mismatches}")
            if existing_ckpt.status == "complete":
                logger.info(
                    "Shard %s/%s/%d already complete, skipping", config_language, split, shard_idx
                )
                stats.source_rows = existing_ckpt.cumulative_source_rows
                stats.qdrant_points_uploaded = existing_ckpt.cumulative_qdrant_points
                return stats
            start_row = existing_ckpt.last_acknowledged_row + 1
            stats.source_rows = existing_ckpt.cumulative_source_rows
            stats.valid_occurrences = existing_ckpt.cumulative_valid_occurrences
            stats.duplicate_occurrences = existing_ckpt.cumulative_duplicate_occurrences
            stats.rejected_occurrences = existing_ckpt.cumulative_rejected_occurrences
            stats.chunks_emitted = existing_ckpt.cumulative_chunks_emitted
            stats.qdrant_points_uploaded = existing_ckpt.cumulative_qdrant_points
            logger.info(
                "Resuming shard %s/%s/%d from row %d", config_language, split, shard_idx, start_row
            )
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning("Corrupt checkpoint %s, starting fresh: %s", ckpt_path, e)

    max_rows = cfg.pilot_rows_per_shard if cfg.mode == "pilot" else None

    def flush_batch(batch: list[PointStruct], last_row: int) -> None:
        if not batch:
            return
        for attempt in range(cfg.max_retries + 1):
            try:
                client.upsert(collection_name=cfg.physical_collection, points=batch, wait=True)
                stats.qdrant_points_uploaded += len(batch)
                # Save checkpoint after acknowledged upsert
                ckpt = IngestCheckpoint(
                    run_id=run_id,
                    dataset_repo=DATASET_REPO,
                    dataset_revision=cfg.dataset_revision,
                    config_language=config_language,
                    split=split,
                    source_shard=shard_idx,
                    chunk_strategy=cfg.chunk_strategy,
                    chunk_strategy_version=cfg.chunk_strategy_version,
                    dense_model_id=cfg.dense_model_id,
                    sparse_model_name=cfg.sparse_model_name,
                    physical_collection=cfg.physical_collection,
                    schema_fingerprint=schema_fp,
                    last_acknowledged_row=last_row,
                    cumulative_source_rows=stats.source_rows,
                    cumulative_valid_occurrences=stats.valid_occurrences,
                    cumulative_duplicate_occurrences=stats.duplicate_occurrences,
                    cumulative_rejected_occurrences=stats.rejected_occurrences,
                    cumulative_chunks_emitted=stats.chunks_emitted,
                    cumulative_qdrant_points=stats.qdrant_points_uploaded,
                    started_at=existing_ckpt.started_at if existing_ckpt else _now_iso(),
                    updated_at=_now_iso(),
                    mode=cfg.mode,
                )
                ckpt.save(ckpt_path)
                return
            except Exception as e:
                if attempt < cfg.max_retries:
                    wait = cfg.retry_backoff_base**attempt
                    logger.warning(
                        "Batch upsert failed (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1,
                        cfg.max_retries,
                        wait,
                        e,
                    )
                    stats.retried_batches += 1
                    time.sleep(wait)
                else:
                    stats.failed_batches += 1
                    logger.error(
                        "Batch upsert permanently failed after %d retries: %s", cfg.max_retries, e
                    )
                    raise

    # Accumulate normalized texts for batch sparse encoding
    pending_occurrences: list[tuple[PassageOccurrence, Chunk]] = []
    last_row_seen = start_row - 1

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
            # Dedup: English globally, translated per-language
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
                pending_occurrences.append((occ, chunk, point_id))  # type: ignore[arg-type]

        # When batch is full, embed and upsert
        if len(pending_occurrences) >= cfg.batch_size:
            _embed_and_flush(
                pending_occurrences,
                dense_embedder,
                sparse_encoder,
                client,
                cfg,
                stats,
                last_row_seen,
                existing_ckpt,
                ckpt_path,
                schema_fp,
                run_id,
                config_language,
                split,
                shard_idx,
            )
            pending_occurrences = []

    # Flush remaining
    if pending_occurrences:
        _embed_and_flush(
            pending_occurrences,
            dense_embedder,
            sparse_encoder,
            client,
            cfg,
            stats,
            last_row_seen,
            existing_ckpt,
            ckpt_path,
            schema_fp,
            run_id,
            config_language,
            split,
            shard_idx,
        )

    dedup_en.flush()
    dedup_lang.flush()

    # Mark complete
    if ckpt_path.exists():
        final_ckpt = IngestCheckpoint.load(ckpt_path)
        final_ckpt.status = "complete"
        final_ckpt.updated_at = _now_iso()
        final_ckpt.save(ckpt_path)

    stats.elapsed_seconds = time.monotonic() - t0
    return stats


def _embed_and_flush(
    pending: list,
    dense_embedder: BaseEmbedder,
    sparse_encoder: BM25SparseEncoder,
    client: QdrantClient,
    cfg: IngestionConfig,
    stats: ShardStats,
    last_row: int,
    existing_ckpt: IngestCheckpoint | None,
    ckpt_path: Path,
    schema_fp: str,
    run_id: str,
    config_language: str,
    split: str,
    shard_idx: int,
) -> None:
    occ_chunks = [(item[0], item[1], item[2]) for item in pending]
    texts = [chunk.text for _, chunk, _ in occ_chunks]

    dense_vecs = dense_embedder.embed_passages([f"passage: {t}" for t in texts])
    sparse_vecs = sparse_encoder.encode_passages_batch(texts)

    points = []
    for (occ, chunk, point_id), dense_vec, sparse_vec in zip(occ_chunks, dense_vecs, sparse_vecs):
        points.append(
            PointStruct(
                id=point_id,
                vector={
                    DENSE_VECTOR_NAME: dense_vec.tolist(),
                    SPARSE_VECTOR_NAME: sparse_vec,
                },
                payload={
                    "text": chunk.text,
                    "language": occ.passage_language,
                    "content_hash": occ.content_hash,
                    "parent_passage_id": occ.content_hash,
                    "chunk_strategy": cfg.chunk_strategy,
                    "chunk_strategy_version": cfg.chunk_strategy_version,
                    "chunk_ordinal": chunk.chunk_ordinal,
                    "chunk_total": chunk.chunk_total,
                    "source_split": split,
                    "index_manifest_id": f"{cfg.mode}-{run_id[:8]}",
                },
            )
        )

    for attempt in range(cfg.max_retries + 1):
        try:
            client.upsert(collection_name=cfg.physical_collection, points=points, wait=True)
            stats.qdrant_points_uploaded += len(points)
            # Save checkpoint after acknowledged upsert
            ckpt = IngestCheckpoint(
                run_id=run_id,
                dataset_repo=DATASET_REPO,
                dataset_revision=cfg.dataset_revision,
                config_language=config_language,
                split=split,
                source_shard=shard_idx,
                chunk_strategy=cfg.chunk_strategy,
                chunk_strategy_version=cfg.chunk_strategy_version,
                dense_model_id=cfg.dense_model_id,
                sparse_model_name=cfg.sparse_model_name,
                physical_collection=cfg.physical_collection,
                schema_fingerprint=schema_fp,
                last_acknowledged_row=last_row,
                cumulative_source_rows=stats.source_rows,
                cumulative_valid_occurrences=stats.valid_occurrences,
                cumulative_duplicate_occurrences=stats.duplicate_occurrences,
                cumulative_rejected_occurrences=stats.rejected_occurrences,
                cumulative_chunks_emitted=stats.chunks_emitted,
                cumulative_qdrant_points=stats.qdrant_points_uploaded,
                started_at=existing_ckpt.started_at if existing_ckpt else _now_iso(),
                updated_at=_now_iso(),
                mode=cfg.mode,
            )
            ckpt.save(ckpt_path)
            return
        except Exception as e:
            if attempt < cfg.max_retries:
                wait = cfg.retry_backoff_base**attempt
                logger.warning(
                    "Batch upsert attempt %d failed, retrying in %.1fs: %s", attempt + 1, wait, e
                )
                stats.retried_batches += 1
                time.sleep(wait)
            else:
                stats.failed_batches += 1
                logger.error("Batch upsert permanently failed: %s", e)
                raise
