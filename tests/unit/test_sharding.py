"""Tests for single-worker mode enforcement and checkpoint compatibility."""

from __future__ import annotations

import json

import pytest

from hhgoa_rag.ingestion.engine import IngestionConfig, _validate_single_worker_for_starter

# ── Single-worker enforcement for Starter ─────────────────────────────────────


def test_single_worker_valid():
    _validate_single_worker_for_starter(1, "pilot")  # should not raise


def test_canary_mode_single_worker_valid():
    _validate_single_worker_for_starter(1, "canary")


def test_pilot_multi_worker_raises():
    with pytest.raises(ValueError, match="num_workers=1"):
        _validate_single_worker_for_starter(2, "pilot")


def test_canary_multi_worker_raises():
    with pytest.raises(ValueError, match="num_workers=1"):
        _validate_single_worker_for_starter(4, "canary")


def test_full_mode_multi_worker_not_restricted_by_this_validator():
    """Full mode multi-worker is blocked elsewhere (StarterFullModeError), not here."""
    _validate_single_worker_for_starter(4, "full")  # no raise from this function


# ── IngestionConfig batch_size validation ─────────────────────────────────────


def test_ingestion_config_batch_size_1_valid():
    cfg = IngestionConfig(
        mode="pilot", pinecone_index="x", pinecone_namespace="pilot_x", batch_size=1
    )
    assert cfg.batch_size == 1


def test_ingestion_config_batch_size_96_valid():
    cfg = IngestionConfig(
        mode="pilot", pinecone_index="x", pinecone_namespace="pilot_x", batch_size=96
    )
    assert cfg.batch_size == 96


def test_ingestion_config_batch_size_97_raises():
    with pytest.raises(ValueError, match="batch_size"):
        IngestionConfig(
            mode="pilot", pinecone_index="x", pinecone_namespace="pilot_x", batch_size=97
        )


def test_ingestion_config_batch_size_0_raises():
    with pytest.raises(ValueError, match="batch_size"):
        IngestionConfig(
            mode="pilot", pinecone_index="x", pinecone_namespace="pilot_x", batch_size=0
        )


def test_ingestion_config_num_workers_0_raises():
    with pytest.raises(ValueError, match="num_workers"):
        IngestionConfig(
            mode="pilot",
            pinecone_index="x",
            pinecone_namespace="pilot_x",
            num_workers=0,
        )


# ── IngestCheckpoint num_workers compatibility ────────────────────────────────


def test_checkpoint_incompatible_on_num_workers_change(tmp_path):
    """Checkpoint saved with num_workers=1 must not resume with num_workers=2."""
    from hhgoa_rag.ingestion.checkpoint import IngestCheckpoint

    base = {
        "run_id": "test-run",
        "dataset_repo": "ai4bharat/MSMARCO-XI",
        "dataset_revision": None,
        "config_language": "bn",
        "split": "train",
        "source_shard": 0,
        "chunk_strategy": "passage_native",
        "chunk_strategy_version": "v1",
        "embed_model": "multilingual-e5-large",
        "pinecone_index": "msmarco-xi",
        "pinecone_namespace": "pilot_test",
        "schema_fingerprint": "fp_abc",
        "last_acknowledged_row": 999,
        "cumulative_source_rows": 1000,
        "cumulative_valid_occurrences": 950,
        "cumulative_duplicate_occurrences": 50,
        "cumulative_rejected_occurrences": 0,
        "cumulative_chunks_emitted": 950,
        "cumulative_indexed_points": 950,
        "started_at": "2026-08-15T00:00:00+00:00",
        "updated_at": "2026-08-15T01:00:00+00:00",
        "mode": "pilot",
        "num_workers": 1,
        "status": "running",
        "warnings": [],
    }
    ckpt_path = tmp_path / "ckpt.json"
    ckpt_path.write_text(json.dumps(base))
    ckpt = IngestCheckpoint.load(ckpt_path)

    # num_workers change must be detected as incompatible
    probe = IngestCheckpoint(
        **{
            **base,
            "num_workers": 2,
            "last_acknowledged_row": 0,
            "cumulative_source_rows": 0,
            "cumulative_valid_occurrences": 0,
            "cumulative_duplicate_occurrences": 0,
            "cumulative_rejected_occurrences": 0,
            "cumulative_chunks_emitted": 0,
            "cumulative_indexed_points": 0,
        }
    )
    compatible, mismatches = ckpt.is_compatible(probe)
    assert not compatible
    assert any("num_workers" in m for m in mismatches)


def test_checkpoint_fails_closed_on_missing_num_workers(tmp_path):
    """Old checkpoint without num_workers must be rejected, not silently accepted."""
    from hhgoa_rag.ingestion.checkpoint import IngestCheckpoint

    old_ckpt = {
        "run_id": "old-run",
        "dataset_repo": "ai4bharat/MSMARCO-XI",
        "dataset_revision": None,
        "config_language": "bn",
        "split": "train",
        "source_shard": 0,
        "chunk_strategy": "passage_native",
        "chunk_strategy_version": "v1",
        "embed_model": "multilingual-e5-large",
        "pinecone_index": "msmarco-xi",
        "pinecone_namespace": "pilot_test",
        "schema_fingerprint": "fp_abc",
        "last_acknowledged_row": 500,
        "cumulative_source_rows": 500,
        "cumulative_valid_occurrences": 490,
        "cumulative_duplicate_occurrences": 10,
        "cumulative_rejected_occurrences": 0,
        "cumulative_chunks_emitted": 490,
        "cumulative_indexed_points": 490,
        "started_at": "2026-08-15T00:00:00+00:00",
        "updated_at": "2026-08-15T01:00:00+00:00",
        "mode": "pilot",
        # num_workers intentionally absent (old format)
        "status": "running",
        "warnings": [],
    }
    ckpt_path = tmp_path / "old.json"
    ckpt_path.write_text(json.dumps(old_ckpt))

    with pytest.raises(RuntimeError, match="num_workers"):
        IngestCheckpoint.load(ckpt_path)
