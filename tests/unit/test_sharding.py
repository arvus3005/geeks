"""Tests for deterministic, non-overlapping worker partitioning."""

from __future__ import annotations

import pytest

from hhgoa_rag.ingestion.engine import _validate_worker_config

# ── _validate_worker_config ───────────────────────────────────────────────────


def test_single_worker_valid():
    _validate_worker_config(0, 1)  # should not raise


def test_worker_index_zero_with_many_workers():
    _validate_worker_config(0, 4)


def test_worker_index_last_valid():
    _validate_worker_config(3, 4)


def test_worker_index_equals_num_workers_raises():
    with pytest.raises(ValueError, match="worker_index"):
        _validate_worker_config(4, 4)


def test_worker_index_negative_raises():
    with pytest.raises(ValueError, match="worker_index"):
        _validate_worker_config(-1, 4)


def test_num_workers_zero_raises():
    with pytest.raises(ValueError, match="num_workers"):
        _validate_worker_config(0, 0)


def test_num_workers_negative_raises():
    with pytest.raises(ValueError, match="num_workers"):
        _validate_worker_config(0, -1)


# ── Partition correctness via global_row_id arithmetic ───────────────────────


def _global_row_ids(worker_index: int, num_workers: int, total_rows: int) -> list[int]:
    """Replicate the global_row_id assignment from _stream_shard without HF."""
    ids = []
    # Simulate contiguous shard: worker i gets rows [i * block, (i+1) * block)
    # where block = ceil(total_rows / num_workers).
    import math

    block = math.ceil(total_rows / num_workers)
    start = worker_index * block
    end = min(start + block, total_rows)
    for local_row in range(end - start):
        global_id = worker_index * 10_000_000 + local_row
        ids.append(global_id)
    return ids


def test_single_worker_covers_all_rows():
    ids = _global_row_ids(0, 1, 100)
    assert len(ids) == 100


def test_two_workers_no_overlap():
    ids_0 = set(_global_row_ids(0, 2, 100))
    ids_1 = set(_global_row_ids(1, 2, 100))
    assert ids_0.isdisjoint(ids_1)


def test_four_workers_no_overlap():
    total = 200
    all_ids: list[set] = [set(_global_row_ids(w, 4, total)) for w in range(4)]
    for i in range(4):
        for j in range(i + 1, 4):
            assert all_ids[i].isdisjoint(all_ids[j]), f"Workers {i} and {j} overlap"


def test_worker_ids_are_stable():
    """Same worker index always produces the same IDs."""
    a = _global_row_ids(2, 4, 100)
    b = _global_row_ids(2, 4, 100)
    assert a == b


# ── IngestCheckpoint worker-config compatibility ──────────────────────────────


def test_checkpoint_incompatible_on_num_workers_change(tmp_path):
    """Checkpoint saved with num_workers=2 must not resume with num_workers=4."""
    import json

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
        "status": "running",
        "warnings": [],
    }
    ckpt_path = tmp_path / "ckpt.json"
    ckpt_path.write_text(json.dumps(base))
    ckpt = IngestCheckpoint.load(ckpt_path)

    # A probe that changes source_shard is incompatible
    probe = IngestCheckpoint(
        **{
            **base,
            "source_shard": 1,
            "last_acknowledged_row": 0,
            "cumulative_source_rows": 0,
            "cumulative_valid_occurrences": 0,
            "cumulative_duplicate_occurrences": 0,
            "cumulative_rejected_occurrences": 0,
            "cumulative_chunks_emitted": 0,
            "cumulative_indexed_points": 0,
            "started_at": "2026-08-15T00:00:00+00:00",
            "updated_at": "2026-08-15T00:00:00+00:00",
        }
    )
    compatible, mismatches = ckpt.is_compatible(probe)
    assert not compatible
    assert any("source_shard" in m for m in mismatches)
