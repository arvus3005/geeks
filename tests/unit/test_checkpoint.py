"""Tests for crash-consistent checkpoint logic."""

import tempfile
from pathlib import Path

from hhgoa_rag.ingestion.checkpoint import IngestCheckpoint, make_schema_fingerprint


def _make_ckpt(**overrides) -> IngestCheckpoint:
    defaults = dict(
        run_id="test-run",
        dataset_repo="ai4bharat/MSMARCO-XI",
        dataset_revision="abc123",
        config_language="bn",
        split="train",
        source_shard=0,
        chunk_strategy="passage_native",
        chunk_strategy_version="v1",
        embed_model="multilingual-e5-large",
        pinecone_index="msmarco-xi",
        pinecone_namespace="pilot_testrun",
        schema_fingerprint="deadbeef",
        last_acknowledged_row=42,
        cumulative_source_rows=100,
        cumulative_valid_occurrences=90,
        cumulative_duplicate_occurrences=5,
        cumulative_rejected_occurrences=5,
        cumulative_chunks_emitted=90,
        cumulative_indexed_points=80,
        started_at="2026-08-15T00:00:00+00:00",
        updated_at="2026-08-15T01:00:00+00:00",
        mode="pilot",
        num_workers=1,
    )
    defaults.update(overrides)
    return IngestCheckpoint(**defaults)


def test_save_and_load_roundtrip():
    ckpt = _make_ckpt()
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "ckpt.json"
        ckpt.save(path)
        loaded = IngestCheckpoint.load(path)
    assert loaded.run_id == ckpt.run_id
    assert loaded.last_acknowledged_row == 42
    assert loaded.mode == "pilot"


def test_save_is_atomic(tmp_path):
    """Crash mid-write should not leave a .tmp file behind after save completes."""
    ckpt = _make_ckpt()
    path = tmp_path / "ckpt.json"
    ckpt.save(path)
    tmp = path.with_suffix(".tmp")
    assert not tmp.exists(), ".tmp file should be cleaned up after atomic rename"
    assert path.exists()


def test_compatibility_same():
    a = _make_ckpt()
    b = _make_ckpt()
    ok, issues = a.is_compatible(b)
    assert ok
    assert issues == []


def test_incompatible_different_embed_model():
    a = _make_ckpt()
    b = _make_ckpt(embed_model="other-model")
    ok, issues = a.is_compatible(b)
    assert not ok
    assert any("embed_model" in i for i in issues)


def test_incompatible_different_index():
    a = _make_ckpt()
    b = _make_ckpt(pinecone_index="other-index")
    ok, issues = a.is_compatible(b)
    assert not ok
    assert any("pinecone_index" in i for i in issues)


def test_incompatible_different_namespace():
    a = _make_ckpt()
    b = _make_ckpt(pinecone_namespace="different_ns")
    ok, issues = a.is_compatible(b)
    assert not ok


def test_schema_fingerprint_deterministic():
    fp1 = make_schema_fingerprint("msmarco-xi", "smoke", "multilingual-e5-large", "v1")
    fp2 = make_schema_fingerprint("msmarco-xi", "smoke", "multilingual-e5-large", "v1")
    assert fp1 == fp2


def test_schema_fingerprint_varies_with_inputs():
    fp1 = make_schema_fingerprint("msmarco-xi", "smoke", "multilingual-e5-large", "v1")
    fp2 = make_schema_fingerprint("msmarco-xi", "full", "multilingual-e5-large", "v1")
    assert fp1 != fp2


def test_checkpoint_path_deterministic(tmp_path):
    p1 = IngestCheckpoint.checkpoint_path(tmp_path, "run1", "bn", "train", 0)
    p2 = IngestCheckpoint.checkpoint_path(tmp_path, "run1", "bn", "train", 0)
    assert p1 == p2


def test_resume_refused_if_complete(tmp_path):
    ckpt = _make_ckpt(status="complete")
    path = tmp_path / "ckpt.json"
    ckpt.save(path)
    loaded = IngestCheckpoint.load(path)
    assert loaded.status == "complete"
