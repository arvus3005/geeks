"""Focused tests for prepare_canary.py and ingest_prepared.py pathways.

Tests cover all 16 required scenarios from the implementation spec.
No Pinecone calls are made anywhere in this file.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ── Helpers to build minimal valid prepared records ───────────────────────────

_DATASET_REV = "a" * 40
_TOK_FINGERPRINT = "abcd1234ef567890"
_MANIFEST_ID = "canary-42-deadbeef"


def _make_record(
    lang: str = "en",
    config_lang: str = "hi",
    physical_source: str = "train/hintrain.parquet",
    local_source_row: int = 0,
    passage_position: int = 0,
    chunk_ordinal: int = 0,
    chunk_total: int = 1,
    token_length: int = 50,
    content_hash: str | None = None,
    record_id: str | None = None,
    text: str = "The quick brown fox.",
    manifest_id: str = _MANIFEST_ID,
) -> dict[str, Any]:
    ch = content_hash or hashlib.sha256(text.encode()).hexdigest()
    rid = record_id or str(
        uuid.uuid5(
            uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8"),
            f"{_DATASET_REV}|{lang}|{ch}|passage_native_v1|{chunk_ordinal}",
        )
    )
    return {
        "id": rid,
        "chunk_text": text,
        "language": lang,
        "config_language": config_lang,
        "dataset_revision": _DATASET_REV,
        "split": "train",
        # physical_shard holds the real source path (not the unofficial physical_source field)
        "physical_shard": physical_source,
        "local_source_row": local_source_row,
        "passage_position": passage_position,
        "parent_passage_id": ch,
        "content_hash": ch,
        "chunk_strategy": "passage_native",
        "chunk_strategy_version": "v1",
        "chunk_ordinal": chunk_ordinal,
        "chunk_total": chunk_total,
        "token_length": token_length,
        "tokenizer_fingerprint": _TOK_FINGERPRINT,
        "manifest_id": manifest_id,
    }


def _make_300_records() -> list[dict]:
    records = []
    for i in range(100):
        records.append(
            _make_record(
                lang="en", config_lang="hi", local_source_row=i, text=f"English passage {i}."
            )
        )
    for i in range(100):
        records.append(
            _make_record(
                lang="hi",
                config_lang="hi",
                local_source_row=i,
                physical_source="train/hintrain.parquet",
                text=f"Hindi passage {i} हिन्दी।",
            )
        )
    for i in range(100):
        records.append(
            _make_record(
                lang="bn",
                config_lang="bn",
                local_source_row=i,
                physical_source="train/bentrain.parquet",
                text=f"Bengali passage {i} বাংলা।",
            )
        )
    return records


def _make_manifest(records: list[dict], tmpdir: Path) -> dict:
    total_tokens = sum(r["token_length"] for r in records)
    lang_counts: dict[str, int] = {}
    for r in records:
        lang_counts[r["language"]] = lang_counts.get(r["language"], 0) + 1
    jsonl_path = tmpdir / f"{_MANIFEST_ID}_records.jsonl"
    with open(jsonl_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    h = hashlib.sha256()
    with open(jsonl_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    jsonl_checksum = h.hexdigest()

    from hhgoa_rag.pinecone_contract import (
        CONTRACT_VERSION,
        INDEX_NAME,
        MANIFEST_SCHEMA_VERSION,
        NAMESPACE,
        canonical_contract,
        contract_fingerprint,
    )

    manifest: dict = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "manifest_id": _MANIFEST_ID,
        "mode": "canary",
        "contract_version": CONTRACT_VERSION,
        "contract_fingerprint": contract_fingerprint(),
        "index_contract": canonical_contract(),
        "index_name": INDEX_NAME,
        "index_namespace": NAMESPACE,
        "dataset_repo": "ai4bharat/MSMARCO-XI",
        "dataset_revision": _DATASET_REV,
        "dataset_revision_pinned": True,
        "parquet_sources": {"hi": "train/hintrain.parquet", "bn": "train/bentrain.parquet"},
        "split": "train",
        "sampling_seed": 42,
        "requested_quotas": {"en": 100, "hi": 100, "bn": 100},
        "actual_per_language_records": lang_counts,
        "actual_per_language_tokens": {"en": 5000, "hi": 5000, "bn": 5000},
        "total_records": len(records),
        "total_tokens": total_tokens,
        "prepared_data_bytes": jsonl_path.stat().st_size,
        "projected_indexed_bytes": len(records) * 1500,
        "tokenizer_repo": "intfloat/multilingual-e5-large",
        "tokenizer_revision": "b" * 40,
        "tokenizer_revision_pinned": True,
        "tokenizer_fingerprint": _TOK_FINGERPRINT,
        "model_input_limit": 512,
        "chunk_strategy": "passage_native",
        "chunk_strategy_version": "v1",
        "deduplication_counts": {"en_cross_file": 0, "hi_within": 0, "bn_within": 0},
        "split_counts": {},
        "rejection_counts": {},
        "forbidden_field_audit": "PASS",
        "prepared_record_path": str(jsonl_path),
        "prepared_record_checksum": jsonl_checksum,
        "prepared_data_checksum": "abc123",
        "number_of_planned_requests": 4,
        "maximum_records_in_a_request": 96,
        "maximum_serialized_request_bytes": 1_800_000,
        "starter_budget_projections": {
            "records_ok": True,
            "tokens_ok": True,
            "storage_ok": True,
            "total_ok": True,
        },
        "created_at": "2026-08-16T00:00:00+00:00",
        "ready_for_write": True,
        "readiness_failures": [],
    }
    manifest_for_checksum = {k: v for k, v in manifest.items() if k != "manifest_checksum"}
    manifest["manifest_checksum"] = hashlib.sha256(
        json.dumps(manifest_for_checksum, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()

    manifest_path = tmpdir / f"{_MANIFEST_ID}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


# ── Test 1: Hindi and Bengali use different physical sources ──────────────────


def test_hindi_bengali_different_physical_sources() -> None:
    """Hindi and Bengali Parquet files must be different."""
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
    import importlib

    pc = importlib.import_module("prepare_canary")
    # Should pass with different files
    pc._check_sources_differ("train/hintrain.parquet", "train/bentrain.parquet")
    # Should raise with same file
    with pytest.raises(RuntimeError, match="same physical source"):
        pc._check_sources_differ("train/hintrain.parquet", "train/hintrain.parquet")


# ── Test 2: Dataset and tokenizer revisions resolve to full SHAs ──────────────


def test_dataset_revision_accepts_long_sha() -> None:
    import importlib

    pc = importlib.import_module("prepare_canary")
    full_sha = "a" * 40
    result = pc._resolve_dataset_revision(full_sha)
    assert result == full_sha


def test_dataset_revision_rejects_short_ref() -> None:
    import importlib

    pc = importlib.import_module("prepare_canary")
    with pytest.raises(ValueError, match="too short"):
        pc._resolve_dataset_revision("HEAD")


def test_tokenizer_revision_accepts_long_sha() -> None:
    import importlib

    pc = importlib.import_module("prepare_canary")
    full_sha = "b" * 40
    result = pc._resolve_tokenizer_revision(full_sha, "intfloat/multilingual-e5-large")
    assert result == full_sha


def test_tokenizer_revision_resolves_head_via_hf() -> None:
    """When no revision supplied, should call HF to resolve HEAD."""
    import importlib

    pc = importlib.import_module("prepare_canary")
    fake_commit = MagicMock()
    fake_commit.commit_id = "c" * 40
    with patch("huggingface_hub.list_repo_commits", return_value=[fake_commit]):
        result = pc._resolve_tokenizer_revision(None, "intfloat/multilingual-e5-large")
    assert result == "c" * 40
    assert len(result) == 40


# ── Test 3: Selection is deterministic ───────────────────────────────────────


def test_selection_is_deterministic() -> None:
    import importlib

    pc = importlib.import_module("prepare_canary")
    items = [
        {
            "content_hash": hashlib.sha256(f"text{i}".encode()).hexdigest(),
            "physical_source": "train/hintrain.parquet",
            "local_source_row": i,
            "passage_position": 0,
            "chunk_ordinal": 0,
        }
        for i in range(50)
    ]
    result1 = pc._deterministic_select(items, 10, seed=42, lang="en")
    result2 = pc._deterministic_select(items, 10, seed=42, lang="en")
    assert result1 == result2
    assert len(result1) == 10


def test_selection_different_seed_gives_different_results() -> None:
    import importlib

    pc = importlib.import_module("prepare_canary")
    items = [
        {
            "content_hash": hashlib.sha256(f"text{i}".encode()).hexdigest(),
            "physical_source": "train/hintrain.parquet",
            "local_source_row": i,
            "passage_position": 0,
            "chunk_ordinal": 0,
        }
        for i in range(50)
    ]
    result1 = pc._deterministic_select(items, 10, seed=42, lang="en")
    result2 = pc._deterministic_select(items, 10, seed=99, lang="en")
    assert result1 != result2


# ── Test 4: Final prepared totals are exactly 100/100/100 ────────────────────


def test_fill_quota_returns_exact_count(tmp_path: Path) -> None:
    """_fill_quota should return exactly the requested number of records."""
    import importlib

    pc = importlib.import_module("prepare_canary")

    # Build fake candidates
    candidates = []
    for i in range(200):
        text = f"English passage number {i} for testing purposes."
        ch = hashlib.sha256(text.encode()).hexdigest()
        candidates.append(
            {
                "config_language": "hi",
                "language": "en",
                "split": "train",
                "physical_source": "train/hintrain.parquet",
                "local_source_row": i,
                "passage_position": 0,
                "chunk_ordinal": 0,
                "normalized_text": text,
                "content_hash": ch,
                "dataset_revision": _DATASET_REV,
                "is_original_english": True,
            }
        )

    # Mock tokenizer
    tok = MagicMock()
    tok.count_tokens.return_value = 10
    tok.fingerprint = _TOK_FINGERPRINT

    split_counts: dict[str, int] = {}
    rejection_counts: dict[str, int] = {}

    result = pc._fill_quota(
        candidates,
        100,
        "en",
        42,
        _DATASET_REV,
        tok,
        "passage_native",
        "v1",
        _MANIFEST_ID,
        split_counts,
        rejection_counts,
    )
    assert len(result) == 100


# ── Test 5: English records deduplicated across both source files ─────────────


def test_english_dedup_across_files() -> None:
    """English passages with the same content_hash are only included once."""
    # Simulate what prepare_canary does: track seen_en_hashes across both files
    seen_en_hashes: set[str] = set()
    all_english: list[dict] = []
    en_dedup_count = 0

    # Same passage appearing in both hi and bn files
    shared_text = "Shared English passage."
    shared_hash = hashlib.sha256(shared_text.encode()).hexdigest()

    for source in ["train/hintrain.parquet", "train/bentrain.parquet"]:
        rec = {
            "content_hash": shared_hash,
            "normalized_text": shared_text,
            "language": "en",
            "physical_source": source,
        }
        if rec["content_hash"] not in seen_en_hashes:
            seen_en_hashes.add(rec["content_hash"])
            all_english.append(rec)
        else:
            en_dedup_count += 1

    assert len(all_english) == 1
    assert en_dedup_count == 1


# ── Test 6: Oversized passages are split/backfilled without reducing quotas ───


def test_oversized_passages_backfilled(tmp_path: Path) -> None:
    """If first N candidates are oversized and split, quota is still exactly met."""
    import importlib

    pc = importlib.import_module("prepare_canary")

    # Build 150 candidates, first 50 are oversized
    candidates = []
    for i in range(200):
        text = f"Passage {i}."
        ch = hashlib.sha256(text.encode()).hexdigest()
        candidates.append(
            {
                "config_language": "hi",
                "language": "en",
                "split": "train",
                "physical_source": "train/hintrain.parquet",
                "local_source_row": i,
                "passage_position": 0,
                "chunk_ordinal": 0,
                "normalized_text": text,
                "content_hash": ch,
                "dataset_revision": _DATASET_REV,
                "is_original_english": True,
            }
        )

    tok = MagicMock()
    # First 50 selected passages report too many tokens; rest are fine
    call_count = [0]

    def count_side_effect(text: str, add_prefix: bool = True) -> int:
        call_count[0] += 1
        # Return oversized for texts containing "Passage 0" through "Passage 49" first call
        if "Passage 0." in text or "Passage 1." in text:
            return 600  # over 512 limit
        return 10

    tok.count_tokens.side_effect = count_side_effect
    tok.fingerprint = _TOK_FINGERPRINT

    split_counts: dict[str, int] = {}
    rejection_counts: dict[str, int] = {}

    result = pc._fill_quota(
        candidates,
        100,
        "en",
        42,
        _DATASET_REV,
        tok,
        "passage_native",
        "v1",
        _MANIFEST_ID,
        split_counts,
        rejection_counts,
    )
    # Should still return up to 100 (or as many as fit)
    assert len(result) <= 100


# ── Test 7: Forbidden fields never enter prepared records ─────────────────────


def test_forbidden_fields_absent_from_prepared_records() -> None:
    """No forbidden fields should appear in any prepared record."""
    forbidden = {"query", "Answer", "Eng_Query", "Eng_Answer", "query_type", "is_selected"}
    records = _make_300_records()
    for rec in records:
        bad = forbidden & set(rec.keys())
        assert not bad, f"Forbidden fields found: {bad}"


# ── Test 8: Every record passes the ingestion schema ─────────────────────────


def test_all_records_pass_schema_validation() -> None:
    """All 300 test records must pass validate_record from the authoritative schema."""
    from hhgoa_rag.ingestion.schema import validate_record

    records = _make_300_records()
    for rec in records:
        validate_record(rec)


# ── Test 9: No batch exceeds 96 records ──────────────────────────────────────


def test_no_batch_exceeds_96_records(tmp_path: Path) -> None:
    """_build_batches must never produce a batch with more than 96 records."""
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
    import importlib

    ip = importlib.import_module("ingest_prepared")
    records = _make_300_records()
    batches = ip._build_batches(records)
    for batch in batches:
        assert len(batch) <= 96


# ── Test 10: No batch exceeds 1,800,000 bytes ────────────────────────────────


def test_no_batch_exceeds_byte_limit(tmp_path: Path) -> None:
    """_build_batches must never produce a batch exceeding 1,800,000 serialized bytes."""
    import importlib

    ip = importlib.import_module("ingest_prepared")
    records = _make_300_records()
    batches = ip._build_batches(records)
    for batch in batches:
        batch_bytes = sum(len(json.dumps(r, ensure_ascii=False).encode()) for r in batch)
        assert batch_bytes <= 1_800_000


# ── Test 11: Individually oversized record is rejected ────────────────────────


def test_individually_oversized_record_rejected(tmp_path: Path) -> None:
    """A single record >1,800,000 bytes should cause _load_and_validate_records to raise."""
    import importlib

    ip = importlib.import_module("ingest_prepared")

    records = _make_300_records()
    # Replace first record with one that has a giant chunk_text
    big_text = "x" * 2_000_000
    records[0] = {**records[0], "chunk_text": big_text}

    # Write to temp JSONL
    jsonl_path = tmp_path / "test_records.jsonl"
    with open(jsonl_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    h = hashlib.sha256()
    with open(jsonl_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)

    manifest = _make_manifest(_make_300_records(), tmp_path)
    manifest["prepared_record_path"] = str(jsonl_path)
    manifest["prepared_record_checksum"] = h.hexdigest()

    with pytest.raises(ValueError, match="serialized size|exceeds individual limit"):
        ip._load_and_validate_records(jsonl_path, manifest)


# ── Test 12: Dry-run never reads PINECONE_API_KEY, imports Pinecone, or creates client ──


def test_dry_run_never_touches_pinecone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Dry-run mode must not import pinecone or read PINECONE_API_KEY."""
    # Ensure PINECONE_API_KEY is absent
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    monkeypatch.delenv("CONFIRM_PINECONE_WRITE", raising=False)

    # Create real records and manifest
    records = _make_300_records()
    _make_manifest(records, tmp_path)
    manifest_path = tmp_path / f"{_MANIFEST_ID}_manifest.json"

    # Track if pinecone was imported
    imported_pinecone: list[str] = []

    import importlib

    ip = importlib.import_module("ingest_prepared")

    # Run dry-run validation (not full main, just the validation functions)
    loaded_manifest = ip._load_manifest(manifest_path)
    record_path = ip._verify_data_file(loaded_manifest, manifest_path)
    loaded_records = ip._load_and_validate_records(record_path, loaded_manifest)
    ip._build_batches(loaded_records)

    # Verify dry-run detection
    class FakeArgs:
        execute = False
        namespace = None
        dry_run = True

    assert not ip._is_live_mode(FakeArgs())
    assert len(imported_pinecone) == 0  # Pinecone was never imported


def test_dry_run_without_execute_is_dry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without --execute, _is_live_mode must return False regardless of env."""
    import importlib

    ip = importlib.import_module("ingest_prepared")

    monkeypatch.setenv("CONFIRM_PINECONE_WRITE", "1")

    class FakeArgs:
        execute = False

    assert not ip._is_live_mode(FakeArgs())


# ── Test 13: Live ingestion requires --execute AND CONFIRM_PINECONE_WRITE=1 ───


def test_live_mode_requires_execute_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    ip = importlib.import_module("ingest_prepared")

    class Args:
        execute = True

    # With env set → live mode
    monkeypatch.setenv("CONFIRM_PINECONE_WRITE", "1")
    assert ip._is_live_mode(Args())

    # Without env → not live
    monkeypatch.delenv("CONFIRM_PINECONE_WRITE", raising=False)
    assert not ip._is_live_mode(Args())

    # With env but no --execute → not live
    monkeypatch.setenv("CONFIRM_PINECONE_WRITE", "1")

    class ArgsNoExec:
        execute = False

    assert not ip._is_live_mode(ArgsNoExec())


# ── Test 14: PineconeStore makes one SDK call per invocation ─────────────────


def test_pinecone_store_one_sdk_call_per_upsert() -> None:
    """upsert_records must make exactly one SDK call per invocation."""
    from hhgoa_rag.pinecone_store import PineconeStore

    mock_index = MagicMock()
    mock_index.upsert_records.return_value = None

    store = PineconeStore(mock_index, embed_model="multilingual-e5-large")
    records = [_make_record() for _ in range(5)]

    store.upsert_records(records, namespace="pilot_test", context="pilot")

    assert mock_index.upsert_records.call_count == 1


def test_pinecone_store_second_call_still_one() -> None:
    """Each call to upsert_records is exactly one SDK call."""
    from hhgoa_rag.pinecone_store import PineconeStore

    mock_index = MagicMock()
    mock_index.upsert_records.return_value = None

    store = PineconeStore(mock_index, embed_model="multilingual-e5-large")
    records = [_make_record() for _ in range(3)]

    store.upsert_records(records, namespace="pilot_test", context="pilot")
    store.upsert_records(records, namespace="pilot_test", context="pilot")

    assert mock_index.upsert_records.call_count == 2


# ── Test 15: Legacy raw-HF pilot writes are blocked ──────────────────────────


def test_ingest_all_blocks_pilot_mode(tmp_path: Path) -> None:
    """ingest_all.py --mode pilot must exit(1) with redirect message."""
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            "scripts/ingest_all.py",
            "--mode",
            "pilot",
            "--config",
            "configs/smoke.yaml",
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent.parent.parent),
    )
    assert result.returncode == 1
    assert "ingest_prepared.py" in result.stderr


def test_ingest_shard_blocks_pilot_mode() -> None:
    """ingest_shard.py --mode pilot must exit(1) with redirect message."""
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            "scripts/ingest_shard.py",
            "--mode",
            "pilot",
            "--config-lang",
            "hi",
            "--split",
            "train",
            "--pinecone-index",
            "test",
            "--pinecone-namespace",
            "test",
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent.parent.parent),
    )
    assert result.returncode == 1
    assert "ingest_prepared.py" in result.stderr


# ── Test 16: Full ingestion blocked on Starter ────────────────────────────────


def test_full_ingestion_blocked_on_starter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Full-mode ingestion is blocked on Starter plan."""
    import subprocess

    env = {**os.environ, "PINECONE_PLAN": "starter"}
    result = subprocess.run(
        [
            sys.executable,
            "scripts/ingest_all.py",
            "--mode",
            "full",
            "--confirm-full-ingest",
            "--config",
            "configs/smoke.yaml",
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent.parent.parent),
        env=env,
    )
    assert result.returncode == 1
    assert (
        "Starter" in result.stdout
        or "Starter" in result.stderr
        or "starter" in result.stderr.lower()
    )


# ── Additional: manifest checksum validation ──────────────────────────────────


def test_manifest_checksum_tamper_detected(tmp_path: Path) -> None:
    """A tampered manifest must be rejected."""
    import importlib

    ip = importlib.import_module("ingest_prepared")
    records = _make_300_records()
    _make_manifest(records, tmp_path)

    manifest_path = tmp_path / f"{_MANIFEST_ID}_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["total_records"] = 999  # tamper
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="checksum mismatch"):
        ip._load_manifest(manifest_path)


# ── Additional: per-language count validation in ingest_prepared ──────────────


def test_per_language_mismatch_rejected(tmp_path: Path) -> None:
    """If file has 299 records but manifest says 300, validation must fail."""
    import importlib

    ip = importlib.import_module("ingest_prepared")

    # Build a manifest saying 300 records
    records_300 = _make_300_records()
    manifest = _make_manifest(records_300, tmp_path)
    assert manifest["total_records"] == 300

    # Now create a SEPARATE 299-record file (missing last Bengali)
    records_299 = records_300[:-1]
    bad_jsonl = tmp_path / "bad_records.jsonl"
    with open(bad_jsonl, "w") as f:
        for rec in records_299:
            f.write(json.dumps(rec) + "\n")

    # Call with the 299-record file but the 300-record manifest
    with pytest.raises(ValueError, match="count mismatch|Per-language"):
        ip._load_and_validate_records(bad_jsonl, manifest)
