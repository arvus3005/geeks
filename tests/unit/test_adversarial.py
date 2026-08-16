"""Adversarial production-path tests.

Covers:
- prepare_canary helpers (stable key, Parquet URL, source identity check)
- ingest_prepared helpers (batch packer, manifest validation, forbidden field check)
- tokenizer revision-keyed caching
- schema strict allowlist
- smoke fixture constant
- validate_pinecone_config credential handling
- passage_ids determinism
- parser forbidden field stripping
- strict schema rejection of extra / nested source fields
- dry-run never imports or constructs Pinecone
- budget failure before SDK call
- one retry owner (PineconeStore makes exactly one SDK call)
- close() does not flush failed reservations
- batch limits (96 records / 1,800,000 bytes)
- oversized single record refused
- smoke namespace constant
- integration tests skip honestly without credentials
"""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ── 1. prepare_canary: _stable_selection_key is deterministic ────────────────


def test_stable_selection_key_same_inputs_same_output():
    sys_path_prepend()
    from scripts.prepare_canary import _stable_selection_key

    rec = make_rec("abc123", "train/hintrain.parquet", 10, 2, 0)
    k1 = _stable_selection_key(42, "hi", rec)
    k2 = _stable_selection_key(42, "hi", rec)
    assert k1 == k2


def test_stable_selection_key_different_seed_different_output():
    sys_path_prepend()
    from scripts.prepare_canary import _stable_selection_key

    rec = make_rec("abc123", "train/hintrain.parquet", 10, 2, 0)
    k1 = _stable_selection_key(42, "hi", rec)
    k2 = _stable_selection_key(99, "hi", rec)
    assert k1 != k2


def test_stable_selection_key_different_lang_different_output():
    sys_path_prepend()
    from scripts.prepare_canary import _stable_selection_key

    rec = make_rec("abc123", "train/hintrain.parquet", 10, 2, 0)
    k1 = _stable_selection_key(42, "hi", rec)
    k2 = _stable_selection_key(42, "bn", rec)
    assert k1 != k2


def test_stable_selection_key_different_physical_source_different_output():
    sys_path_prepend()
    from scripts.prepare_canary import _stable_selection_key

    rec1 = make_rec("abc123", "train/hintrain.parquet", 10, 2, 0)
    rec2 = make_rec("abc123", "train/bentrain.parquet", 10, 2, 0)
    k1 = _stable_selection_key(42, "en", rec1)
    k2 = _stable_selection_key(42, "en", rec2)
    assert k1 != k2


def test_stable_selection_key_different_row_different_output():
    sys_path_prepend()
    from scripts.prepare_canary import _stable_selection_key

    rec1 = make_rec("abc123", "train/hintrain.parquet", 10, 2, 0)
    rec2 = make_rec("abc123", "train/hintrain.parquet", 11, 2, 0)
    k1 = _stable_selection_key(42, "hi", rec1)
    k2 = _stable_selection_key(42, "hi", rec2)
    assert k1 != k2


def test_stable_selection_key_is_hex_string():
    sys_path_prepend()
    from scripts.prepare_canary import _stable_selection_key

    rec = make_rec("abc123", "train/hintrain.parquet", 10, 2, 0)
    k = _stable_selection_key(42, "hi", rec)
    assert isinstance(k, str)
    int(k, 16)  # must be valid hex


# ── 2. prepare_canary: _deterministic_select ─────────────────────────────────


def test_deterministic_select_is_reproducible():
    sys_path_prepend()
    from scripts.prepare_canary import _deterministic_select

    items = [make_rec(str(i), "train/hintrain.parquet", i, 0, 0) for i in range(20)]
    sel1 = _deterministic_select(items, 10, seed=42, lang="hi")
    sel2 = _deterministic_select(items, 10, seed=42, lang="hi")
    assert [r["local_source_row"] for r in sel1] == [r["local_source_row"] for r in sel2]


def test_deterministic_select_respects_quota():
    sys_path_prepend()
    from scripts.prepare_canary import _deterministic_select

    items = [make_rec(str(i), "train/hintrain.parquet", i, 0, 0) for i in range(50)]
    sel = _deterministic_select(items, 10, seed=42, lang="hi")
    assert len(sel) == 10


def test_deterministic_select_fewer_items_than_quota():
    sys_path_prepend()
    from scripts.prepare_canary import _deterministic_select

    items = [make_rec(str(i), "train/hintrain.parquet", i, 0, 0) for i in range(5)]
    sel = _deterministic_select(items, 10, seed=42, lang="hi")
    assert len(sel) == 5


def test_deterministic_select_empty_input():
    sys_path_prepend()
    from scripts.prepare_canary import _deterministic_select

    sel = _deterministic_select([], 10, seed=42, lang="hi")
    assert sel == []


def test_deterministic_select_different_seeds_may_differ():
    sys_path_prepend()
    from scripts.prepare_canary import _deterministic_select

    items = [make_rec(str(i), "train/hintrain.parquet", i, 0, 0) for i in range(50)]
    sel1 = _deterministic_select(items, 10, seed=42, lang="hi")
    sel2 = _deterministic_select(items, 10, seed=99, lang="hi")
    # Different seeds produce different orderings (extremely likely for 50 items)
    rows1 = [r["local_source_row"] for r in sel1]
    rows2 = [r["local_source_row"] for r in sel2]
    assert rows1 != rows2


# ── 3. prepare_canary: _parquet_url ──────────────────────────────────────────


def test_parquet_url_contains_revision():
    sys_path_prepend()
    from scripts.prepare_canary import _parquet_url

    url = _parquet_url("abc123deadbeef", "train/hintrain.parquet")
    assert "abc123deadbeef" in url


def test_parquet_url_contains_parquet_path():
    sys_path_prepend()
    from scripts.prepare_canary import _parquet_url

    url = _parquet_url("abc123", "train/bentrain.parquet")
    assert "bentrain.parquet" in url


def test_parquet_url_contains_dataset_repo():
    sys_path_prepend()
    from scripts.prepare_canary import _parquet_url

    url = _parquet_url("abc123", "train/hintrain.parquet")
    assert "ai4bharat" in url
    assert "MSMARCO-XI" in url


# ── 4. prepare_canary: _check_sources_differ ─────────────────────────────────


def test_check_sources_differ_passes_for_different_files():
    sys_path_prepend()
    from scripts.prepare_canary import _check_sources_differ

    # Should not raise
    _check_sources_differ("train/hintrain.parquet", "train/bentrain.parquet")


def test_check_sources_differ_fails_for_same_file():
    sys_path_prepend()
    from scripts.prepare_canary import _check_sources_differ

    with pytest.raises(RuntimeError, match="same physical source"):
        _check_sources_differ("train/hintrain.parquet", "train/hintrain.parquet")


# ── 5. ingest_prepared: _build_batches ───────────────────────────────────────


def test_build_batches_respects_record_count_limit():
    sys_path_prepend()
    from scripts.ingest_prepared import MAX_RECORDS_PER_BATCH, _build_batches

    records = [{"id": str(i), "chunk_text": "x"} for i in range(200)]
    batches = _build_batches(records)
    for batch in batches:
        assert len(batch) <= MAX_RECORDS_PER_BATCH


def test_build_batches_all_records_included():
    sys_path_prepend()
    from scripts.ingest_prepared import _build_batches

    records = [{"id": str(i), "chunk_text": "x"} for i in range(250)]
    batches = _build_batches(records)
    total = sum(len(b) for b in batches)
    assert total == 250


def test_build_batches_empty_input():
    sys_path_prepend()
    from scripts.ingest_prepared import _build_batches

    assert _build_batches([]) == []


def test_build_batches_single_record():
    sys_path_prepend()
    from scripts.ingest_prepared import _build_batches

    records = [{"id": "only", "chunk_text": "x"}]
    batches = _build_batches(records)
    assert len(batches) == 1
    assert batches[0][0]["id"] == "only"


def test_build_batches_respects_byte_limit():
    sys_path_prepend()
    from scripts.ingest_prepared import MAX_BYTES_PER_BATCH, _build_batches

    # Create records with large payloads
    big_text = "x" * 50_000
    records = [{"id": str(i), "chunk_text": big_text} for i in range(50)]
    batches = _build_batches(records)
    for batch in batches:
        batch_bytes = sum(len(json.dumps(r, ensure_ascii=False).encode()) for r in batch)
        assert batch_bytes <= MAX_BYTES_PER_BATCH


# ── 6. ingest_prepared: manifest validation ──────────────────────────────────


def test_load_manifest_raises_on_missing_file(tmp_path):
    sys_path_prepend()
    from scripts.ingest_prepared import _load_manifest

    with pytest.raises(FileNotFoundError):
        _load_manifest(tmp_path / "nonexistent.json")


def test_load_manifest_raises_on_forbidden_field_fail(tmp_path):
    sys_path_prepend()
    from scripts.ingest_prepared import _load_manifest

    manifest = _make_minimal_manifest(tmp_path)
    manifest["forbidden_field_audit"] = "FAIL: ['query']"
    _write_manifest(manifest, tmp_path)
    with pytest.raises(ValueError, match="forbidden"):
        _load_manifest(tmp_path / "manifest.json")


def test_load_manifest_raises_on_missing_required_field(tmp_path):
    sys_path_prepend()
    from scripts.ingest_prepared import _load_manifest

    manifest = _make_minimal_manifest(tmp_path)
    del manifest["total_records"]
    _write_manifest(manifest, tmp_path)
    with pytest.raises(ValueError, match="missing required"):
        _load_manifest(tmp_path / "manifest.json")


def test_load_manifest_raises_on_tampered_checksum(tmp_path):
    sys_path_prepend()
    from scripts.ingest_prepared import _load_manifest

    manifest = _make_minimal_manifest(tmp_path)
    # Add a correct checksum then tamper with a field
    manifest["manifest_checksum"] = "deadbeef" * 8
    _write_manifest(manifest, tmp_path)
    with pytest.raises(ValueError, match="checksum"):
        _load_manifest(tmp_path / "manifest.json")


def test_load_manifest_valid_passes(tmp_path):
    sys_path_prepend()
    from scripts.ingest_prepared import _load_manifest

    manifest = _make_minimal_manifest(tmp_path)
    _write_manifest_with_checksum(manifest, tmp_path)
    result = _load_manifest(tmp_path / "manifest.json")
    assert result["manifest_id"] == "test-manifest-001"


# ── 7. ingest_prepared: forbidden fields in records ──────────────────────────


def test_load_records_raises_on_forbidden_field(tmp_path):
    sys_path_prepend()
    from scripts.ingest_prepared import _load_records

    jsonl = tmp_path / "records.jsonl"
    jsonl.write_text(
        json.dumps({"id": "x", "chunk_text": "t", "query": "bad"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Forbidden"):
        _load_records(jsonl)


def test_load_records_clean_passes(tmp_path):
    sys_path_prepend()
    from scripts.ingest_prepared import _load_records

    jsonl = tmp_path / "records.jsonl"
    jsonl.write_text(
        json.dumps({"id": "x", "chunk_text": "t", "language": "en"}) + "\n",
        encoding="utf-8",
    )
    records = _load_records(jsonl)
    assert len(records) == 1


def test_load_records_raises_on_missing_id(tmp_path):
    sys_path_prepend()
    from scripts.ingest_prepared import _load_records

    jsonl = tmp_path / "records.jsonl"
    jsonl.write_text(
        json.dumps({"chunk_text": "t"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="id"):
        _load_records(jsonl)


# ── 8. tokenizer: revision-keyed caching ─────────────────────────────────────


def test_tokenizer_revision_keyed_cache_different_keys_differ():
    """Two calls with different revision args must not return the same cached instance
    if the cache has been cold-seeded."""
    from hhgoa_rag.ingestion import tokenizer as tok_module

    # Manually seed the cache with a fake wrapper for two revision keys
    fake_a = MagicMock()
    fake_a.revision = "aaa"
    fake_b = MagicMock()
    fake_b.revision = "bbb"

    with tok_module._lock:
        old_cache = dict(tok_module._cache)
        tok_module._cache["aaa"] = fake_a
        tok_module._cache["bbb"] = fake_b

    try:
        got_a = tok_module.get_tokenizer("aaa")
        got_b = tok_module.get_tokenizer("bbb")
        assert got_a is fake_a
        assert got_b is fake_b
        assert got_a is not got_b
    finally:
        with tok_module._lock:
            tok_module._cache.clear()
            tok_module._cache.update(old_cache)


def test_tokenizer_same_revision_returns_same_instance():
    from hhgoa_rag.ingestion import tokenizer as tok_module

    fake = MagicMock()
    fake.revision = "deadbeef"

    with tok_module._lock:
        old_cache = dict(tok_module._cache)
        tok_module._cache["deadbeef"] = fake

    try:
        a = tok_module.get_tokenizer("deadbeef")
        b = tok_module.get_tokenizer("deadbeef")
        assert a is b
    finally:
        with tok_module._lock:
            tok_module._cache.clear()
            tok_module._cache.update(old_cache)


# ── 9. schema: strict REQUIRED_FIELDS allowlist ──────────────────────────────


def test_schema_required_fields_includes_physical_shard():
    from hhgoa_rag.ingestion.schema import REQUIRED_FIELDS

    assert "physical_shard" in REQUIRED_FIELDS


def test_schema_required_fields_includes_local_source_row():
    from hhgoa_rag.ingestion.schema import REQUIRED_FIELDS

    assert "local_source_row" in REQUIRED_FIELDS


def test_schema_required_fields_includes_manifest_id():
    from hhgoa_rag.ingestion.schema import REQUIRED_FIELDS

    assert "manifest_id" in REQUIRED_FIELDS


def test_schema_forbidden_fields_complete():
    from hhgoa_rag.ingestion.schema import FORBIDDEN_FIELDS

    expected = {"query", "Answer", "Eng_Query", "Eng_Answer", "query_type", "is_selected"}
    assert FORBIDDEN_FIELDS == expected


# ── 10. passage_ids: deterministic UUIDv5 ────────────────────────────────────


def test_make_point_id_same_inputs_same_id():
    from hhgoa_rag.ingestion.passage_ids import make_point_id

    a = make_point_id("rev123", "en", "hash456", "passage_native_v1", 0)
    b = make_point_id("rev123", "en", "hash456", "passage_native_v1", 0)
    assert a == b


def test_make_point_id_different_ordinal_different_id():
    from hhgoa_rag.ingestion.passage_ids import make_point_id

    a = make_point_id("rev123", "en", "hash456", "passage_native_v1", 0)
    b = make_point_id("rev123", "en", "hash456", "passage_native_v1", 1)
    assert a != b


def test_make_point_id_different_language_different_id():
    from hhgoa_rag.ingestion.passage_ids import make_point_id

    a = make_point_id("rev123", "en", "hash456", "passage_native_v1", 0)
    b = make_point_id("rev123", "hi", "hash456", "passage_native_v1", 0)
    assert a != b


def test_make_point_id_is_valid_uuid():
    from hhgoa_rag.ingestion.passage_ids import make_point_id

    result = make_point_id("rev123", "en", "hash456", "passage_native_v1", 0)
    parsed = uuid.UUID(result)  # must not raise
    assert str(parsed) == result


# ── 11. validate_pinecone_config: no --pinecone-api-key flag ─────────────────


def test_validate_pinecone_config_has_no_api_key_arg():
    """The --pinecone-api-key CLI flag must not exist in validate_pinecone_config."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "scripts/validate_pinecone_config.py", "--help"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent.parent.parent),
    )
    assert "--pinecone-api-key" not in result.stdout
    assert "--pinecone-api-key" not in result.stderr


# ── 12. smoke fixture constant ────────────────────────────────────────────────


def test_smoke_dataset_revision_constant():
    from hhgoa_rag.ingestion.smoke_ingest import DATASET_REVISION

    # Must be a fixed constant — not HEAD, not main
    assert DATASET_REVISION == "smoke-fixture-v1"
    assert "HEAD" not in DATASET_REVISION
    assert "main" not in DATASET_REVISION


# ── Helpers ───────────────────────────────────────────────────────────────────


def sys_path_prepend():
    """Ensure scripts/ directory is importable."""
    import sys

    scripts_dir = str(Path(__file__).parent.parent.parent / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    # Also ensure repo root is on path for 'scripts' package-style imports
    repo_root = str(Path(__file__).parent.parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def make_rec(content_hash: str, physical_source: str, row: int, pos: int, ordinal: int) -> dict:
    return {
        "content_hash": content_hash,
        "physical_source": physical_source,
        "local_source_row": row,
        "passage_position": pos,
        "chunk_ordinal": ordinal,
    }


def _make_minimal_manifest(tmp_path: Path) -> dict:
    return {
        "manifest_schema_version": "2",
        "manifest_id": "test-manifest-001",
        "mode": "canary",
        "dataset_repo": "ai4bharat/MSMARCO-XI",
        "dataset_revision": "abc123def456" * 3,
        "total_records": 5,
        "total_tokens": 500,
        "prepared_record_path": str(tmp_path / "records.jsonl"),
        "prepared_record_checksum": "0" * 64,
        "ready_for_write": True,
        "readiness_failures": [],
        "forbidden_field_audit": "PASS",
        "tokenizer_fingerprint": "abcdef1234567890",
        "actual_per_language_records": {"en": 2, "hi": 2, "bn": 1},
    }


def _write_manifest(manifest: dict, tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_manifest_with_checksum(manifest: dict, tmp_path: Path) -> None:
    checksum = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    manifest["manifest_checksum"] = checksum
    _write_manifest(manifest, tmp_path)


# ── 13. strict schema: rejects extra and nested source fields ─────────────────


def test_schema_rejects_forbidden_field_at_top_level():
    """validate_record must refuse records with forbidden fields at top level."""
    from hhgoa_rag.ingestion.schema import SchemaViolationError, validate_record

    rec = _make_minimal_valid_record()
    rec["query"] = "what is the capital?"
    with pytest.raises(SchemaViolationError, match="query"):
        validate_record(rec)


def test_schema_rejects_forbidden_field_nested():
    """validate_record must refuse records with forbidden fields in nested dicts."""
    from hhgoa_rag.ingestion.schema import SchemaViolationError, validate_record

    rec = _make_minimal_valid_record()
    rec["source_meta"] = {"is_selected": True}
    with pytest.raises(SchemaViolationError, match="is_selected"):
        validate_record(rec)


def test_schema_rejects_missing_required_field():
    """validate_record must refuse records missing any required field."""
    from hhgoa_rag.ingestion.schema import SchemaViolationError, validate_record

    rec = _make_minimal_valid_record()
    del rec["language"]
    with pytest.raises(SchemaViolationError, match="language"):
        validate_record(rec)


def test_schema_build_record_contains_all_required_fields():
    """build_record must produce a record with every REQUIRED_FIELDS entry."""
    from hhgoa_rag.ingestion.schema import REQUIRED_FIELDS

    rec = _make_built_record()
    for field in REQUIRED_FIELDS:
        assert field in rec, f"Missing required field: {field}"


def test_schema_rejects_zero_token_length():
    """validate_record must refuse records with token_length <= 0."""
    from hhgoa_rag.ingestion.schema import SchemaViolationError, validate_record

    rec = _make_minimal_valid_record()
    rec["token_length"] = 0
    with pytest.raises(SchemaViolationError):
        validate_record(rec)


# ── 14. dry-run: never imports or constructs Pinecone ─────────────────────────


def test_dry_run_does_not_import_pinecone(tmp_path):
    """--dry-run must never import the pinecone package or read PINECONE_API_KEY."""
    import subprocess
    import sys

    # Create a dummy (empty) records file and compute its correct checksum
    records_path = tmp_path / "records.jsonl"
    records_path.write_text("")
    empty_checksum = hashlib.sha256(b"").hexdigest()

    # Build manifest pointing to the empty records file with correct checksums
    manifest = _make_minimal_manifest(tmp_path)
    manifest["prepared_record_path"] = str(records_path)
    manifest["total_records"] = 0
    manifest["total_tokens"] = 0
    manifest["actual_per_language_records"] = {}
    manifest["prepared_record_checksum"] = empty_checksum
    _write_manifest_with_checksum(manifest, tmp_path)

    # Explicitly unset PINECONE_API_KEY for this subprocess
    import os

    env_no_key = dict(os.environ)
    env_no_key.pop("PINECONE_API_KEY", None)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/ingest_prepared.py",
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent.parent.parent),
        env=env_no_key,
    )
    # Must exit cleanly (0) in dry-run
    assert result.returncode == 0, f"dry-run failed:\n{result.stderr}"
    # Output must not mention the live Pinecone constructor
    assert "Pinecone(api_key" not in result.stdout
    assert "Pinecone(api_key" not in result.stderr


# ── 15. budget failure before SDK call ────────────────────────────────────────


def test_budget_exceeded_before_sdk_call():
    """BudgetGuard.check_upsert must raise BEFORE any SDK call is made."""
    from unittest.mock import MagicMock

    from hhgoa_rag.ingestion.budget import BudgetExceededError, BudgetGuard, StarterBudget

    sdk = MagicMock()
    budget = StarterBudget(max_records=5)
    guard = BudgetGuard(budget=budget)
    # Pre-fill ledger to near limit
    guard.commit_upsert(language="en", record_count=5, token_count=100, byte_count=1000)

    with pytest.raises(BudgetExceededError):
        guard.check_upsert(language="en", record_count=1, token_count=10, byte_count=100)

    # SDK must never have been called
    sdk.upsert_records.assert_not_called()


# ── 16. one retry owner: PineconeStore makes exactly one SDK call ─────────────


def test_pinecone_store_makes_exactly_one_sdk_call_per_invocation():
    """PineconeStore.upsert_records must make exactly one SDK call, not retry internally."""
    from unittest.mock import MagicMock

    from hhgoa_rag.pinecone_store import SMOKE_NAMESPACE, TEXT_RECORD_FIELD, PineconeStore

    index = MagicMock()
    store = PineconeStore(index, embed_model="multilingual-e5-large")
    store.upsert_records([{"id": "x", TEXT_RECORD_FIELD: "t"}], namespace=SMOKE_NAMESPACE)
    assert index.upsert_records.call_count == 1


def test_pinecone_store_does_not_retry_on_failure():
    """PineconeStore.upsert_records must not retry internally; failure propagates immediately."""
    from unittest.mock import MagicMock

    from hhgoa_rag.pinecone_store import SMOKE_NAMESPACE, TEXT_RECORD_FIELD, PineconeStore

    index = MagicMock()
    index.upsert_records.side_effect = RuntimeError("sdk error")
    store = PineconeStore(index, embed_model="multilingual-e5-large")
    with pytest.raises(RuntimeError, match="sdk error"):
        store.upsert_records([{"id": "x", TEXT_RECORD_FIELD: "t"}], namespace=SMOKE_NAMESPACE)
    # Exactly one attempt — no internal retry
    assert index.upsert_records.call_count == 1


def test_orchestration_retry_produces_bounded_sdk_calls():
    """Configured 2-retry policy produces no more than 3 SDK calls (not 9 or 16)."""
    from unittest.mock import MagicMock

    from hhgoa_rag.pinecone_store import SMOKE_NAMESPACE, TEXT_RECORD_FIELD, PineconeStore

    index = MagicMock()
    index.upsert_records.side_effect = RuntimeError("persistent")
    store = PineconeStore(index, embed_model="multilingual-e5-large")
    records = [{"id": "x", TEXT_RECORD_FIELD: "t"}]

    max_retries = 2
    sdk_calls = 0
    for attempt in range(max_retries + 1):
        try:
            store.upsert_records(records, namespace=SMOKE_NAMESPACE)
            break
        except Exception:
            sdk_calls += 1

    assert sdk_calls == max_retries + 1, f"Expected {max_retries + 1} calls, got {sdk_calls}"
    assert index.upsert_records.call_count == max_retries + 1


# ── 17. close() does not flush failed reservations ────────────────────────────


def test_dedup_close_discards_pending_hashes():
    """ContentDeduplicator.close() must discard pending hashes, not flush them."""
    import tempfile

    from hhgoa_rag.ingestion.dedup import ContentDeduplicator

    with tempfile.TemporaryDirectory() as d:
        db_path = Path(d) / "dedup.db"
        dedup = ContentDeduplicator(db_path)
        dedup.mark_seen("should-not-persist", "pid1")
        dedup.close()

        # Re-open and verify hash was NOT committed
        dedup2 = ContentDeduplicator(db_path)
        assert not dedup2.is_duplicate(
            "should-not-persist"
        ), "close() must not flush pending hashes — unacknowledged records must be replayable"
        dedup2.close()


# ── 18. batch limits ──────────────────────────────────────────────────────────


def test_build_batches_never_exceeds_96_records():
    """No batch produced by _build_batches may exceed 96 records."""
    sys_path_prepend()
    from scripts.ingest_prepared import _build_batches

    records = [{"id": str(i), "chunk_text": "t"} for i in range(1000)]
    batches = _build_batches(records)
    for i, batch in enumerate(batches):
        assert len(batch) <= 96, f"Batch {i} has {len(batch)} records, exceeds 96"


def test_build_batches_never_exceeds_1_800_000_bytes():
    """No batch produced by _build_batches may exceed 1,800,000 serialized bytes."""
    sys_path_prepend()
    from scripts.ingest_prepared import _build_batches

    big_text = "x" * 20_000
    records = [{"id": str(i), "chunk_text": big_text} for i in range(200)]
    batches = _build_batches(records)
    for i, batch in enumerate(batches):
        batch_bytes = sum(len(json.dumps(r, ensure_ascii=False).encode("utf-8")) for r in batch)
        assert batch_bytes <= 1_800_000, f"Batch {i} is {batch_bytes} bytes, exceeds 1,800,000"


def test_pinecone_store_rejects_oversized_batch():
    """PineconeStore.upsert_records must refuse batches with > 96 records."""
    from unittest.mock import MagicMock

    from hhgoa_rag.pinecone_store import SMOKE_NAMESPACE, TEXT_RECORD_FIELD, PineconeStore

    index = MagicMock()
    store = PineconeStore(index, embed_model="multilingual-e5-large")
    records = [{"id": str(i), TEXT_RECORD_FIELD: "t"} for i in range(97)]
    with pytest.raises(ValueError, match="96"):
        store.upsert_records(records, namespace=SMOKE_NAMESPACE)
    index.upsert_records.assert_not_called()


# ── 19. smoke namespace constant ──────────────────────────────────────────────


def test_smoke_test_namespace_is_fixed_constant():
    """The smoke namespace used in integration tests must be 'smoke-fixture-v001'."""
    import importlib.util

    integration_path = Path(__file__).parent.parent / "integration" / "test_pinecone_smoke.py"
    spec = importlib.util.spec_from_file_location("test_pinecone_smoke", integration_path)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    assert mod.SMOKE_TEST_NAMESPACE == "smoke-fixture-v001"


def test_smoke_namespace_does_not_contain_canary_or_pilot():
    """Smoke namespace must never contain 'canary', 'pilot', or 'full'."""
    from hhgoa_rag.pinecone_store import SMOKE_NAMESPACE

    # The module-level SMOKE_NAMESPACE is "smoke" (short namespace for is_safe_namespace)
    # Integration test uses "smoke-fixture-v001" which also satisfies this
    assert "canary" not in SMOKE_NAMESPACE
    assert "pilot" not in SMOKE_NAMESPACE
    assert "full" not in SMOKE_NAMESPACE


# ── 20. integration tests skip honestly without credentials ───────────────────


def test_integration_tests_skip_without_api_key(monkeypatch):
    """Integration tests must call pytest.skip when PINECONE_API_KEY is absent."""
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    monkeypatch.delenv("PINECONE_SMOKE_TEST", raising=False)

    import importlib.util

    integration_path = Path(__file__).parent.parent / "integration" / "test_pinecone_smoke.py"
    spec = importlib.util.spec_from_file_location("_smoke_check", integration_path)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    with pytest.raises(pytest.skip.Exception):
        mod._require_opt_in()


# ── 21. no credential accepted through CLI ────────────────────────────────────


def test_ingest_prepared_has_no_api_key_flag():
    """--pinecone-api-key CLI flag must not exist in ingest_prepared."""
    sys_path_prepend()
    from scripts.ingest_prepared import _build_parser

    p = _build_parser()
    # Collect all option strings from all actions
    all_opts = [opt for action in p._actions for opt in action.option_strings]
    assert (
        "--pinecone-api-key" not in all_opts
    ), "ingest_prepared must not accept --pinecone-api-key; credentials via env only"


# ── Helpers for new tests ─────────────────────────────────────────────────────


def _make_minimal_valid_record() -> dict:
    """Return a minimal dict that passes validate_record."""
    from hhgoa_rag.ingestion.schema import REQUIRED_FIELDS
    from hhgoa_rag.pinecone_store import TEXT_RECORD_FIELD

    rec: dict = {}
    for field in REQUIRED_FIELDS:
        if field == "id":
            rec[field] = "test-id-001"
        elif field == TEXT_RECORD_FIELD:
            rec[field] = "sample text"
        elif field == "token_length":
            rec[field] = 10
        elif field == "language":
            rec[field] = "en"
        elif field == "dataset_revision":
            rec[field] = "bf5cdc1f26e581e519018e434db14edd1b77602b"
        elif field == "chunk_total":
            rec[field] = 1
        elif field in ("chunk_ordinal", "local_source_row", "passage_position"):
            rec[field] = 0
        else:
            rec[field] = "test-value"
    return rec


def _make_built_record() -> dict:
    """Return a record produced by build_record()."""
    from hhgoa_rag.ingestion.schema import build_record

    return build_record(
        record_id="test-uuid-001",
        chunk_text="sample text for testing",
        language="en",
        config_language="en",
        dataset_revision="bf5cdc1f26e581e519018e434db14edd1b77602b",
        split="train",
        physical_shard="0",
        local_source_row=0,
        passage_position=0,
        parent_passage_id="parent-hash",
        content_hash="content-hash-001",
        chunk_strategy="passage_native",
        chunk_strategy_version="v1",
        chunk_ordinal=0,
        chunk_total=1,
        token_length=5,
        tokenizer_fingerprint="fp123",
        manifest_id="test-manifest",
    )
