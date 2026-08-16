"""Tests for the authoritative Pinecone record schema module."""

from __future__ import annotations

import pytest

from hhgoa_rag.ingestion.schema import (
    FORBIDDEN_FIELDS,
    REQUIRED_FIELDS,
    SchemaViolationError,
    build_record,
    check_forbidden_fields,
    validate_record,
)
from hhgoa_rag.pinecone_store import TEXT_RECORD_FIELD


def _valid_kwargs() -> dict:
    return dict(
        record_id="test-uuid-0001",
        chunk_text="The capital of India is New Delhi.",
        language="en",
        config_language="hi",
        dataset_revision="abc123def456",
        split="train",
        physical_shard="0",
        local_source_row=42,
        passage_position=0,
        parent_passage_id="deadbeef1234",
        content_hash="deadbeef1234",
        chunk_strategy="passage_native",
        chunk_strategy_version="v1",
        chunk_ordinal=0,
        chunk_total=1,
        token_length=15,
        tokenizer_fingerprint="fp_abc123",
        manifest_id="canary-abc12345",
    )


# ── build_record ──────────────────────────────────────────────────────────────


def test_build_record_returns_all_required_fields():
    rec = build_record(**_valid_kwargs())
    assert REQUIRED_FIELDS.issubset(set(rec.keys()))


def test_build_record_id_is_correct():
    rec = build_record(**_valid_kwargs())
    assert rec["id"] == "test-uuid-0001"


def test_build_record_chunk_text_field():
    rec = build_record(**_valid_kwargs())
    assert rec[TEXT_RECORD_FIELD] == "The capital of India is New Delhi."


def test_build_record_no_forbidden_fields():
    rec = build_record(**_valid_kwargs())
    assert not (FORBIDDEN_FIELDS & set(rec.keys()))


# ── validate_record ───────────────────────────────────────────────────────────


def test_validate_record_passes_valid():
    rec = build_record(**_valid_kwargs())
    validate_record(rec)  # should not raise


def test_validate_record_raises_on_missing_required():
    rec = build_record(**_valid_kwargs())
    del rec["language"]
    with pytest.raises(SchemaViolationError, match="missing required"):
        validate_record(rec)


def test_validate_record_raises_on_empty_id():
    rec = build_record(**_valid_kwargs())
    rec["id"] = ""
    with pytest.raises(SchemaViolationError, match="id"):
        validate_record(rec)


def test_validate_record_raises_on_zero_token_length():
    kw = _valid_kwargs()
    kw["token_length"] = 0
    with pytest.raises(SchemaViolationError):
        build_record(**kw)


def test_validate_record_raises_on_forbidden_query():
    rec = build_record(**_valid_kwargs())
    rec["query"] = "What is the capital?"
    with pytest.raises(SchemaViolationError, match="query"):
        validate_record(rec)


def test_validate_record_raises_on_forbidden_answer():
    rec = build_record(**_valid_kwargs())
    rec["Answer"] = "New Delhi"
    with pytest.raises(SchemaViolationError):
        validate_record(rec)


def test_validate_record_detects_nested_forbidden():
    rec = build_record(**_valid_kwargs())
    rec["metadata"] = {"query": "sneaky", "other": "fine"}
    with pytest.raises(SchemaViolationError, match="query"):
        validate_record(rec)


# ── check_forbidden_fields (non-raising scan) ─────────────────────────────────


def test_check_forbidden_clean_record():
    rec = build_record(**_valid_kwargs())
    assert check_forbidden_fields(rec) == []


def test_check_forbidden_finds_query():
    rec = build_record(**_valid_kwargs())
    rec["query"] = "test"
    assert "query" in check_forbidden_fields(rec)


def test_check_forbidden_finds_nested():
    rec = build_record(**_valid_kwargs())
    rec["nested"] = {"Eng_Query": "test"}
    assert "Eng_Query" in check_forbidden_fields(rec)


def test_check_forbidden_all_fields():
    rec = build_record(**_valid_kwargs())
    for field in FORBIDDEN_FIELDS:
        rec[field] = "test_value"
    found = check_forbidden_fields(rec)
    assert FORBIDDEN_FIELDS.issubset(set(found))


# ── Contract: upsert dictionaries match schema ────────────────────────────────


def test_record_matches_schema_for_hindi():
    kw = _valid_kwargs()
    kw.update(language="hi", config_language="hi", chunk_text="भारत की राजधानी नई दिल्ली है।")
    rec = build_record(**kw)
    assert rec["language"] == "hi"
    validate_record(rec)


def test_record_matches_schema_for_bengali():
    kw = _valid_kwargs()
    kw.update(language="bn", config_language="bn", chunk_text="ভারতের রাজধানী নতুন দিল্লি।")
    rec = build_record(**kw)
    assert rec["language"] == "bn"
    validate_record(rec)
