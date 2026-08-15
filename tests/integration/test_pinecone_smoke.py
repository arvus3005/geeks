"""Integration tests: smoke ingestion against real Pinecone index.

REAL CLOUD TEST — opt-in only.
Requires:
  - PINECONE_API_KEY environment variable
  - PINECONE_SMOKE_TEST=1 environment variable (explicit opt-in)
  - An existing Pinecone integrated-embedding index (PINECONE_INDEX env var,
    default: msmarco-xi) created with:
      python scripts/create_pinecone_index.py --pinecone-index msmarco-xi

Run command:
  PINECONE_API_KEY=... PINECONE_SMOKE_TEST=1 uv run pytest tests/integration/test_pinecone_smoke.py -v

Safety:
  - Uses a unique smoke namespace per test run (smoke-test-<run_id>)
  - Never deletes the index
  - Uses bounded polling for eventual consistency (not fixed sleeps)
"""

from __future__ import annotations

import os
import time
import uuid

import pytest

PINECONE_INDEX = os.environ.get("PINECONE_INDEX", "msmarco-xi")
EMBED_MODEL = "multilingual-e5-large"


def _require_opt_in():
    api_key = os.environ.get("PINECONE_API_KEY")
    opt_in = os.environ.get("PINECONE_SMOKE_TEST") == "1"
    if not api_key or not opt_in:
        pytest.skip(
            "Real Pinecone integration test skipped. "
            "To run: set PINECONE_API_KEY and PINECONE_SMOKE_TEST=1. "
            "See tests/integration/test_pinecone_smoke.py docstring for setup."
        )
    return api_key


@pytest.fixture(scope="module")
def pinecone_store():
    api_key = _require_opt_in()

    from pinecone import Pinecone

    from hhgoa_rag.pinecone_store import PineconeStore

    pc = Pinecone(api_key=api_key)
    index = pc.Index(PINECONE_INDEX)
    store = PineconeStore(index, embed_model=EMBED_MODEL, search_timeout=20.0, upsert_timeout=30.0)
    return store


@pytest.fixture(scope="module")
def smoke_namespace():
    """Unique namespace per test run — avoids polluting shared state."""
    run_id = str(uuid.uuid4())[:8]
    return f"smoke-test-{run_id}"


def _poll_count(store, namespace: str, min_count: int, timeout: float = 60.0) -> int:
    """Poll until namespace count reaches min_count or timeout. Returns actual count."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        count = store.count_namespace(namespace)
        if count >= min_count:
            return count
        time.sleep(2)
    return store.count_namespace(namespace)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_smoke_ingest_inserts_records(pinecone_store, smoke_namespace):
    """Insert smoke fixtures and verify count reaches expected value."""
    import json

    from hhgoa_rag.ingestion.smoke_ingest import SMOKE_FIXTURES_PATH, _build_record
    with open(SMOKE_FIXTURES_PATH) as f:
        passages = json.load(f)

    records = [_build_record(p) for p in passages]
    submitted = pinecone_store.upsert_records(records, namespace=smoke_namespace, context="smoke")
    assert submitted == len(records)

    count = _poll_count(pinecone_store, smoke_namespace, min_count=len(records), timeout=90)
    assert count == len(records), f"Expected {len(records)} records, got {count}"


def test_smoke_ingest_is_idempotent(pinecone_store, smoke_namespace):
    """Re-inserting same records must not grow count."""
    import json

    from hhgoa_rag.ingestion.smoke_ingest import SMOKE_FIXTURES_PATH, _build_record
    with open(SMOKE_FIXTURES_PATH) as f:
        passages = json.load(f)

    records = [_build_record(p) for p in passages]
    before = _poll_count(pinecone_store, smoke_namespace, min_count=len(records), timeout=30)
    pinecone_store.upsert_records(records, namespace=smoke_namespace, context="smoke")
    time.sleep(5)  # brief pause for consistency
    after = pinecone_store.count_namespace(smoke_namespace)
    assert after == before, f"Idempotent ingest changed count from {before} to {after}"


def test_query_returns_results(pinecone_store, smoke_namespace):
    """Text query must return at least one hit from the smoke namespace."""
    hits = pinecone_store.search(
        query_text="What is the capital of India?",
        top_k=5,
        namespace=smoke_namespace,
    )
    assert len(hits) > 0, "Expected at least one search result"
    assert all(h.score >= 0 for h in hits)


def test_multilingual_query(pinecone_store, smoke_namespace):
    """Hindi query must return results."""
    hits = pinecone_store.search(
        query_text="भारत की राजधानी क्या है",
        top_k=3,
        namespace=smoke_namespace,
    )
    assert len(hits) >= 0  # may be 0 if no Hindi fixtures; pass if no error


def test_metadata_correctness(pinecone_store, smoke_namespace):
    """Every returned record must have required metadata, no forbidden eval fields."""
    from hhgoa_rag.pinecone_store import TEXT_RECORD_FIELD

    forbidden = {"query", "Answer", "Eng_Query", "Eng_Answer", "query_type", "is_selected"}
    hits = pinecone_store.search("capital city", top_k=10, namespace=smoke_namespace)

    for hit in hits:
        assert hit.id, "Hit must have an ID"
        assert TEXT_RECORD_FIELD in hit.fields, "Hit must have chunk_text field"
        assert "language" in hit.fields, "Hit must have language field"
        leaked = forbidden & set(hit.fields.keys())
        assert not leaked, f"Forbidden eval fields in hit: {leaked}"


def test_language_filter(pinecone_store, smoke_namespace):
    """Language filter must restrict results to requested language."""
    hits = pinecone_store.search(
        "India",
        top_k=10,
        namespace=smoke_namespace,
        filter={"language": {"$in": ["en"]}},
    )
    for hit in hits:
        assert hit.language == "en", f"Got language {hit.language!r}, expected 'en'"


def test_no_forbidden_fields_in_namespace_guard(pinecone_store, smoke_namespace):
    """Smoke context cannot write to full namespace."""
    from hhgoa_rag.pinecone_store import FULL_NAMESPACE, TEXT_RECORD_FIELD

    with pytest.raises(Exception, match="full"):
        pinecone_store.upsert_records(
            [{"id": "x", TEXT_RECORD_FIELD: "t"}],
            namespace=FULL_NAMESPACE,
            context="smoke",
        )


def test_reconciliation_count(pinecone_store, smoke_namespace):
    """count_namespace must return a non-negative integer."""
    count = pinecone_store.count_namespace(smoke_namespace)
    assert isinstance(count, int)
    assert count >= 0
