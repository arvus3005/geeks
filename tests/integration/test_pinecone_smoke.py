"""Integration tests: smoke ingestion against real Pinecone index.

REAL CLOUD TEST — opt-in only.
Requires:
  - PINECONE_API_KEY environment variable
  - PINECONE_SMOKE_TEST=1 environment variable (explicit opt-in)
  - An existing Pinecone integrated-embedding index (PINECONE_INDEX env var,
    default: msmarco-xi) created with multilingual-e5-large.

Run command:
  PINECONE_API_KEY=... PINECONE_SMOKE_TEST=1 uv run pytest tests/integration/ -v

Safety:
  - Uses a FIXED deterministic namespace 'smoke-fixture-v001' for all runs.
  - Repeated runs are idempotent — same fixture IDs overwrite, count stays constant.
  - Never deletes the index.
  - Never targets canary, pilot or full namespaces.
  - Never invokes reranking.
  - Bounded eventual-consistency polling with timeout.
  - Skips honestly without credentials.
"""

from __future__ import annotations

import os
import time

import pytest

PINECONE_INDEX = os.environ.get("PINECONE_INDEX", "msmarco-xi")
EMBED_MODEL = "multilingual-e5-large"

# Fixed, deterministic smoke namespace — repeated runs are idempotent
SMOKE_TEST_NAMESPACE = "smoke-fixture-v001"


def _require_opt_in() -> str:
    opt_in = os.environ.get("PINECONE_SMOKE_TEST") == "1"
    if not opt_in:
        pytest.skip(
            "Real Pinecone integration test skipped. "
            "To run: set PINECONE_API_KEY and PINECONE_SMOKE_TEST=1. "
            "See tests/integration/test_pinecone_smoke.py for setup instructions."
        )
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        pytest.fail(
            "PINECONE_SMOKE_TEST=1 was set, but PINECONE_API_KEY is missing or empty. "
            "Integration tests require a valid PINECONE_API_KEY."
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


def _poll_count(store, namespace: str, min_count: int, timeout: float = 90.0) -> int:
    """Poll until namespace count reaches min_count or timeout. Returns actual count."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        count = store.count_namespace(namespace)
        if count >= min_count:
            return count
        time.sleep(2)
    return store.count_namespace(namespace)


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_smoke_ingest_inserts_records(pinecone_store):
    """Insert smoke fixtures into fixed namespace and verify count reaches expected value."""
    import json

    from hhgoa_rag.ingestion.smoke_ingest import SMOKE_FIXTURES_PATH, _build_record

    with open(SMOKE_FIXTURES_PATH) as f:
        passages = json.load(f)

    records = [_build_record(p) for p in passages]
    submitted = pinecone_store.upsert_records(
        records, namespace=SMOKE_TEST_NAMESPACE, context="smoke"
    )
    assert submitted == len(records)

    count = _poll_count(pinecone_store, SMOKE_TEST_NAMESPACE, min_count=len(records), timeout=90)
    assert count == len(records), f"Expected {len(records)} records, got {count}"


def test_smoke_ingest_is_idempotent(pinecone_store):
    """Re-inserting same records must not grow the count."""
    import json

    from hhgoa_rag.ingestion.smoke_ingest import SMOKE_FIXTURES_PATH, _build_record

    with open(SMOKE_FIXTURES_PATH) as f:
        passages = json.load(f)

    records = [_build_record(p) for p in passages]
    before = _poll_count(pinecone_store, SMOKE_TEST_NAMESPACE, min_count=len(records), timeout=30)
    pinecone_store.upsert_records(records, namespace=SMOKE_TEST_NAMESPACE, context="smoke")
    time.sleep(5)
    after = pinecone_store.count_namespace(SMOKE_TEST_NAMESPACE)
    assert after == before, f"Idempotent ingest changed count from {before} to {after}"


def test_query_returns_results(pinecone_store):
    """Text query must return at least one hit from the smoke namespace."""
    hits = pinecone_store.search(
        query_text="What is the capital of India?",
        top_k=5,
        namespace=SMOKE_TEST_NAMESPACE,
    )
    assert len(hits) > 0, "Expected at least one search result"
    assert all(h.score >= 0 for h in hits)


def test_multilingual_query_hindi(pinecone_store):
    """Hindi query must not raise an error."""
    hits = pinecone_store.search(
        query_text="भारत की राजधानी क्या है",
        top_k=3,
        namespace=SMOKE_TEST_NAMESPACE,
    )
    assert isinstance(hits, list)


def test_multilingual_query_bengali(pinecone_store):
    """Bengali query must not raise an error."""
    hits = pinecone_store.search(
        query_text="ভারতের রাজধানী কী",
        top_k=3,
        namespace=SMOKE_TEST_NAMESPACE,
    )
    assert isinstance(hits, list)


def test_metadata_correctness(pinecone_store):
    """Every returned record must have required metadata and no forbidden eval fields."""
    from hhgoa_rag.ingestion.schema import FORBIDDEN_FIELDS
    from hhgoa_rag.pinecone_store import TEXT_RECORD_FIELD

    hits = pinecone_store.search("capital city", top_k=10, namespace=SMOKE_TEST_NAMESPACE)

    for hit in hits:
        assert hit.id, "Hit must have an ID"
        assert TEXT_RECORD_FIELD in hit.fields, f"Hit must have {TEXT_RECORD_FIELD} field"
        assert "language" in hit.fields, "Hit must have language field"
        leaked = FORBIDDEN_FIELDS & set(hit.fields.keys())
        assert not leaked, f"Forbidden eval fields in hit: {leaked}"


def test_language_filter(pinecone_store):
    """Language filter must restrict results to requested language."""
    hits = pinecone_store.search(
        "India",
        top_k=10,
        namespace=SMOKE_TEST_NAMESPACE,
        filter={"language": {"$in": ["en"]}},
    )
    for hit in hits:
        assert hit.language == "en", f"Got language {hit.language!r}, expected 'en'"


def test_namespace_guard_blocks_full_writes(pinecone_store):
    """Smoke context cannot write to full namespace."""
    from hhgoa_rag.pinecone_store import FULL_NAMESPACE, TEXT_RECORD_FIELD

    with pytest.raises(Exception, match="full"):
        pinecone_store.upsert_records(
            [{"id": "x", TEXT_RECORD_FIELD: "t"}],
            namespace=FULL_NAMESPACE,
            context="smoke",
        )


def test_count_namespace_is_non_negative_integer(pinecone_store):
    """count_namespace must return a non-negative integer."""
    count = pinecone_store.count_namespace(SMOKE_TEST_NAMESPACE)
    assert isinstance(count, int)
    assert count >= 0


def test_smoke_namespace_is_not_canary_or_pilot(pinecone_store):
    """Confirm test does not write to canary/pilot/full namespaces."""
    assert SMOKE_TEST_NAMESPACE.startswith("smoke-")
    assert "canary" not in SMOKE_TEST_NAMESPACE
    assert "pilot" not in SMOKE_TEST_NAMESPACE
    assert "full" not in SMOKE_TEST_NAMESPACE
