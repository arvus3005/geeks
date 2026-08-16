"""Unit tests for PineconeStore using a fake Pinecone index."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from hhgoa_rag.pinecone_store import (
    FULL_NAMESPACE,
    PILOT_NAMESPACE_PREFIX,
    SMOKE_NAMESPACE,
    TEXT_RECORD_FIELD,
    PineconeProviderError,
    PineconeStore,
    SearchHit,
    is_safe_namespace,
)


def _make_hit(id: str, score: float, text: str = "sample", lang: str = "en") -> object:
    h = MagicMock()
    h.id = id
    h.score = score
    h.fields = {TEXT_RECORD_FIELD: text, "language": lang}
    return h


def _make_search_response(hits: list) -> MagicMock:
    resp = MagicMock()
    resp.result.hits = hits
    return resp


def _make_store(index: MagicMock | None = None) -> PineconeStore:
    if index is None:
        index = MagicMock()
    return PineconeStore(index, embed_model="multilingual-e5-large")


# ── Namespace safety ──────────────────────────────────────────────────────────


def test_smoke_is_safe():
    assert is_safe_namespace(SMOKE_NAMESPACE)


def test_pilot_is_safe():
    assert is_safe_namespace(f"{PILOT_NAMESPACE_PREFIX}run1")


def test_full_is_not_safe():
    assert not is_safe_namespace(FULL_NAMESPACE)


def test_arbitrary_is_not_safe():
    assert not is_safe_namespace("production")


def test_namespace_guard_blocks_full_in_smoke_context():
    store = _make_store()
    with pytest.raises(ValueError, match="full"):
        store.upsert_records(
            [{"id": "x", TEXT_RECORD_FIELD: "t"}], namespace=FULL_NAMESPACE, context="smoke"
        )


def test_namespace_guard_allows_full_in_full_context():
    index = MagicMock()
    index.upsert_records.return_value = MagicMock()
    store = _make_store(index)
    # Should not raise
    count = store.upsert_records(
        [{"id": "x", TEXT_RECORD_FIELD: "t"}], namespace=FULL_NAMESPACE, context="full"
    )
    assert count == 1


# ── Upsert records ────────────────────────────────────────────────────────────


def test_upsert_returns_record_count():
    index = MagicMock()
    store = _make_store(index)
    records = [
        {"id": f"id-{i}", TEXT_RECORD_FIELD: f"text {i}", "language": "en"} for i in range(5)
    ]
    count = store.upsert_records(records, namespace=SMOKE_NAMESPACE)
    assert count == 5
    index.upsert_records.assert_called_once()


def test_upsert_empty_returns_zero():
    store = _make_store()
    count = store.upsert_records([], namespace=SMOKE_NAMESPACE)
    assert count == 0


def test_upsert_retries_on_failure():
    index = MagicMock()
    index.upsert_records.side_effect = [RuntimeError("timeout"), None]
    store = PineconeStore(index, embed_model="multilingual-e5-large", retry_backoff=0.0)
    count = store.upsert_records([{"id": "x", TEXT_RECORD_FIELD: "t"}], namespace=SMOKE_NAMESPACE)
    assert count == 1
    assert index.upsert_records.call_count == 2


def test_upsert_raises_provider_error_after_max_retries():
    index = MagicMock()
    index.upsert_records.side_effect = RuntimeError("persistent failure")
    store = PineconeStore(
        index, embed_model="multilingual-e5-large", max_retries=2, retry_backoff=0.0
    )
    with pytest.raises(PineconeProviderError):
        store.upsert_records([{"id": "x", TEXT_RECORD_FIELD: "t"}], namespace=SMOKE_NAMESPACE)
    assert index.upsert_records.call_count == 3  # initial + 2 retries


# ── Search ────────────────────────────────────────────────────────────────────


def test_search_returns_hits():
    index = MagicMock()
    raw_hits = [
        _make_hit("id1", 0.9, "New Delhi is the capital"),
        _make_hit("id2", 0.7, "India geography"),
    ]
    index.search_records.return_value = _make_search_response(raw_hits)
    store = _make_store(index)

    hits = store.search("capital of India", top_k=5, namespace=SMOKE_NAMESPACE)
    assert len(hits) == 2
    assert isinstance(hits[0], SearchHit)
    assert hits[0].id == "id1"
    assert hits[0].score == pytest.approx(0.9)
    assert "New Delhi" in hits[0].text


def test_search_with_language_filter():
    index = MagicMock()
    index.search_records.return_value = _make_search_response([])
    store = _make_store(index)
    store.search("query", top_k=5, namespace=SMOKE_NAMESPACE, filter={"language": {"$in": ["hi"]}})
    call_kwargs = index.search_records.call_args.kwargs
    assert call_kwargs["filter"] == {"language": {"$in": ["hi"]}}


def test_search_sends_text_input_field():
    index = MagicMock()
    index.search_records.return_value = _make_search_response([])
    store = _make_store(index)
    store.search("test query", top_k=3, namespace=SMOKE_NAMESPACE)
    call_kwargs = index.search_records.call_args.kwargs
    assert call_kwargs["inputs"] == {"text": "test query"}
    assert call_kwargs["top_k"] == 3
    assert call_kwargs["namespace"] == SMOKE_NAMESPACE


def test_search_raises_provider_error_on_failure():
    index = MagicMock()
    index.search_records.side_effect = RuntimeError("network error")
    store = _make_store(index)
    with pytest.raises(PineconeProviderError):
        store.search("query", top_k=5, namespace=SMOKE_NAMESPACE)


# ── SearchHit helpers ─────────────────────────────────────────────────────────


def test_search_hit_text_and_language():
    hit = SearchHit(id="x", score=0.8, fields={TEXT_RECORD_FIELD: "Hello world", "language": "en"})
    assert hit.text == "Hello world"
    assert hit.language == "en"


def test_search_hit_missing_fields():
    hit = SearchHit(id="x", score=0.5, fields={})
    assert hit.text == ""
    assert hit.language == ""


# ── Metadata correctness ──────────────────────────────────────────────────────


def test_record_has_no_forbidden_fields():
    """Ensure records built in tests never include eval-time fields."""
    forbidden = {"query", "Answer", "Eng_Query", "Eng_Answer", "query_type", "is_selected"}
    record = {
        "id": "test-id",
        TEXT_RECORD_FIELD: "some passage text",
        "language": "en",
        "content_hash": "abc123",
        "chunk_strategy": "passage_native",
        "chunk_strategy_version": "v1",
        "chunk_ordinal": 0,
        "source_split": "smoke",
        "index_manifest_id": "smoke-v001",
    }
    assert not (set(record.keys()) & forbidden)


# ── Stats ─────────────────────────────────────────────────────────────────────


def test_count_namespace_returns_zero_when_missing():
    index = MagicMock()
    stats = MagicMock()
    stats.namespaces = {}
    index.describe_index_stats.return_value = stats
    store = _make_store(index)
    assert store.count_namespace("nonexistent") == 0


def test_count_namespace_returns_count():
    index = MagicMock()
    stats = MagicMock()
    ns_info = MagicMock()
    ns_info.vector_count = 42
    stats.namespaces = {"smoke": ns_info}
    index.describe_index_stats.return_value = stats
    store = _make_store(index)
    assert store.count_namespace("smoke") == 42


def test_describe_index_stats_raises_on_error():
    index = MagicMock()
    index.describe_index_stats.side_effect = RuntimeError("api error")
    store = _make_store(index)
    with pytest.raises(PineconeProviderError):
        store.describe_index_stats()
