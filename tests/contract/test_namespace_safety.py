"""Contract: smoke/pilot ingestion cannot write to the full namespace."""

from unittest.mock import MagicMock

import pytest

from hhgoa_rag.pinecone_store import (
    FULL_NAMESPACE,
    PILOT_NAMESPACE_PREFIX,
    SMOKE_NAMESPACE,
    TEXT_RECORD_FIELD,
    PineconeStore,
    is_safe_namespace,
)


def _store() -> PineconeStore:
    return PineconeStore(MagicMock(), embed_model="multilingual-e5-large")


def test_smoke_context_refuses_full_namespace():
    store = _store()
    with pytest.raises(ValueError, match="full"):
        store.upsert_records(
            [{"id": "x", TEXT_RECORD_FIELD: "t"}], namespace=FULL_NAMESPACE, context="smoke"
        )


def test_pilot_context_refuses_full_namespace():
    store = _store()
    with pytest.raises(ValueError, match="full"):
        store.upsert_records(
            [{"id": "x", TEXT_RECORD_FIELD: "t"}], namespace=FULL_NAMESPACE, context="pilot"
        )


def test_smoke_context_allows_smoke_namespace():
    index = MagicMock()
    store = PineconeStore(index, embed_model="multilingual-e5-large")
    store.upsert_records(
        [{"id": "x", TEXT_RECORD_FIELD: "t"}], namespace=SMOKE_NAMESPACE, context="smoke"
    )
    index.upsert_records.assert_called_once()


def test_pilot_context_allows_pilot_namespace():
    index = MagicMock()
    store = PineconeStore(index, embed_model="multilingual-e5-large")
    store.upsert_records(
        [{"id": "x", TEXT_RECORD_FIELD: "t"}],
        namespace=f"{PILOT_NAMESPACE_PREFIX}run1",
        context="pilot",
    )
    index.upsert_records.assert_called_once()


def test_full_context_allows_full_namespace():
    index = MagicMock()
    store = PineconeStore(index, embed_model="multilingual-e5-large")
    store.upsert_records(
        [{"id": "x", TEXT_RECORD_FIELD: "t"}], namespace=FULL_NAMESPACE, context="full"
    )
    index.upsert_records.assert_called_once()


def test_is_safe_namespace_smoke():
    assert is_safe_namespace(SMOKE_NAMESPACE)


def test_is_safe_namespace_pilot():
    assert is_safe_namespace(f"{PILOT_NAMESPACE_PREFIX}abc123")


def test_is_safe_namespace_full_is_not_safe():
    assert not is_safe_namespace(FULL_NAMESPACE)
