"""Behavioural retrieval tests — no Pinecone credentials required.

Tests provider-independent utilities: FakeEmbedder determinism,
PineconeStore search result mapping, and namespace safety.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from hhgoa_rag.pinecone_store import (
    SMOKE_NAMESPACE,
    TEXT_RECORD_FIELD,
    PineconeStore,
    SearchHit,
)
from hhgoa_rag.retrieval.embedder import FakeEmbedder


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


def test_fake_embedder_normalized():
    e = FakeEmbedder()
    v = e.embed_query("hello")
    assert abs(np.linalg.norm(v) - 1.0) < 1e-5


def test_fake_embedder_deterministic():
    e = FakeEmbedder()
    a = e.embed_query("test passage about India")
    b = e.embed_query("test passage about India")
    assert np.allclose(a, b)


def test_search_hit_text_extraction():
    hit = SearchHit(
        id="x", score=0.9, fields={TEXT_RECORD_FIELD: "New Delhi is the capital", "language": "en"}
    )
    assert "New Delhi" in hit.text
    assert hit.language == "en"


def test_search_returns_scored_results():
    index = MagicMock()
    raw = [_make_hit("id1", 0.95, "capital of India"), _make_hit("id2", 0.70, "geography")]
    index.search_records.return_value = _make_search_response(raw)
    store = PineconeStore(index, embed_model="multilingual-e5-large")

    hits = store.search("India capital", top_k=5, namespace=SMOKE_NAMESPACE)
    assert len(hits) == 2
    assert hits[0].score > hits[1].score  # sorted by score descending


def test_search_sends_correct_namespace():
    index = MagicMock()
    index.search_records.return_value = _make_search_response([])
    store = PineconeStore(index, embed_model="multilingual-e5-large")
    store.search("query", top_k=3, namespace="pilot_abc")
    assert index.search_records.call_args.kwargs["namespace"] == "pilot_abc"


def test_multilingual_text_round_trips():
    """Non-ASCII text stored and retrieved correctly."""
    hindi_text = "भारत की राजधानी नई दिल्ली है"
    hit = SearchHit(id="h1", score=0.8, fields={TEXT_RECORD_FIELD: hindi_text, "language": "hi"})
    assert hit.text == hindi_text
    assert hit.language == "hi"
