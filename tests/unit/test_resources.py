"""Tests for lifespan resource container."""

from hhgoa_rag.api.resources import AppResources


def test_initial_state_not_ready():
    r = AppResources()
    assert r.ready is False
    assert r.embedder is None
    assert r.sparse_encoder is None
    assert r.qdrant_client is None


def test_mark_ready():
    r = AppResources()
    r.mark_ready()
    assert r.ready is True


def test_mark_not_ready():
    r = AppResources()
    r.mark_ready()
    r.mark_not_ready("test_reason")
    assert r.ready is False
    assert r.readiness_detail["reason"] == "test_reason"


def test_readiness_detail_accumulated():
    r = AppResources()
    r.readiness_detail["embedder"] = "FakeEmbedder"
    r.readiness_detail["sparse_encoder"] = "Qdrant/bm25"
    assert r.readiness_detail["embedder"] == "FakeEmbedder"
