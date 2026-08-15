"""Tests for lifespan resource container."""

from hhgoa_rag.api.resources import AppResources


def test_initial_state_not_ready():
    r = AppResources()
    assert r.ready is False
    assert r.pinecone_store is None


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
    r.readiness_detail["pinecone_index"] = "msmarco-xi"
    r.readiness_detail["pinecone_embed_model"] = "multilingual-e5-large"
    assert r.readiness_detail["pinecone_index"] == "msmarco-xi"
