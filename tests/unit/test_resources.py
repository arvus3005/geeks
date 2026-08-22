"""Tests for lifespan resource container."""

from hhgoa_rag.api.resources import AppResources


def test_initial_state_not_ready():
    r = AppResources()
    assert r.ready is False


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
    r.readiness_detail["retrieval_backend"] = "local_hybrid_sharded"
    r.readiness_detail["languages"] = ["hi", "bn"]
    assert r.readiness_detail["retrieval_backend"] == "local_hybrid_sharded"
