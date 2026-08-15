from hhgoa_rag.schemas.query import QueryRequest, QueryResponse, TimingsMs


def test_query_request_defaults():
    r = QueryRequest(question="What is India?")
    assert r.language_hint is None
    assert r.debug is False
    assert r.request_id  # auto-generated


def test_query_request_custom_id():
    r = QueryRequest(question="test", request_id="abc123")
    assert r.request_id == "abc123"


def test_timings_ms_all_zero():
    t = TimingsMs()
    assert t.total_backend == 0.0


def test_query_response_valid():
    r = QueryResponse(
        request_id="r1",
        question="q",
        detected_language="en",
        answer="Test answer.",
        decision="allow",
        reason_code="answered",
        grounded=True,
        confidence=0.8,
        citations=[],
        sources=[],
        retrieval_mode="dense_sparse_rrf",
        index_manifest_id="smoke-v001",
        model_manifest_id="e5-small-v001",
        timings_ms=TimingsMs(),
        total_backend_ms=45.0,
    )
    assert r.decision == "allow"
