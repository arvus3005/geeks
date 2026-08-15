"""Integration tests: smoke ingestion against real local Qdrant."""

import json

import pytest
from qdrant_client import QdrantClient

QDRANT_URL = "http://localhost:6333"
SMOKE_COLLECTION = "msmarco_xi_passages_smoke_test_v001"
ALL_LANG_CODES = [
    "as",
    "bn",
    "gu",
    "hi",
    "kn",
    "ml",
    "mr",
    "ne",
    "or",
    "pa",
    "sa",
    "ta",
    "te",
    "ur",
    "en",
]


@pytest.fixture(scope="module")
def qdrant_client():
    try:
        client = QdrantClient(url=QDRANT_URL, timeout=5)
        client.get_collections()
        return client
    except Exception:
        pytest.skip("Qdrant not available")


@pytest.fixture(scope="module")
def smoke_collection(qdrant_client):
    from hhgoa_rag.ingestion.smoke_ingest import run_smoke_ingest
    from hhgoa_rag.qdrant_lifecycle import create_collection

    # Force create for test collection
    existing = {c.name for c in qdrant_client.get_collections().collections}
    if SMOKE_COLLECTION in existing:
        qdrant_client.delete_collection(SMOKE_COLLECTION)
    create_collection(qdrant_client, SMOKE_COLLECTION, force=False)
    run_smoke_ingest(qdrant_url=QDRANT_URL, collection=SMOKE_COLLECTION)
    yield SMOKE_COLLECTION
    qdrant_client.delete_collection(SMOKE_COLLECTION)


def test_collection_exists(qdrant_client, smoke_collection):
    cols = {c.name for c in qdrant_client.get_collections().collections}
    assert smoke_collection in cols


def test_point_count_matches_fixtures(qdrant_client, smoke_collection):
    with open("tests/fixtures/smoke_passages.json") as f:
        fixtures = json.load(f)
    count = qdrant_client.count(smoke_collection).count
    assert count == len(fixtures)


def test_idempotent_ingest(qdrant_client, smoke_collection):
    from hhgoa_rag.ingestion.smoke_ingest import run_smoke_ingest

    run_smoke_ingest(qdrant_url=QDRANT_URL, collection=smoke_collection)
    with open("tests/fixtures/smoke_passages.json") as f:
        fixtures = json.load(f)
    count = qdrant_client.count(smoke_collection).count
    assert count == len(fixtures), "Idempotent ingest changed point count"


def test_no_forbidden_fields_in_payload(qdrant_client, smoke_collection):
    forbidden = {"query", "Answer", "Eng_Query", "Eng_Answer", "query_type", "is_selected"}
    scroll, _ = qdrant_client.scroll(smoke_collection, limit=100, with_payload=True)
    for point in scroll:
        if point.payload:
            leaked = forbidden & set(point.payload.keys())
            assert not leaked, f"Forbidden fields found: {leaked}"


def test_dense_query(qdrant_client, smoke_collection):
    from hhgoa_rag.retrieval.embedder import FakeEmbedder

    embedder = FakeEmbedder()
    qvec = embedder.embed_query("query: capital of India")
    results = qdrant_client.search(
        collection_name=smoke_collection,
        query_vector=("dense", qvec.tolist()),
        limit=5,
    )
    assert len(results) > 0


def test_hybrid_rrf_query(qdrant_client, smoke_collection):
    from hhgoa_rag.retrieval.embedder import FakeEmbedder
    from hhgoa_rag.retrieval.hybrid import HybridRetriever

    embedder = FakeEmbedder()
    retriever = HybridRetriever(qdrant_client, smoke_collection, dense_k=5, sparse_k=5, fused_k=5)
    qvec = embedder.embed_query("query: capital of India")
    results = retriever.retrieve(qvec, "capital of India")
    assert isinstance(results, list)


def test_language_filter(qdrant_client, smoke_collection):
    from hhgoa_rag.retrieval.embedder import FakeEmbedder
    from hhgoa_rag.retrieval.hybrid import HybridRetriever

    embedder = FakeEmbedder()
    retriever = HybridRetriever(qdrant_client, smoke_collection)
    qvec = embedder.embed_query("query: India")
    results = retriever.retrieve(qvec, "India", language_filter=["bn"])
    for r in results:
        assert r["payload"].get("language") == "bn"


def test_alias_refused_for_smoke(qdrant_client, smoke_collection):
    from hhgoa_rag.qdrant_lifecycle import switch_alias

    with pytest.raises(ValueError):
        switch_alias(qdrant_client, smoke_collection, smoke_ok=False)


@pytest.mark.parametrize("lang", ALL_LANG_CODES)
def test_language_has_points(qdrant_client, smoke_collection, lang):
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    result = qdrant_client.count(
        collection_name=smoke_collection,
        count_filter=Filter(must=[FieldCondition(key="language", match=MatchValue(value=lang))]),
    )
    assert result.count >= 1, f"No points found for language '{lang}'"
