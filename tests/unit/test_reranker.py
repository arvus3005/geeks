"""Tests for Pinecone reranking boundary."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from hhgoa_rag.reranking.reranker import (
    DEFAULT_RERANK_MODEL,
    PineconeReranker,
    PineconeRerankError,
    RankedResult,
    RerankUsage,
    RetrievalOnlyPassthrough,
)


def _make_hit(i: int, score: float = 0.9) -> dict:
    return {
        "id": f"rec-{i}",
        "chunk_text": f"passage text number {i}",
        "language": "en",
        "content_hash": f"hash{i}",
        "_retrieval_score": score,
    }


def _mock_rerank_response(hits: list[dict], scores: list[float]) -> MagicMock:
    resp = MagicMock()
    items = []
    for hit, score in zip(hits, scores):
        item = MagicMock()
        item.score = score
        item.document = hit
        items.append(item)
    resp.results = items
    resp.usage = MagicMock()
    resp.usage.rerank_units = len(hits)
    resp.usage.search_units = None
    return resp


def _make_reranker(
    top_n: int = 3, candidate_k: int = 8, model: str = DEFAULT_RERANK_MODEL
) -> tuple[PineconeReranker, MagicMock]:
    pc = MagicMock()
    reranker = PineconeReranker(pc, model=model, candidate_k=candidate_k, top_n=top_n)
    return reranker, pc


# ── Configuration validation ──────────────────────────────────────────────────


def test_top_n_cannot_exceed_candidate_k():
    pc = MagicMock()
    with pytest.raises(ValueError, match="top_n"):
        PineconeReranker(pc, candidate_k=5, top_n=10)


def test_candidate_k_must_be_positive():
    pc = MagicMock()
    with pytest.raises(ValueError, match="candidate_k"):
        PineconeReranker(pc, candidate_k=0, top_n=3)


def test_top_n_must_be_positive():
    pc = MagicMock()
    with pytest.raises(ValueError, match="top_n"):
        PineconeReranker(pc, candidate_k=8, top_n=0)


def test_rank_fields_must_include_chunk_text():
    pc = MagicMock()
    with pytest.raises(ValueError, match="chunk_text"):
        PineconeReranker(pc, rank_fields=["some_other_field"])


def test_default_model():
    reranker, _ = _make_reranker()
    assert reranker.model == DEFAULT_RERANK_MODEL


def test_default_rank_fields_include_chunk_text():
    reranker, _ = _make_reranker()
    assert "chunk_text" in reranker.rank_fields


# ── Rerank call correctness ───────────────────────────────────────────────────


def test_rerank_uses_correct_model():
    reranker, pc = _make_reranker()
    hits = [_make_hit(i) for i in range(5)]
    pc.inference.rerank.return_value = _mock_rerank_response(hits[:3], [0.9, 0.7, 0.5])
    reranker.rerank("query", hits)
    call_kwargs = pc.inference.rerank.call_args.kwargs
    assert call_kwargs["model"] == DEFAULT_RERANK_MODEL


def test_rerank_uses_chunk_text_rank_field():
    reranker, pc = _make_reranker()
    hits = [_make_hit(i) for i in range(5)]
    pc.inference.rerank.return_value = _mock_rerank_response(hits[:3], [0.9, 0.7, 0.5])
    reranker.rerank("query", hits)
    call_kwargs = pc.inference.rerank.call_args.kwargs
    assert "chunk_text" in call_kwargs["rank_fields"]


def test_rerank_sends_correct_top_n():
    reranker, pc = _make_reranker(top_n=2, candidate_k=8)
    hits = [_make_hit(i) for i in range(8)]
    pc.inference.rerank.return_value = _mock_rerank_response(hits[:2], [0.9, 0.7])
    reranker.rerank("query", hits)
    call_kwargs = pc.inference.rerank.call_args.kwargs
    assert call_kwargs["top_n"] == 2


def test_rerank_returns_ranked_results():
    reranker, pc = _make_reranker(top_n=3)
    hits = [_make_hit(i) for i in range(8)]
    pc.inference.rerank.return_value = _mock_rerank_response(hits[:3], [0.95, 0.8, 0.6])
    results, usage = reranker.rerank("query", hits)
    assert len(results) == 3
    assert all(isinstance(r, RankedResult) for r in results)
    assert all(r.score_type == "rerank" for r in results)


def test_rerank_final_rank_sequential():
    reranker, pc = _make_reranker(top_n=3)
    hits = [_make_hit(i) for i in range(8)]
    pc.inference.rerank.return_value = _mock_rerank_response(hits[:3], [0.9, 0.7, 0.5])
    results, _ = reranker.rerank("query", hits)
    assert [r.final_rank for r in results] == [0, 1, 2]


def test_rerank_preserves_chunk_text():
    reranker, pc = _make_reranker(top_n=2)
    hits = [_make_hit(i) for i in range(5)]
    pc.inference.rerank.return_value = _mock_rerank_response(hits[:2], [0.9, 0.7])
    results, _ = reranker.rerank("query", hits)
    assert all(r.chunk_text for r in results)


def test_rerank_maps_usage():
    reranker, pc = _make_reranker()
    hits = [_make_hit(i) for i in range(4)]
    pc.inference.rerank.return_value = _mock_rerank_response(hits[:3], [0.9, 0.7, 0.5])
    _, usage = reranker.rerank("query", hits)
    assert isinstance(usage, RerankUsage)
    assert usage.elapsed_ms is not None and usage.elapsed_ms >= 0


def test_rerank_empty_results():
    reranker, pc = _make_reranker()
    pc.inference.rerank.return_value = _mock_rerank_response([], [])
    results, _ = reranker.rerank("query", [_make_hit(0)])
    assert results == []


# ── Error handling (fail-closed) ──────────────────────────────────────────────


def test_rerank_raises_on_api_failure():
    reranker, pc = _make_reranker()
    pc.inference.rerank.side_effect = RuntimeError("timeout")
    with pytest.raises(PineconeRerankError):
        reranker.rerank("query", [_make_hit(0)])


def test_rerank_missing_chunk_text_raises():
    reranker, pc = _make_reranker()
    bad_hit = {"id": "x", "language": "en"}  # missing chunk_text
    with pytest.raises(ValueError, match="chunk_text"):
        reranker.rerank("query", [bad_hit])


# ── RetrievalOnlyPassthrough ──────────────────────────────────────────────────


def test_passthrough_returns_top_n():
    pt = RetrievalOnlyPassthrough(top_n=3)
    hits = [_make_hit(i, score=1.0 - i * 0.1) for i in range(8)]
    results, usage = pt.rerank("query", hits)
    assert len(results) == 3


def test_passthrough_score_type_is_retrieval():
    pt = RetrievalOnlyPassthrough(top_n=2)
    hits = [_make_hit(i) for i in range(5)]
    results, _ = pt.rerank("query", hits)
    assert all(r.score_type == "retrieval" for r in results)


def test_passthrough_no_rerank_score():
    pt = RetrievalOnlyPassthrough(top_n=2)
    hits = [_make_hit(i) for i in range(5)]
    results, _ = pt.rerank("query", hits)
    assert all(r.rerank_score is None for r in results)


def test_passthrough_preserves_order():
    pt = RetrievalOnlyPassthrough(top_n=3)
    hits = [{"id": f"id{i}", "chunk_text": f"text {i}"} for i in range(5)]
    results, _ = pt.rerank("query", hits)
    assert [r.record_id for r in results] == ["id0", "id1", "id2"]


def test_passthrough_empty_hits():
    pt = RetrievalOnlyPassthrough(top_n=3)
    results, _ = pt.rerank("query", [])
    assert results == []


def test_passthrough_top_n_validation():
    with pytest.raises(ValueError):
        RetrievalOnlyPassthrough(top_n=0)
