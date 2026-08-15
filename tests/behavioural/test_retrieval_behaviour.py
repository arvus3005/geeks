"""Behavioural retrieval tests — no Qdrant required."""

import numpy as np

from hhgoa_rag.retrieval.embedder import FakeEmbedder
from hhgoa_rag.retrieval.sparse_encoder import BM25SparseEncoder


def _enc() -> BM25SparseEncoder:
    return BM25SparseEncoder()


def test_empty_query_sparse():
    vec = _enc().encode_query("")
    # BM25 on empty string may return empty — just verify it doesn't crash
    assert isinstance(vec.indices, list)


def test_gibberish_sparse():
    vec = _enc().encode_query("xkcd1234!@#$")
    assert isinstance(vec.indices, list)


def test_multilingual_sparse_not_empty():
    vec = _enc().encode_passage("भारत की राजधानी")
    assert len(vec.indices) > 0
    assert any(v > 0 for v in vec.values)


def test_fake_embedder_normalized():
    e = FakeEmbedder()
    v = e.embed_query("hello")
    assert abs(np.linalg.norm(v) - 1.0) < 1e-5


def test_sparse_indices_sorted_passage():
    vec = _enc().encode_passage("a b c d test passage")
    for i in range(len(vec.indices) - 1):
        assert vec.indices[i] <= vec.indices[i + 1]
