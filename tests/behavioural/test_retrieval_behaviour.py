"""Behavioural retrieval tests — no Qdrant required."""

import numpy as np

from hhgoa_rag.retrieval.embedder import FakeEmbedder
from hhgoa_rag.retrieval.hybrid import text_to_sparse


def test_empty_query_sparse():
    vec = text_to_sparse("")
    assert len(vec.indices) > 0  # has fallback


def test_gibberish_sparse():
    vec = text_to_sparse("xkcd1234!@#$")
    assert isinstance(vec.indices, list)


def test_multilingual_sparse_not_empty():
    vec = text_to_sparse("भारत की राजधानी")
    assert len(vec.indices) > 0
    assert any(v > 0 for v in vec.values)


def test_fake_embedder_normalized():
    e = FakeEmbedder()
    v = e.embed_query("hello")
    assert abs(np.linalg.norm(v) - 1.0) < 1e-5


def test_sparse_values_sum_to_one_for_uniform_text():
    vec = text_to_sparse("a b c d")
    # Each token appears once, 4 tokens, so each value = 1/4
    # (assuming no collisions)
    assert abs(sum(vec.values) - 1.0) < 1e-5
