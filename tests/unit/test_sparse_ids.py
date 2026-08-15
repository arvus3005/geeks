"""Test stable cross-process sparse token IDs."""

import hashlib

from hhgoa_rag.retrieval.hybrid import _stable_token_id, text_to_sparse


def test_stable_token_id_deterministic():
    a = _stable_token_id("hello")
    b = _stable_token_id("hello")
    assert a == b


def test_stable_token_id_no_builtin_hash():
    expected = int(hashlib.sha256(b"hello").hexdigest()[:5], 16)
    assert _stable_token_id("hello") == expected


def test_sparse_vector_sorted_indices():
    vec = text_to_sparse("hello world foo bar")
    for i in range(len(vec.indices) - 1):
        assert vec.indices[i] <= vec.indices[i + 1], "Indices must be sorted"


def test_sparse_handles_empty():
    vec = text_to_sparse("")
    assert len(vec.indices) > 0  # fallback [0]


def test_sparse_multilingual():
    vec = text_to_sparse("ভারতের রাজধানী নয়াদিল্লি")
    assert len(vec.indices) > 0
