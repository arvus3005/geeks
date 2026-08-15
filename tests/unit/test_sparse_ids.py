"""Test stable cross-process sparse encoding (FastEmbed BM25)."""

from qdrant_client.models import SparseVector

from hhgoa_rag.retrieval.sparse_encoder import BM25SparseEncoder


def _enc() -> BM25SparseEncoder:
    return BM25SparseEncoder()


def test_passage_encode_returns_sparse_vector():
    enc = _enc()
    vec = enc.encode_passage("The capital of India is New Delhi.")
    assert isinstance(vec, SparseVector)
    assert len(vec.indices) > 0
    assert len(vec.values) > 0


def test_query_encode_returns_sparse_vector():
    enc = _enc()
    vec = enc.encode_query("capital India")
    assert isinstance(vec, SparseVector)
    assert len(vec.indices) > 0


def test_indices_sorted():
    enc = _enc()
    vec = enc.encode_passage("hello world foo bar baz")
    for i in range(len(vec.indices) - 1):
        assert vec.indices[i] <= vec.indices[i + 1], "Indices must be sorted"


def test_deterministic_across_calls():
    enc = _enc()
    a = enc.encode_passage("test passage about India")
    b = enc.encode_passage("test passage about India")
    assert a.indices == b.indices
    assert a.values == b.values


def test_passage_and_query_have_same_indices_for_same_text():
    """BM25 passage vs query may differ in values but should share most indices."""
    enc = _enc()
    p = enc.encode_passage("India")
    q = enc.encode_query("India")
    # At least one index in common
    assert set(p.indices) & set(q.indices), "Passage and query share no token indices"


def test_batch_matches_individual():
    enc = _enc()
    texts = ["capital of India", "Ganges river", "Mount Everest"]
    batch = enc.encode_passages_batch(texts)
    for text, bvec in zip(texts, batch):
        single = enc.encode_passage(text)
        assert bvec.indices == single.indices
        assert bvec.values == single.values


def test_multilingual_not_empty():
    enc = _enc()
    vec = enc.encode_passage("ভারতের রাজধানী নয়াদিল্লি")
    assert len(vec.indices) > 0


def test_model_name():
    enc = _enc()
    assert enc.model_name == "Qdrant/bm25"
