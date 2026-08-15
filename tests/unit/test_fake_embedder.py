"""Test FakeEmbedder determinism across 'processes' (simulated)."""

import hashlib

import numpy as np

from hhgoa_rag.retrieval.embedder import FakeEmbedder


def test_fake_embedder_deterministic():
    e = FakeEmbedder()
    a = e.embed_query("test text")
    b = e.embed_query("test text")
    np.testing.assert_array_equal(a, b)


def test_fake_embedder_no_builtin_hash():
    e = FakeEmbedder()
    expected_seed = int(hashlib.sha256(b"test").hexdigest()[:8], 16)
    rng = np.random.default_rng(expected_seed)
    v = rng.standard_normal(384).astype(np.float32)
    v /= np.linalg.norm(v)
    np.testing.assert_allclose(e.embed_query("test"), v, rtol=1e-5)


def test_fake_embedder_different_texts():
    e = FakeEmbedder()
    a = e.embed_query("hello")
    b = e.embed_query("world")
    assert not np.allclose(a, b)


def test_fake_embedder_normalized():
    e = FakeEmbedder()
    v = e.embed_query("hello")
    assert abs(np.linalg.norm(v) - 1.0) < 1e-5
