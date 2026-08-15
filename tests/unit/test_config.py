import pytest
from hhgoa_rag.config.settings import Settings


def test_default_corpus_mode():
    s = Settings()
    assert s.corpus_mode == "smoke"


def test_corpus_mode_from_env(monkeypatch):
    monkeypatch.setenv("CORPUS_MODE", "full")
    s = Settings()
    assert s.corpus_mode == "full"


def test_max_query_chars_default():
    s = Settings()
    assert s.max_query_chars == 512


def test_qdrant_url_default():
    s = Settings()
    assert "localhost" in s.qdrant_url


def test_missing_sarvam_key_ok():
    # Sarvam key is optional at startup (fails only at STT time)
    s = Settings()
    assert s.sarvam_api_key is None
