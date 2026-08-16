"""Tests for the multilingual-E5 tokenizer module.

These tests are skipped if HuggingFace is unavailable.
They do NOT test the smoke/pilot tokenizer in a live scenario — unit only.
"""

from __future__ import annotations

import pytest


def _has_hf() -> bool:
    try:
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


HF_AVAILABLE = _has_hf()


@pytest.fixture(scope="module")
def tokenizer():
    if not HF_AVAILABLE:
        pytest.skip("transformers not installed")
    from hhgoa_rag.ingestion.tokenizer import get_tokenizer

    try:
        return get_tokenizer()
    except RuntimeError as e:
        pytest.skip(f"Tokenizer unavailable: {e}")


def test_tokenizer_returns_positive_count(tokenizer):
    n = tokenizer.count_tokens("The capital of India is New Delhi.")
    assert n > 0


def test_tokenizer_count_includes_special_tokens(tokenizer):
    """Token count with prefix must exceed raw wordpiece count."""
    n_with_prefix = tokenizer.count_tokens("hello world", add_prefix=True)
    n_without = tokenizer.count_tokens("hello world", add_prefix=False)
    assert n_with_prefix > n_without


def test_tokenizer_hindi_text(tokenizer):
    n = tokenizer.count_tokens("भारत की राजधानी नई दिल्ली है।")
    assert n > 0


def test_tokenizer_bengali_text(tokenizer):
    n = tokenizer.count_tokens("ভারতের রাজধানী নতুন দিল্লি।")
    assert n > 0


def test_tokenizer_fingerprint_is_hex_string(tokenizer):
    fp = tokenizer.fingerprint
    assert isinstance(fp, str)
    assert len(fp) == 16
    int(fp, 16)  # must be valid hex


def test_fits_model_limit_short_text(tokenizer):
    assert tokenizer.fits_model_limit("Hello world")


def test_exceeds_model_limit_very_long_text(tokenizer):
    # 2000 repetitions of "word " easily exceeds 512 tokens
    long_text = "word " * 2000
    assert not tokenizer.fits_model_limit(long_text)


def test_model_input_limit_is_512(tokenizer):
    assert tokenizer.model_input_limit == 512


def test_tokenizer_info_has_correct_repo(tokenizer):
    info = tokenizer.info
    assert "multilingual-e5" in info.repo.lower()


def test_tokenizer_singleton_cached():
    """Two calls to get_tokenizer() return the same object."""
    if not HF_AVAILABLE:
        pytest.skip("transformers not installed")
    from hhgoa_rag.ingestion.tokenizer import get_tokenizer

    try:
        a = get_tokenizer()
        b = get_tokenizer()
    except RuntimeError:
        pytest.skip("Tokenizer unavailable")
    assert a is b
