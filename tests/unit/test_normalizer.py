import pytest
from hhgoa_rag.ingestion.normalizer import normalize_text, is_valid_passage


def test_nfc_normalization_bengali():
    text = "বাংলা"
    result = normalize_text(text)
    import unicodedata

    assert unicodedata.is_normalized("NFC", result)


def test_nfc_normalization_devanagari():
    text = "हिन्दी"
    result = normalize_text(text)
    import unicodedata

    assert unicodedata.is_normalized("NFC", result)


def test_whitespace_collapse():
    assert normalize_text("hello   world") == "hello world"


def test_empty_returns_empty():
    assert normalize_text("") == ""


def test_is_valid_passage_ok():
    assert is_valid_passage("This is a valid passage with enough text.")


def test_is_valid_passage_too_short():
    assert not is_valid_passage("short")
