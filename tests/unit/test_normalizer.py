from hhgoa_rag.ingestion.normalizer import content_hash, is_valid_passage, normalize_text


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


def test_content_hash_normalizes_before_hashing():
    a = content_hash("hello  world")
    b = content_hash("hello world")
    assert a == b  # whitespace collapse means same hash


def test_content_hash_different_texts():
    assert content_hash("foo") != content_hash("bar")


def test_content_hash_is_sha256_hex():
    h = content_hash("test")
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)
