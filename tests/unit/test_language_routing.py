from hhgoa_rag.retrieval.language_routing import detect_script, get_language_filter


def test_detect_devanagari():
    assert detect_script("हिन्दी") == "Devanagari"


def test_detect_bengali():
    assert detect_script("বাংলা") == "Bengali"


def test_detect_latin():
    assert detect_script("hello world") == "Latin"


def test_filter_hindi():
    f = get_language_filter("hi", None)
    assert "hi" in f and "en" in f


def test_filter_bengali():
    f = get_language_filter("bn", None)
    assert "bn" in f and "en" in f


def test_filter_english():
    f = get_language_filter("en", None)
    assert "en" in f


def test_hint_overrides():
    f = get_language_filter("en", "hi")
    assert "hi" in f
