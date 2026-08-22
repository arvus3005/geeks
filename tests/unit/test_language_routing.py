from hhgoa_rag.retrieval.language_routing import detect_script, get_language_filter


def test_detect_devanagari():
    assert detect_script("हिन्दी") == "Devanagari"


def test_detect_bengali():
    assert detect_script("বাংলা") == "Bengali"


def test_detect_latin():
    assert detect_script("hello world") == "Latin"


def test_filter_hindi():
    # Filter returns shard-GROUP codes (full_local_index/ directory
    # prefixes), not semantic language labels. "en" isn't its own shard
    # group -- every MSMARCO-XI config's shared English pool lives only
    # inside hi's segments (see README's indexing-status section) -- so
    # Hindi's filter is ["hi", "mr"] (mr for the Devanagari script
    # ambiguity), which also covers the English pool via hi.
    f = get_language_filter("hi", None)
    assert "hi" in f and "mr" in f


def test_filter_bengali():
    # Own-language shard + hi (the only place the English pool is indexed).
    f = get_language_filter("bn", None)
    assert "bn" in f and "hi" in f


def test_filter_english():
    # No "en" shard group exists -- English routes to hi, the only shard
    # group that actually contains English passages.
    f = get_language_filter("en", None)
    assert "hi" in f


def test_hint_overrides():
    f = get_language_filter("en", "hi")
    assert "hi" in f
