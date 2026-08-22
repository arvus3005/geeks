from hhgoa_rag.retrieval.language_routing import detect_language, detect_script, get_language_filter


def test_detect_devanagari():
    assert detect_script("हिन्दी") == "Devanagari"


def test_detect_bengali():
    assert detect_script("বাংলা") == "Bengali"


def test_detect_latin():
    assert detect_script("hello world") == "Latin"


def test_detect_kannada():
    assert detect_script("ಕನ್ನಡ") == "Kannada"


def test_detect_malayalam():
    assert detect_script("മലയാളം") == "Malayalam"


def test_detect_odia():
    assert detect_script("ଓଡ଼ିଆ") == "Odia"


def test_detect_gurmukhi():
    assert detect_script("ਪੰਜਾਬੀ") == "Gurmukhi"


# detect_language exercises the _SCRIPT_TO_LANGS mapping, not just the
# regex range -- a script range can be defined but never wired into that
# dict (this happened for real with Gurmukhi during development: the range
# matched correctly but detect_language still fell through to "en" because
# _SCRIPT_TO_LANGS had no "Gurmukhi" key). These would have caught it.
def test_detect_language_kannada():
    assert detect_language("ಕನ್ನಡ") == "kn"


def test_detect_language_malayalam():
    assert detect_language("മലയാളം") == "ml"


def test_detect_language_odia():
    assert detect_language("ଓଡ଼ିଆ") == "or"


def test_detect_language_gurmukhi():
    assert detect_language("ਪੰਜਾਬੀ") == "pa"


def test_detect_language_nepali_defaults_to_hi_in_devanagari_ambiguity():
    # ne shares Devanagari with hi/mr; detect_language's single-guess return
    # is "hi" first by design (see _SCRIPT_TO_LANGS ordering) -- the real
    # ne coverage comes from get_language_filter fanning out to all three,
    # not from this function guessing "ne" for Devanagari text.
    assert detect_language("नेपाली") == "hi"


def test_filter_as_kn_ml_or_pa_each_reach_own_shard():
    for lang in ("kn", "ml", "or", "pa"):
        f = get_language_filter(None, lang)
        assert lang in f and "hi" in f


def test_filter_assamese_reaches_bn_and_as():
    f = get_language_filter(None, "as")
    assert "as" in f and "bn" in f and "hi" in f


def test_filter_nepali_hint_reaches_all_three_devanagari_shards():
    f = get_language_filter(None, "ne")
    assert "ne" in f and "hi" in f and "mr" in f


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
