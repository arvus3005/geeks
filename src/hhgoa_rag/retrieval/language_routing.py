import re

DEVANAGARI_RANGE = re.compile(r"[ऀ-ॿ]")  # shared by hi AND mr -- script alone can't disambiguate
BENGALI_RANGE = re.compile(r"[ঀ-৿]")
GUJARATI_RANGE = re.compile(r"[઀-૿]")
TAMIL_RANGE = re.compile(r"[஀-௿]")
ARABIC_RANGE = re.compile(r"[؀-ۿ]")  # ur (Urdu) uses Perso-Arabic script

SUPPORTED_LANGUAGES = {
    "as",
    "bn",
    "gu",
    "hi",
    "kn",
    "ml",
    "mr",
    "ne",
    "or",
    "pa",
    "sa",
    "ta",
    "te",
    "ur",
    "en",
}

# Languages that actually have a built local index shard as of 2026-08-22
# (see README's indexing-status table). Querying a language outside this
# set falls back to "en" rather than a shard that doesn't exist.
INDEXED_LANGUAGES = {"hi", "bn", "gu", "ta", "mr", "ur", "en"}


def detect_script(text: str) -> str:
    if DEVANAGARI_RANGE.search(text):
        return "Devanagari"
    if BENGALI_RANGE.search(text):
        return "Bengali"
    if GUJARATI_RANGE.search(text):
        return "Gujarati"
    if TAMIL_RANGE.search(text):
        return "Tamil"
    if ARABIC_RANGE.search(text):
        return "Arabic"
    return "Latin"


# Devanagari is shared by Hindi and Marathi -- there is no script-level way
# to tell them apart without a real statistical language ID model (deliberately
# not used here, see detect_language's docstring). Query BOTH shards for
# Devanagari text; the local hybrid store's per-shard fan-out cost is small
# enough (measured ~70ms for a single 32-shard language) that querying two
# languages' shards together is still comfortably inside the 200ms budget.
_SCRIPT_TO_LANGS = {
    "Devanagari": ["hi", "mr"],
    "Bengali": ["bn"],
    "Gujarati": ["gu"],
    "Tamil": ["ta"],
    "Arabic": ["ur"],
}


def detect_language(text: str) -> str:
    """Cheap script-range detection -- no model or profile data to load.
    Replaces `langdetect`, whose first real call lazily loads ~58MB of
    language-profile data. Returns a single best-guess code for logging /
    the API response's `detected_language` field; use get_language_filter
    for the actual shard-selection list, which handles the Devanagari
    hi/mr ambiguity properly instead of guessing one.
    """
    script = detect_script(text)
    langs = _SCRIPT_TO_LANGS.get(script)
    return langs[0] if langs else "en"


def get_language_filter(detected_lang: str, hint: str | None) -> list[str]:
    """Return the list of shard-group codes (matching full_local_index/
    directory prefixes) that should be queried for this language.

    NOTE: "en" is not itself a shard-group prefix -- every MSMARCO-XI
    config's shared English passage pool was measured to live ONLY inside
    hi's segments (see README's indexing-status section), not duplicated
    into any other language's shards. So an "en"-detected query, and every
    other indexed language's English fallback, both route to the "hi"
    shard group -- that's the only place English passages are indexed.
    """
    raw = hint or detected_lang or "en"
    lang = raw.split("-")[0].lower()
    if lang in ("en", "hi", "mr"):
        return ["hi", "mr"]  # covers English pool + Devanagari ambiguity, see _SCRIPT_TO_LANGS
    if lang in INDEXED_LANGUAGES:
        return [lang, "hi"]  # own-language shard + hi for the English pool
    if lang in SUPPORTED_LANGUAGES:
        # A real MSMARCO-XI language with no built shard yet -- fall back to
        # hi (English pool) rather than claiming coverage that doesn't exist.
        return ["hi"]
    return list(INDEXED_LANGUAGES)  # conservative: search everything we have
