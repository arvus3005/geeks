import re

DEVANAGARI_RANGE = re.compile(r"[ऀ-ॿ]")  # shared by hi, mr, AND ne -- script alone can't disambiguate
BENGALI_RANGE = re.compile(r"[ঀ-৿]")  # shared by bn AND as (Assamese uses the same Unicode block)
GUJARATI_RANGE = re.compile(r"[઀-૿]")
TAMIL_RANGE = re.compile(r"[஀-௿]")
ARABIC_RANGE = re.compile(r"[؀-ۿ]")  # ur (Urdu) uses Perso-Arabic script
KANNADA_RANGE = re.compile(r"[ಀ-೿]")
MALAYALAM_RANGE = re.compile(r"[ഀ-ൿ]")
ODIA_RANGE = re.compile(r"[଀-୿]")
GURMUKHI_RANGE = re.compile(r"[ਁ-ੴ]")  # pa (Punjabi)

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
INDEXED_LANGUAGES = {"hi", "bn", "gu", "ta", "mr", "ur", "ne", "as", "kn", "ml", "or", "pa", "en"}

# Subset of INDEXED_LANGUAGES built from a row-capped (--max-rows-per-config)
# run, per CLAUDE.md's "every sample artifact must be labeled smoke, pilot,
# or experiment" rule -- these are real, servable shards, just not the full
# corpus for that language (unlike the other 6, which are). "ne" specifically
# was a full run trimmed to its first 500k-passage segment after the full
# build's real disk footprint (measured ~73.5GB projected, 2x mr/ur's density)
# turned out not to fit -- same "not the full corpus" status as the rest of
# this set for serving/labeling purposes, even though it started as a full
# attempt. as/kn/ml/or were built directly as 5,000-row-per-split pilots
# (--max-rows-per-config 5000, ~99k passages each) after disk math showed a
# 5th full language wouldn't fit in the time/disk actually available; pa/sa/te
# were queued for the same treatment but the run was stopped at a 15GB free-
# disk floor before they got a turn (pa's partial segment was incomplete and
# deleted, not counted here). Consulted by system.py's status endpoint and
# README -- keep this updated if more pilot languages are added.
PILOT_LANGUAGES: set[str] = {"ne", "as", "kn", "ml", "or"}


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
    if KANNADA_RANGE.search(text):
        return "Kannada"
    if MALAYALAM_RANGE.search(text):
        return "Malayalam"
    if ODIA_RANGE.search(text):
        return "Odia"
    if GURMUKHI_RANGE.search(text):
        return "Gurmukhi"
    return "Latin"


# Devanagari is shared by Hindi, Marathi, AND (since 2026-08-22) Nepali --
# there is no script-level way to tell them apart without a real statistical
# language ID model (deliberately not used here, see detect_language's
# docstring). Query ALL THREE shards for Devanagari text with no explicit
# hint; the local hybrid store's per-shard fan-out cost is small enough
# (measured ~70ms for a single 32-shard language, and the fan-out is now
# thread-pooled -- see sharded_local_hybrid_store.search) that a third shard
# is still comfortably inside the 200ms budget.
_SCRIPT_TO_LANGS = {
    "Devanagari": ["hi", "mr", "ne"],
    "Bengali": ["bn", "as"],  # Assamese uses the same Unicode block as Bengali
    "Gujarati": ["gu"],
    "Tamil": ["ta"],
    "Arabic": ["ur"],
    "Kannada": ["kn"],
    "Malayalam": ["ml"],
    "Odia": ["or"],
    "Gurmukhi": ["pa"],
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


def get_language_filter(detected_lang: str | None, hint: str | None) -> list[str]:
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
    if lang in ("en", "hi", "mr", "ne"):
        return ["hi", "mr", "ne"]  # covers English pool + 3-way Devanagari ambiguity
    if lang in ("bn", "as"):
        return ["bn", "as", "hi"]  # covers English pool + Bengali/Assamese script ambiguity
    if lang in INDEXED_LANGUAGES:
        return [lang, "hi"]  # own-language shard + hi for the English pool
    if lang in SUPPORTED_LANGUAGES:
        # A real MSMARCO-XI language with no built shard yet -- fall back to
        # hi (English pool) rather than claiming coverage that doesn't exist.
        return ["hi"]
    return list(INDEXED_LANGUAGES)  # conservative: search everything we have
