import re

DEVANAGARI_RANGE = re.compile(r"[ऀ-ॿ]")
BENGALI_RANGE = re.compile(r"[ঀ-৿]")

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


def detect_script(text: str) -> str:
    if DEVANAGARI_RANGE.search(text):
        return "Devanagari"
    if BENGALI_RANGE.search(text):
        return "Bengali"
    return "Latin"


def detect_language(text: str) -> str:
    """Cheap en/hi/bn detection via Unicode script ranges — no model or
    profile data to load. Sufficient for our 3-language corpus: Devanagari
    and Bengali scripts don't overlap with Latin or each other, so a
    codepoint check is exact for script (not dialect) at this scope.
    Replaces `langdetect`, whose first real call lazily loads ~58MB of
    language-profile data — real memory on a 512MB-constrained deployment.
    """
    script = detect_script(text)
    if script == "Devanagari":
        return "hi"
    if script == "Bengali":
        return "bn"
    return "en"


def get_language_filter(detected_lang: str, hint: str | None) -> list[str]:
    """Return list of language codes for the language metadata filter."""
    lang = hint or detected_lang or "en"
    if lang == "hi":
        return ["hi", "en"]
    if lang == "bn":
        return ["bn", "en"]
    if lang == "en":
        return ["en"]
    if lang in SUPPORTED_LANGUAGES:
        return [lang, "en"]
    return list(SUPPORTED_LANGUAGES)  # conservative: search all
