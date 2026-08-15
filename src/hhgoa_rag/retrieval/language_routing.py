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


def get_language_filter(detected_lang: str, hint: str | None) -> list[str]:
    """Return list of language codes to include in Qdrant filter."""
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
