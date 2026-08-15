import unicodedata
import re


def normalize_text(text: str) -> str:
    """NFC normalize, collapse whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_valid_passage(text: str, min_chars: int = 10) -> bool:
    normalized = normalize_text(text)
    return len(normalized) >= min_chars
