import hashlib
import re
import unicodedata


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_valid_passage(text: str, min_chars: int = 10) -> bool:
    return len(normalize_text(text)) >= min_chars


def content_hash(raw_text: str) -> str:
    """Always normalizes before hashing — callers cannot accidentally hash differently."""
    normalized = normalize_text(raw_text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
