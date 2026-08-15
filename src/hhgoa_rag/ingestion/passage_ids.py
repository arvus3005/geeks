import hashlib


def make_passage_id(split: str, language: str, query_id: str, passage_idx: int) -> str:
    """Deterministic passage occurrence ID, namespaced by split+language."""
    raw = f"{split}:{language}:{query_id}:{passage_idx}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def make_content_hash(text: str) -> str:
    """Normalized content hash for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
