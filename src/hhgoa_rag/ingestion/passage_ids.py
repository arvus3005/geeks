import hashlib
import uuid

# Namespace for MSMARCO-XI point IDs (DNS namespace reused as a stable namespace)
_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def make_content_hash(normalized_text: str) -> str:
    """SHA-256 of NFC-normalized UTF-8 text."""
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()


def make_point_id(
    dataset_revision: str,
    language: str,
    content_hash: str,
    chunk_strategy_version: str,
    chunk_ordinal: int,
) -> str:
    """Deterministic UUIDv5 Qdrant point ID. Same input = same ID across processes."""
    name = f"{dataset_revision}|{language}|{content_hash}|{chunk_strategy_version}|{chunk_ordinal}"
    return str(uuid.uuid5(_NS, name))


def make_passage_id(split: str, language: str, query_id: str, passage_idx: int) -> str:
    """Legacy occurrence ID for provenance (not used as Qdrant point ID)."""
    raw = f"{split}:{language}:{query_id}:{passage_idx}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]
