"""Authoritative Pinecone record schema for MSMARCO-XI ingestion.

All ingestion paths (smoke, canary, pilot) MUST build records through
`build_record()` and validate them through `validate_record()`.

Forbidden fields are recursively checked — raw HuggingFace dataset dicts
must never pass through to the Pinecone layer.
"""

from __future__ import annotations

from typing import Any

from ..pinecone_store import TEXT_RECORD_FIELD

# Fields that must exist in every indexed record
REQUIRED_FIELDS: frozenset[str] = frozenset(
    {
        "id",
        TEXT_RECORD_FIELD,  # "chunk_text"
        "language",
        "config_language",
        "dataset_revision",
        "split",
        "physical_shard",
        "local_source_row",
        "passage_position",
        "parent_passage_id",
        "content_hash",
        "chunk_strategy",
        "chunk_strategy_version",
        "chunk_ordinal",
        "chunk_total",
        "token_length",
        "tokenizer_fingerprint",
        "manifest_id",
    }
)

# Fields from MSMARCO-XI evaluation annotations that must NEVER appear
FORBIDDEN_FIELDS: frozenset[str] = frozenset(
    {
        "query",
        "Answer",
        "Eng_Query",
        "Eng_Answer",
        "query_type",
        "is_selected",
    }
)


class SchemaViolationError(Exception):
    """Raised when a record violates the authoritative schema."""


def build_record(
    *,
    record_id: str,
    chunk_text: str,
    language: str,
    config_language: str,
    dataset_revision: str,
    split: str,
    physical_shard: str,
    local_source_row: int,
    passage_position: int,
    parent_passage_id: str,
    content_hash: str,
    chunk_strategy: str,
    chunk_strategy_version: str,
    chunk_ordinal: int,
    chunk_total: int,
    token_length: int,
    tokenizer_fingerprint: str,
    manifest_id: str,
) -> dict[str, Any]:
    """Build a validated Pinecone record.  Raises SchemaViolationError on violation."""
    record: dict[str, Any] = {
        "id": record_id,
        TEXT_RECORD_FIELD: chunk_text,
        "language": language,
        "config_language": config_language,
        "dataset_revision": dataset_revision,
        "split": split,
        "physical_shard": physical_shard,
        "local_source_row": local_source_row,
        "passage_position": passage_position,
        "parent_passage_id": parent_passage_id,
        "content_hash": content_hash,
        "chunk_strategy": chunk_strategy,
        "chunk_strategy_version": chunk_strategy_version,
        "chunk_ordinal": chunk_ordinal,
        "chunk_total": chunk_total,
        "token_length": token_length,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "manifest_id": manifest_id,
    }
    validate_record(record)
    return record


def validate_record(record: dict[str, Any]) -> None:
    """Raise SchemaViolationError if the record violates the authoritative schema."""
    _check_forbidden_recursive(record, path="record")
    missing = REQUIRED_FIELDS - set(record.keys())
    if missing:
        raise SchemaViolationError(f"Record missing required fields: {sorted(missing)}")
    if not record.get("id"):
        raise SchemaViolationError("Record 'id' must be a non-empty string")
    if not record.get(TEXT_RECORD_FIELD):
        raise SchemaViolationError(f"Record '{TEXT_RECORD_FIELD}' must be a non-empty string")
    if not isinstance(record.get("token_length"), int) or record["token_length"] <= 0:
        raise SchemaViolationError("Record 'token_length' must be a positive integer")


def _check_forbidden_recursive(obj: Any, path: str) -> None:
    """Recursively scan obj for forbidden field names. Raises SchemaViolationError."""
    if isinstance(obj, dict):
        bad = FORBIDDEN_FIELDS & set(obj.keys())
        if bad:
            raise SchemaViolationError(
                f"Forbidden eval fields found at {path}: {sorted(bad)}. "
                "These fields must never enter Pinecone records."
            )
        for k, v in obj.items():
            _check_forbidden_recursive(v, f"{path}.{k}")
    elif isinstance(obj, list | tuple):
        for i, item in enumerate(obj):
            _check_forbidden_recursive(item, f"{path}[{i}]")


def check_forbidden_fields(record: dict[str, Any]) -> list[str]:
    """Return list of forbidden fields found recursively (empty = clean)."""
    found: list[str] = []
    _collect_forbidden(record, found)
    return found


def _collect_forbidden(obj: Any, found: list[str]) -> None:
    if isinstance(obj, dict):
        bad = FORBIDDEN_FIELDS & set(obj.keys())
        found.extend(sorted(bad))
        for v in obj.values():
            _collect_forbidden(v, found)
    elif isinstance(obj, list | tuple):
        for item in obj:
            _collect_forbidden(item, found)
