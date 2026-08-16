"""Authoritative Pinecone record schema for MSMARCO-XI ingestion.

All ingestion paths (smoke, canary, pilot) MUST build records through
`build_record()` and validate them through `validate_record()`.

Forbidden fields are recursively checked — raw HuggingFace dataset dicts
must never pass through to the Pinecone layer.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..pinecone_store import TEXT_RECORD_FIELD

# Languages permitted in prepared records.
ALLOWED_LANGUAGES: frozenset[str] = frozenset({"en", "hi", "bn"})

# Pinecone limits (integrated dense embedding, Starter).
MAX_ID_LENGTH = 512
MAX_METADATA_BYTES = 40_960

_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")

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

    # Non-empty string ID with Pinecone length bound.
    record_id = record.get("id")
    if not isinstance(record_id, str) or not record_id:
        raise SchemaViolationError("Record 'id' must be a non-empty string")
    if len(record_id) > MAX_ID_LENGTH:
        raise SchemaViolationError(
            f"Record 'id' exceeds Pinecone max length {MAX_ID_LENGTH}: {len(record_id)}"
        )

    # Non-empty text.
    text = record.get(TEXT_RECORD_FIELD)
    if not isinstance(text, str) or not text.strip():
        raise SchemaViolationError(f"Record '{TEXT_RECORD_FIELD}' must be a non-empty string")

    # Language must be allowed.
    language = record.get("language")
    if language not in ALLOWED_LANGUAGES:
        raise SchemaViolationError(
            f"Record 'language' {language!r} not in allowed {sorted(ALLOWED_LANGUAGES)}"
        )

    # Positive token length.
    if not isinstance(record.get("token_length"), int) or record["token_length"] <= 0:
        raise SchemaViolationError("Record 'token_length' must be a positive integer")

    # chunk_ordinal / chunk_total consistency: 0 <= ordinal < total.
    ordinal = record.get("chunk_ordinal")
    total = record.get("chunk_total")
    if not isinstance(ordinal, int) or not isinstance(total, int):
        raise SchemaViolationError("chunk_ordinal and chunk_total must be integers")
    if total <= 0:
        raise SchemaViolationError(f"chunk_total must be positive, got {total}")
    if not (0 <= ordinal < total):
        raise SchemaViolationError(
            f"chunk_ordinal must satisfy 0 <= ordinal < total; got ordinal={ordinal}, total={total}"
        )

    # Dataset revision must be a full 40-char hex commit SHA.
    revision = record.get("dataset_revision")
    if not isinstance(revision, str) or not _HEX40_RE.match(revision):
        raise SchemaViolationError(
            f"dataset_revision must be a full 40-char hex SHA, got {revision!r}"
        )

    # Bounded metadata payload for Pinecone.
    metadata_bytes = len(
        json.dumps({k: v for k, v in record.items() if k != "id"}, ensure_ascii=False).encode(
            "utf-8"
        )
    )
    if metadata_bytes > MAX_METADATA_BYTES:
        raise SchemaViolationError(
            f"Record metadata {metadata_bytes} bytes exceeds Pinecone limit {MAX_METADATA_BYTES}"
        )


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
