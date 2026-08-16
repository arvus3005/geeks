"""Canonical Pinecone index contract — single source of truth.

All lifecycle, validation, ingestion, and reporting code MUST import constants
from here.  Do NOT hand-write contract dicts anywhere else in the codebase.

Immutable fields:
    INDEX_NAME          : "msmarco-xi"
    NAMESPACE           : "pilot_v1"
    CLOUD               : "aws"
    REGION              : "us-east-1"
    MODEL               : "multilingual-e5-large"
    DIMENSION           : 1024
    METRIC              : "cosine"
    TEXT_FIELD          : "chunk_text"
    FIELD_MAP           : {"text": "chunk_text"}  (MappingProxyType — immutable)
    WRITE_PARAMETERS    : {"input_type": "passage", "truncate": "NONE"}  (immutable)
    READ_PARAMETERS     : {"input_type": "query", "truncate": "NONE"}  (immutable)
    MAX_INPUT_TOKENS    : 507
    MAX_BATCH_SIZE      : 96
    DATASET_REPO        : "ai4bharat/MSMARCO-XI"
    DATASET_REVISION    : "bf5cdc1f26e581e519018e434db14edd1b77602b"
    TOKENIZER_REPO      : "intfloat/multilingual-e5-large"
    TOKENIZER_REVISION  : "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3"

    CONTRACT_VERSION    : "1"
    MANIFEST_SCHEMA_VERSION : "3"
"""

from __future__ import annotations

import copy
import hashlib
import json
from types import MappingProxyType
from typing import Final

# ---------------------------------------------------------------------------
# Contract version
# ---------------------------------------------------------------------------

CONTRACT_VERSION: Final[str] = "1"
MANIFEST_SCHEMA_VERSION: Final[str] = "3"

# ---------------------------------------------------------------------------
# Immutable index identity
# ---------------------------------------------------------------------------

INDEX_NAME: Final[str] = "msmarco-xi"
NAMESPACE: Final[str] = "pilot_v1"
CLOUD: Final[str] = "aws"
REGION: Final[str] = "us-east-1"

# ---------------------------------------------------------------------------
# Embedding model contract
# ---------------------------------------------------------------------------

MODEL: Final[str] = "multilingual-e5-large"
DIMENSION: Final[int] = 1024
METRIC: Final[str] = "cosine"
TEXT_FIELD: Final[str] = "chunk_text"

# Structurally immutable mappings — callers must not mutate these.
# At SDK/JSON boundaries, use dict(FIELD_MAP) or canonical_contract() to get plain dicts.
FIELD_MAP: Final[MappingProxyType[str, str]] = MappingProxyType({"text": "chunk_text"})
WRITE_PARAMETERS: Final[MappingProxyType[str, str]] = MappingProxyType(
    {"input_type": "passage", "truncate": "NONE"}
)
READ_PARAMETERS: Final[MappingProxyType[str, str]] = MappingProxyType(
    {"input_type": "query", "truncate": "NONE"}
)

# ---------------------------------------------------------------------------
# Hard limits
# ---------------------------------------------------------------------------

MAX_INPUT_TOKENS: Final[int] = 507
MAX_BATCH_SIZE: Final[int] = 96

# ---------------------------------------------------------------------------
# Dataset / tokenizer provenance
# ---------------------------------------------------------------------------

DATASET_REPO: Final[str] = "ai4bharat/MSMARCO-XI"
DATASET_REVISION: Final[str] = "bf5cdc1f26e581e519018e434db14edd1b77602b"
TOKENIZER_REPO: Final[str] = "intfloat/multilingual-e5-large"
TOKENIZER_REVISION: Final[str] = "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3"


# ---------------------------------------------------------------------------
# Canonical dict representation
# ---------------------------------------------------------------------------


def canonical_contract() -> dict:
    """Return the canonical contract as a fresh, stable, sorted, JSON-serialisable dict.

    Always returns a deep copy so callers cannot mutate canonical state.
    The structure is deterministic: keys are sorted before serialisation so
    that :func:`contract_fingerprint` produces the same SHA-256 regardless of
    insertion order.
    """
    return copy.deepcopy(
        {
            "contract_version": CONTRACT_VERSION,
            "cloud": CLOUD,
            "dataset_repo": DATASET_REPO,
            "dataset_revision": DATASET_REVISION,
            "dimension": DIMENSION,
            "field_map": dict(FIELD_MAP),
            "index_name": INDEX_NAME,
            "max_batch_size": MAX_BATCH_SIZE,
            "max_input_tokens": MAX_INPUT_TOKENS,
            "metric": METRIC,
            "model": MODEL,
            "namespace": NAMESPACE,
            "read_parameters": dict(READ_PARAMETERS),
            "region": REGION,
            "text_field": TEXT_FIELD,
            "tokenizer_repo": TOKENIZER_REPO,
            "tokenizer_revision": TOKENIZER_REVISION,
            "write_parameters": dict(WRITE_PARAMETERS),
        }
    )


def canonical_contract_json() -> str:
    """Return stable canonical JSON (keys sorted, ASCII-safe)."""
    return json.dumps(canonical_contract(), sort_keys=True, ensure_ascii=True)


def contract_fingerprint() -> str:
    """Return the SHA-256 hex digest of the canonical contract JSON."""
    return hashlib.sha256(canonical_contract_json().encode("utf-8")).hexdigest()
