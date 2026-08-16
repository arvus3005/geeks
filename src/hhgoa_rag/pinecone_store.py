"""Typed Pinecone storage/retrieval boundary.

Wraps Pinecone integrated-embedding index with:
- idempotent record upsert (same IDs are overwritten, not duplicated)
- text-query search via Pinecone server-side embedding
- namespace selection and safety guard (smoke/pilot cannot write to full namespace)
- index statistics / health
- structured provider errors

The indexed text field is ``chunk_text``; the field mapping on the index is
``{"text": "chunk_text"}``, meaning Pinecone embeds the ``chunk_text`` value
when ingesting and the ``text`` input value when searching.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .pinecone_contract import FIELD_MAP as _CANONICAL_FIELD_MAP
from .pinecone_contract import MAX_BATCH_SIZE
from .pinecone_contract import TEXT_FIELD as _CANONICAL_TEXT_FIELD

logger = logging.getLogger(__name__)

# The field_map key used in create_index_for_model AND in search inputs
EMBED_INPUT_FIELD = "text"
# The record field that holds the passage text — derived from canonical contract.
# Callers may use TEXT_RECORD_FIELD for back-compat; it is identical to TEXT_FIELD.
TEXT_RECORD_FIELD: str = _CANONICAL_TEXT_FIELD  # "chunk_text"
# The field_map dict passed to IndexEmbed — derived from canonical contract.
# Callers may use this dict; it is a plain-dict copy of the immutable contract map.
FIELD_MAP: dict[str, str] = dict(_CANONICAL_FIELD_MAP)

FULL_NAMESPACE = "full"
SMOKE_NAMESPACE = "smoke"
PILOT_NAMESPACE_PREFIX = "pilot_"


class PineconeProviderError(Exception):
    """Raised when a Pinecone API call fails after exhausting retries."""

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.cause = cause


@dataclass
class SearchHit:
    id: str
    score: float
    fields: dict[str, Any]

    @property
    def text(self) -> str:
        return self.fields.get(TEXT_RECORD_FIELD, "")

    @property
    def language(self) -> str:
        return str(self.fields.get("language", ""))


def is_safe_namespace(namespace: str) -> bool:
    """True for smoke/pilot namespaces; False for the full-corpus namespace."""
    return namespace == SMOKE_NAMESPACE or namespace.startswith(PILOT_NAMESPACE_PREFIX)


def _check_namespace_guard(namespace: str, context: str) -> None:
    """Refuse writes to the full namespace from non-full contexts."""
    if namespace == FULL_NAMESPACE and context != "full":
        raise ValueError(
            f"Cannot write to namespace '{FULL_NAMESPACE}' in '{context}' context. "
            "Use --confirm-full-ingest and pass context='full' to allow this."
        )


class PineconeStore:
    """Single-index Pinecone client with integrated multilingual embedding.

    Retry policy ownership
    ----------------------
    ``upsert_records()`` makes **exactly one** SDK call per invocation.
    It does NOT retry internally.  The ingestion orchestration layer (engine.py,
    ingest_prepared.py) owns the retry policy so that each attempt can be
    atomically recorded in the state DB before the call is made.

    ``max_retries`` and ``retry_backoff`` are accepted for API compatibility
    but are ignored for upserts.  They are retained so existing callers that
    pass keyword arguments do not break.
    """

    def __init__(
        self,
        index: Any,  # pinecone.Index
        embed_model: str,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
        upsert_timeout: float = 30.0,
        search_timeout: float = 10.0,
    ) -> None:
        self._index = index
        self.embed_model = embed_model
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._upsert_timeout = upsert_timeout
        self._search_timeout = search_timeout

    def upsert_records(
        self,
        records: list[dict[str, Any]],
        namespace: str,
        context: str = "smoke",
    ) -> int:
        """Idempotent upsert — same ID overwrites the existing record.

        Makes **exactly one** SDK call.  The caller is responsible for retry
        logic.  Raises the raw SDK exception on failure so the caller can
        decide whether to retry and can record the attempt before doing so.

        context must be 'full' to write to the full namespace.
        Returns the number of records submitted.
        """
        _check_namespace_guard(namespace, context)
        if not records:
            return 0

        # Defense-in-depth: validate batch limits before the SDK call.
        if len(records) > MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch has {len(records)} records, exceeding the {MAX_BATCH_SIZE}-record Pinecone limit. "
                "Split the batch before calling upsert_records."
            )

        self._index.upsert_records(
            records=records,
            namespace=namespace,
            timeout=self._upsert_timeout,
        )
        return len(records)

    def search(
        self,
        query_text: str,
        top_k: int,
        namespace: str,
        filter: dict[str, Any] | None = None,
        fields: list[str] | None = None,
    ) -> list[SearchHit]:
        """Text search using Pinecone server-side integrated embedding."""
        kwargs: dict[str, Any] = {
            "namespace": namespace,
            "inputs": {EMBED_INPUT_FIELD: query_text},
            "top_k": top_k,
            "timeout": self._search_timeout,
        }
        if filter:
            kwargs["filter"] = filter
        if fields:
            kwargs["fields"] = fields

        try:
            resp = self._index.search_records(**kwargs)
        except Exception as e:
            raise PineconeProviderError(f"Pinecone search failed: {e}", cause=e) from e

        return [
            SearchHit(
                id=hit.id,
                score=float(hit.score),
                fields=dict(hit.fields) if hit.fields else {},
            )
            for hit in resp.result.hits
        ]

    def describe_index_stats(self) -> dict[str, Any]:
        """Return index statistics including per-namespace vector counts."""
        try:
            resp = self._index.describe_index_stats()
            return resp.to_dict() if hasattr(resp, "to_dict") else dict(resp)
        except Exception as e:
            raise PineconeProviderError(f"describe_index_stats failed: {e}", cause=e) from e

    def count_namespace(self, namespace: str) -> int:
        """Return the vector count for a specific namespace. Returns 0 if namespace absent."""
        try:
            stats = self._index.describe_index_stats()
            ns_map = getattr(stats, "namespaces", None) or {}
            ns_info = ns_map.get(namespace)
            if ns_info is None:
                return 0
            return int(getattr(ns_info, "vector_count", 0))
        except Exception:
            return 0

    def list_vector_ids(
        self,
        namespace: str,
        prefix: str | None = None,
        limit: int = 100,
    ) -> list[str]:
        """List all vector IDs in a namespace with complete pagination handling.

        Raises PineconeProviderError if enumeration is unsupported, fails, or
        returns malformed data.
        """
        all_ids: list[str] = []
        pagination_token: str | None = None

        if hasattr(self._index, "list_paginated"):
            while True:
                kwargs: dict[str, Any] = {
                    "namespace": namespace,
                    "limit": limit,
                    "timeout": self._search_timeout,
                }
                if prefix:
                    kwargs["prefix"] = prefix
                if pagination_token:
                    kwargs["pagination_token"] = pagination_token

                try:
                    resp = self._index.list_paginated(**kwargs)
                except Exception as e:
                    raise PineconeProviderError(
                        f"list_paginated failed for namespace '{namespace}': {e}",
                        cause=e,
                    ) from e

                if resp is None:
                    raise PineconeProviderError(
                        f"list_paginated returned None for namespace '{namespace}'"
                    )

                vectors = getattr(resp, "vectors", None)
                if vectors is None and isinstance(resp, dict):
                    vectors = resp.get("vectors")
                if vectors is None:
                    raise PineconeProviderError(
                        f"list_paginated response missing 'vectors' field for namespace '{namespace}'"
                    )

                for item in vectors:
                    item_id = getattr(item, "id", None)
                    if item_id is None and isinstance(item, dict):
                        item_id = item.get("id")
                    elif isinstance(item, str):
                        item_id = item
                    if item_id is None or not isinstance(item_id, str):
                        raise PineconeProviderError(
                            f"Malformed vector item in list response: {item!r}"
                        )
                    all_ids.append(item_id)

                pagination = getattr(resp, "pagination", None)
                if pagination is None and isinstance(resp, dict):
                    pagination = resp.get("pagination")

                next_token: str | None = None
                if pagination is not None:
                    next_token = getattr(pagination, "next", None)
                    if next_token is None and isinstance(pagination, dict):
                        next_token = pagination.get("next")

                if not next_token:
                    break
                pagination_token = next_token

            return all_ids

        elif hasattr(self._index, "list"):
            try:
                pages = self._index.list(
                    namespace=namespace, prefix=prefix, timeout=self._search_timeout
                )
                for page in pages:
                    vectors = getattr(page, "vectors", None)
                    if vectors is None and isinstance(page, dict):
                        vectors = page.get("vectors")
                    elif isinstance(page, list | tuple):
                        vectors = page
                    if vectors is None:
                        raise PineconeProviderError(f"Malformed page in list iterator: {page!r}")
                    for item in vectors:
                        item_id = getattr(item, "id", None)
                        if item_id is None and isinstance(item, dict):
                            item_id = item.get("id")
                        elif isinstance(item, str):
                            item_id = item
                        if item_id is None or not isinstance(item_id, str):
                            raise PineconeProviderError(
                                f"Malformed vector item in list response: {item!r}"
                            )
                        all_ids.append(item_id)
                return all_ids
            except Exception as e:
                if isinstance(e, PineconeProviderError):
                    raise
                raise PineconeProviderError(
                    f"list iterator failed for namespace '{namespace}': {e}",
                    cause=e,
                ) from e
        else:
            raise PineconeProviderError(
                "Index object does not support ID enumeration (neither list_paginated nor list found)"
            )

    def health(self) -> dict[str, Any]:
        """Return a health-check dict. Raises PineconeProviderError on failure."""
        stats = self.describe_index_stats()
        return {"status": "ok", "index_stats": stats}
