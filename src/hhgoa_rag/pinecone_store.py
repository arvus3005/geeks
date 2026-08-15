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
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# The field_map key used in create_index_for_model AND in search inputs
EMBED_INPUT_FIELD = "text"
# The record field that holds the passage text
TEXT_RECORD_FIELD = "chunk_text"
# The field_map dict passed to IndexEmbed
FIELD_MAP: dict[str, str] = {EMBED_INPUT_FIELD: TEXT_RECORD_FIELD}

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
    """Single-index Pinecone client with integrated multilingual embedding."""

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

        context must be 'full' to write to the full namespace.
        Returns the number of records submitted.
        """
        _check_namespace_guard(namespace, context)
        if not records:
            return 0

        for attempt in range(self._max_retries + 1):
            try:
                self._index.upsert_records(
                    records=records,
                    namespace=namespace,
                    timeout=self._upsert_timeout,
                )
                return len(records)
            except Exception as e:
                if attempt < self._max_retries:
                    wait = self._retry_backoff**attempt
                    logger.warning(
                        "Pinecone upsert attempt %d/%d failed, retrying in %.1fs: %s",
                        attempt + 1,
                        self._max_retries,
                        wait,
                        e,
                    )
                    time.sleep(wait)
                else:
                    raise PineconeProviderError(
                        f"Pinecone upsert permanently failed after {self._max_retries} retries",
                        cause=e,
                    ) from e
        return 0  # unreachable

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

    def health(self) -> dict[str, Any]:
        """Return a health-check dict. Raises PineconeProviderError on failure."""
        stats = self.describe_index_stats()
        return {"status": "ok", "index_stats": stats}
