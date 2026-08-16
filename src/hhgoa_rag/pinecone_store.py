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

# Conservative guard against runaway pagination during ID enumeration.
# At the default page size of 100 IDs, this bounds enumeration to ~1M IDs —
# far above any canary/pilot namespace, while preventing infinite loops.
MAX_ENUMERATION_PAGES = 10_000


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
        """Return the vector count for a specific namespace.

        Fail-closed: returns 0 ONLY when a valid Pinecone statistics response
        proves the namespace is absent or empty. Provider exceptions, malformed
        responses, and invalid counts (booleans, negative, non-integer) raise
        PineconeProviderError. An exception is NEVER interpreted as "empty".
        """
        try:
            stats = self._index.describe_index_stats()
        except Exception as e:
            raise PineconeProviderError(
                f"count_namespace failed for namespace '{namespace}': "
                f"provider error while fetching index statistics: {e}",
                cause=e,
            ) from e

        namespaces = getattr(stats, "namespaces", None)
        if namespaces is None and isinstance(stats, dict):
            namespaces = stats.get("namespaces")
        if namespaces is None or not isinstance(namespaces, dict):
            raise PineconeProviderError(
                f"count_namespace received a malformed statistics response for "
                f"namespace '{namespace}': missing or non-dict 'namespaces' field."
            )

        ns_info = namespaces.get(namespace)
        if ns_info is None:
            # A valid response that omits the namespace proves it is absent/empty.
            return 0

        vc = getattr(ns_info, "vector_count", None)
        if vc is None and isinstance(ns_info, dict):
            vc = ns_info.get("vector_count")

        # Strict type acceptance: only a genuine, non-negative Python ``int``
        # is valid. ``bool`` (a subclass of int), floats (300.0, 300.9),
        # numeric strings ("300"), None, and negatives are all malformed and
        # must NOT be coerced. Coercion would silently accept
        # ``300.9 -> 300`` or ``"300" -> 300``.
        if isinstance(vc, bool) or not isinstance(vc, int):
            raise PineconeProviderError(
                f"count_namespace received a non-integer vector_count for "
                f"namespace '{namespace}': {vc!r}."
            )
        if vc < 0:
            raise PineconeProviderError(
                f"count_namespace received a negative vector_count for "
                f"namespace '{namespace}': {vc}."
            )
        return vc

    def list_vector_ids(
        self,
        namespace: str,
        prefix: str | None = None,
        limit: int = 100,
    ) -> list[str]:
        """List all vector IDs in a namespace with complete pagination handling.

        Fail-closed against pagination hazards:
          - Repeated pagination tokens raise PineconeProviderError.
          - A next token that yields no progress raises PineconeProviderError.
          - Duplicate vector IDs across pages raise PineconeProviderError.
          - Non-string / empty IDs raise PineconeProviderError.
          - A conservative maximum-page guard bounds runaway pagination.

        Raises PineconeProviderError if enumeration is unsupported, fails, or
        returns malformed data.
        """
        all_ids: list[str] = []
        seen_ids: set[str] = set()

        def _extract_id(item: object) -> str:
            item_id: object = None
            if isinstance(item, str):
                item_id = item
            else:
                item_id = getattr(item, "id", None)
                if item_id is None and isinstance(item, dict):
                    item_id = item.get("id")
            if not isinstance(item_id, str) or not item_id:
                raise PineconeProviderError(f"Malformed vector item in list response: {item!r}")
            return item_id

        def _record(item_id: str) -> None:
            if item_id in seen_ids:
                raise PineconeProviderError(
                    f"Duplicate vector ID enumerated across pages in namespace "
                    f"'{namespace}': {item_id!r}. Refusing to proceed with unreliable enumeration."
                )
            seen_ids.add(item_id)
            all_ids.append(item_id)

        if hasattr(self._index, "list_paginated"):
            pagination_token: str | None = None
            seen_tokens: set[str] = set()
            for _page_num in range(MAX_ENUMERATION_PAGES):
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

                ids_before = len(all_ids)
                for item in vectors:
                    _record(_extract_id(item))
                progressed = len(all_ids) > ids_before

                pagination = getattr(resp, "pagination", None)
                if pagination is None and isinstance(resp, dict):
                    pagination = resp.get("pagination")

                next_token: str | None = None
                if pagination is not None:
                    next_token = getattr(pagination, "next", None)
                    if next_token is None and isinstance(pagination, dict):
                        next_token = pagination.get("next")

                if not next_token:
                    return all_ids

                # Fail closed on a repeated token (would loop forever).
                if next_token in seen_tokens:
                    raise PineconeProviderError(
                        f"list_paginated repeated pagination token for namespace "
                        f"'{namespace}'. Refusing to loop on unreliable enumeration."
                    )
                # Fail closed if a next token is supplied but the page made no progress.
                if not progressed:
                    raise PineconeProviderError(
                        f"list_paginated supplied a next token but returned no new IDs for "
                        f"namespace '{namespace}'. Refusing to proceed with stalled enumeration."
                    )
                seen_tokens.add(next_token)
                pagination_token = next_token

            raise PineconeProviderError(
                f"list_paginated exceeded the maximum page guard "
                f"({MAX_ENUMERATION_PAGES}) for namespace '{namespace}'. "
                "Refusing to proceed with unbounded enumeration."
            )

        elif hasattr(self._index, "list"):
            try:
                pages = self._index.list(
                    namespace=namespace, prefix=prefix, timeout=self._search_timeout
                )
                page_count = 0
                for page in pages:
                    page_count += 1
                    if page_count > MAX_ENUMERATION_PAGES:
                        raise PineconeProviderError(
                            f"list iterator exceeded the maximum page guard "
                            f"({MAX_ENUMERATION_PAGES}) for namespace '{namespace}'."
                        )
                    vectors = getattr(page, "vectors", None)
                    if vectors is None and isinstance(page, dict):
                        vectors = page.get("vectors")
                    elif isinstance(page, list | tuple):
                        vectors = page
                    if vectors is None:
                        raise PineconeProviderError(f"Malformed page in list iterator: {page!r}")
                    for item in vectors:
                        _record(_extract_id(item))
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
