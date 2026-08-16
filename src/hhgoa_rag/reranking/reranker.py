"""Pinecone reranking boundary.

Supports:
- retrieval-only mode (bypass reranking)
- search-plus-rerank mode via Pinecone rerank API
- configurable model, candidate K, and final top-N
- fail-closed behaviour: raises PineconeRerankError on failure
- usage and timing metadata when returned by Pinecone
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..pinecone_contract import TEXT_FIELD

DEFAULT_RERANK_MODEL = "bge-reranker-v2-m3"
RANK_FIELDS = [TEXT_FIELD]


class PineconeRerankError(Exception):
    """Raised when the Pinecone rerank call fails."""

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.cause = cause


@dataclass
class RankedResult:
    """A single result after retrieval (and optionally reranking)."""

    record_id: str
    chunk_text: str
    metadata: dict[str, Any]
    retrieval_score: float | None
    rerank_score: float | None
    score_type: str  # "retrieval" | "rerank"
    final_rank: int


@dataclass
class RerankUsage:
    rerank_units: int | None = None
    search_units: int | None = None
    elapsed_ms: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _validate_config(candidate_k: int, top_n: int) -> None:
    if candidate_k <= 0:
        raise ValueError(f"candidate_k must be positive, got {candidate_k}")
    if top_n <= 0:
        raise ValueError(f"top_n must be positive, got {top_n}")
    if top_n > candidate_k:
        raise ValueError(f"top_n ({top_n}) cannot exceed candidate_k ({candidate_k})")


def _ensure_chunk_text(hits: list[dict]) -> None:
    for hit in hits:
        if TEXT_FIELD not in hit:
            raise ValueError(f"{TEXT_FIELD} missing from hit: {hit.get('id', '<no-id>')}")


class PineconeReranker:
    """Reranking boundary wrapping the Pinecone rerank API."""

    def __init__(
        self,
        pc: Any,  # pinecone.Pinecone client
        model: str = DEFAULT_RERANK_MODEL,
        candidate_k: int = 8,
        top_n: int = 3,
        rerank_timeout: float = 10.0,
        rank_fields: list[str] | None = None,
    ) -> None:
        _validate_config(candidate_k, top_n)
        self._pc = pc
        self.model = model
        self.candidate_k = candidate_k
        self.top_n = top_n
        self._rerank_timeout = rerank_timeout
        self.rank_fields = rank_fields or RANK_FIELDS
        if TEXT_FIELD not in self.rank_fields:
            raise ValueError(f"rank_fields must include '{TEXT_FIELD}'")

    def rerank(
        self,
        query: str,
        hits: list[dict[str, Any]],
        top_n: int | None = None,
    ) -> tuple[list[RankedResult], RerankUsage]:
        """Rerank hits and return top-N ranked results with usage info.

        hits: list of dicts with at least 'id', TEXT_FIELD, and optional metadata.
        Raises PineconeRerankError on failure (fail-closed).
        """
        effective_top_n = top_n if top_n is not None else self.top_n
        if effective_top_n <= 0:
            raise ValueError(f"top_n must be positive, got {effective_top_n}")
        _ensure_chunk_text(hits)

        t0 = time.monotonic()
        try:
            resp = self._pc.inference.rerank(
                model=self.model,
                query=query,
                documents=hits,
                rank_fields=self.rank_fields,
                top_n=effective_top_n,
                return_documents=True,
            )
        except Exception as e:
            raise PineconeRerankError(f"Pinecone rerank failed: {e}", cause=e) from e

        elapsed_ms = (time.monotonic() - t0) * 1000

        results: list[RankedResult] = []
        raw_results = getattr(resp, "results", None) or getattr(resp, "data", None) or []
        for rank, item in enumerate(raw_results):
            doc = getattr(item, "document", None) or {}
            if isinstance(doc, dict):
                doc_dict = doc
            else:
                doc_dict = dict(doc) if doc else {}

            chunk_text = doc_dict.get(TEXT_FIELD, "")
            record_id = doc_dict.get("id", str(rank))
            rerank_score = float(getattr(item, "score", 0.0))
            retrieval_score = doc_dict.get("_retrieval_score")

            metadata = {
                k: v for k, v in doc_dict.items() if k not in ("id", TEXT_FIELD, "_retrieval_score")
            }

            results.append(
                RankedResult(
                    record_id=record_id,
                    chunk_text=chunk_text,
                    metadata=metadata,
                    retrieval_score=float(retrieval_score) if retrieval_score is not None else None,
                    rerank_score=rerank_score,
                    score_type="rerank",
                    final_rank=rank,
                )
            )

        usage_raw = getattr(resp, "usage", None)
        usage = RerankUsage(elapsed_ms=elapsed_ms)
        if usage_raw is not None:
            usage.rerank_units = getattr(usage_raw, "rerank_units", None)
            usage.search_units = getattr(usage_raw, "search_units", None)

        return results, usage


class RetrievalOnlyPassthrough:
    """Returns retrieval hits in order without calling the rerank API."""

    def __init__(self, top_n: int = 3) -> None:
        if top_n <= 0:
            raise ValueError(f"top_n must be positive, got {top_n}")
        self.top_n = top_n

    def rerank(
        self,
        query: str,
        hits: list[dict[str, Any]],
        top_n: int | None = None,
    ) -> tuple[list[RankedResult], RerankUsage]:
        effective_top_n = top_n if top_n is not None else self.top_n
        selected = hits[:effective_top_n]
        results = [
            RankedResult(
                record_id=h.get("id", str(i)),
                chunk_text=h.get(TEXT_FIELD, ""),
                metadata={k: v for k, v in h.items() if k not in ("id", TEXT_FIELD)},
                retrieval_score=h.get("_retrieval_score"),
                rerank_score=None,
                score_type="retrieval",
                final_rank=i,
            )
            for i, h in enumerate(selected)
        ]
        return results, RerankUsage()
