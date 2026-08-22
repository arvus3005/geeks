"""Thin wrapper exposing this project's real retrieval path (query embed +
local hybrid BM25/HNSW search) as the search()/warmup() interface
docs/benchmark.py expects -- so that file runs as-is against this project's
actual production code, not a separate reimplementation.

Uses the same modules the live API route uses (hhgoa_rag.retrieval.local_embedder,
hhgoa_rag.retrieval.sharded_local_hybrid_store) and the same language-routing
default ("hi" -- covers the shared English pool, see language_routing.py's
own docstring) that a text query with no language hint gets in query.py.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class SearchResponse:
    total_ms: float
    embed_ms: float
    search_ms: float
    hits: list[dict]


def warmup() -> None:
    from hhgoa_rag.retrieval import local_embedder
    from hhgoa_rag.retrieval.sharded_local_hybrid_store import warm_all_shards

    local_embedder._lazy_load()
    warm_all_shards()


def search(query: str, top_k: int = 5) -> SearchResponse:
    from hhgoa_rag.retrieval.local_embedder import embed_query
    from hhgoa_rag.retrieval.sharded_local_hybrid_store import search as hybrid_search

    t0 = time.monotonic()
    t_embed_start = time.monotonic()
    query_vector = embed_query(query)
    embed_ms = (time.monotonic() - t_embed_start) * 1000.0

    t_search_start = time.monotonic()
    hits = hybrid_search(query_text=query, query_vector=query_vector, languages=["hi"], top_k=top_k)
    search_ms = (time.monotonic() - t_search_start) * 1000.0

    total_ms = (time.monotonic() - t0) * 1000.0
    return SearchResponse(total_ms=total_ms, embed_ms=embed_ms, search_ms=search_ms, hits=hits)
