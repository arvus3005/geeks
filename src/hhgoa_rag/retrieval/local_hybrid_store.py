"""Self-hosted hybrid retrieval — BM25 (bm25s) + HNSW (usearch), fused via
Reciprocal Rank Fusion. In-process, no network hop, no hosting quota — an
alternative to PineconeStore built from the same corpus (see
scripts/build_local_hybrid_index.py for how the index artifacts are built).

Why hybrid, not dense-only
--------------------------
Dense embedding search (what Pinecone/local_embedder alone provide) is good
at semantic/paraphrase matches but can miss exact keyword hits — names,
numbers, rare terms — that a lexical method catches reliably. Retrieve
from both, then fuse.

Why RRF, not a weighted score blend
------------------------------------
BM25 scores and cosine similarity are not on comparable scales — blending
them by weighted sum requires calibration that hasn't been done here. RRF
only uses each method's *rank* (1st, 2nd, 3rd place, ...), which is
scale-free: `score = sum(1 / (k + rank))` across both lists. k=60 is the
standard default from the original RRF paper.

Loaded once at process startup, per CLAUDE.md.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

INDEX_DIR = Path("artifacts/local_index")
RRF_K = 60

_bm25 = None
_hnsw = None
_passages: list[dict] | None = None  # index = usearch key = bm25 doc index


def _lazy_load() -> None:
    global _bm25, _hnsw, _passages
    if _passages is not None:
        return

    import bm25s
    from usearch.index import Index

    from hhgoa_rag.retrieval.local_embedder import EMBED_DIM

    manifest_path = INDEX_DIR / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(
            f"No local hybrid index at {INDEX_DIR} — run "
            "`uv run python -m scripts.build_local_hybrid_index` first."
        )

    logger.info("Loading local hybrid index from %s", INDEX_DIR)
    _bm25 = bm25s.BM25.load(str(INDEX_DIR / "bm25"))

    _hnsw = Index(ndim=EMBED_DIM, metric="cos", dtype="f32")
    _hnsw.load(str(INDEX_DIR / "hnsw.usearch"))

    _passages = []
    with (INDEX_DIR / "passages.jsonl").open() as f:
        for line in f:
            _passages.append(json.loads(line))

    logger.info("Local hybrid index ready: %d passages", len(_passages))


def _bm25_search(query_text: str, top_k: int) -> dict[int, int]:
    """Returns {passage_key: rank} (rank is 1-indexed, best first)."""
    import hhgoa_rag.retrieval.local_embedder as le

    assert _bm25 is not None
    query_tokens = [le._sp.encode(query_text, out_type=str)]
    results, _scores = _bm25.retrieve(
        query_tokens, k=min(top_k, len(_passages or [])), show_progress=False
    )
    return {int(key): rank + 1 for rank, key in enumerate(results[0])}


def _hnsw_search(query_vector: list[float], top_k: int) -> dict[int, int]:
    import numpy as np

    assert _hnsw is not None
    matches = _hnsw.search(np.array(query_vector, dtype=np.float32), top_k)
    return {int(key): rank + 1 for rank, key in enumerate(matches.keys)}


def search(query_text: str, query_vector: list[float], top_k: int = 20) -> list[dict]:
    """Hybrid search fusing BM25 lexical rank and HNSW dense rank via RRF.

    Returns a list of dicts shaped like PineconeStore.search_by_vector's
    SearchHit for drop-in compatibility: {id, score, fields}.
    """
    _lazy_load()
    assert _passages is not None

    candidate_k = max(top_k * 3, 50)  # widen the candidate pool before fusing
    bm25_ranks = _bm25_search(query_text, candidate_k)
    hnsw_ranks = _hnsw_search(query_vector, candidate_k)

    all_keys = set(bm25_ranks) | set(hnsw_ranks)
    rrf_scores = {
        key: 1.0 / (RRF_K + bm25_ranks.get(key, candidate_k + RRF_K))
        + 1.0 / (RRF_K + hnsw_ranks.get(key, candidate_k + RRF_K))
        for key in all_keys
    }

    ranked = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

    hits = []
    for key, score in ranked:
        p = _passages[key]
        hits.append({"id": p["id"], "score": score, "fields": p["metadata"]})
    return hits
