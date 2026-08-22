"""Self-hosted hybrid retrieval over the FULL local index (all languages,
all segments) — BM25 (bm25s) + HNSW (usearch), fused via Reciprocal Rank
Fusion, exactly like local_hybrid_store.py but sharded across many
independent per-segment indexes instead of one merged one.

Why sharded instead of merged (read this before "fixing" it into one index)
-----------------------------------------------------------------------------
Measured (2026-08-22, this machine, 25.8GB RAM): merging the full corpus
into ONE BM25 index needs ~368GB RAM to build (bm25s's score-matrix
construction is proportional to total corpus tokens, ~5.15B for the full
corpus) and ONE HNSW index needs ~181GB RAM for the vectors+graph alone
(measured 3.349GB per million 384-dim f32 vectors). Both are 7-14x more
than this machine has, and no clever token/vector encoding on the *input*
side changes that — bm25s's own internal build arrays are what's
proportional to total corpus size, not just the input format.

The fix that actually fits: don't merge. Every finished language already
lives as many small per-segment shards under artifacts/full_local_index/
(this project already segments its indexing output specifically to bound
peak build RAM — see scripts/build_full_local_index.py's SEGMENTING
section). At query time:
  1. Route to only the shards for the detected/requested language(s) (see
     language_routing.get_language_filter) — never all 112 shards.
  2. Live serving further caps to MAX_SEGMENTS_PER_LANGUAGE segments per
     language (see that constant's own comment for the real memory
     numbers behind the chosen value) and loads those FULLY into RAM
     (usearch's `.load()`, bm25s's `mmap=False`) rather than mmap/view.
     An mmap-based version of this module was tried first and measured
     real cold-page latency spikes (P100=286ms across a 120-query,
     6-language benchmark) even after startup "warming", because
     different real queries traverse different regions of the HNSW graph
     than a single dummy warmup search touches — mmap only avoids RAM
     usage if you accept that variance. Since the live-serving set is
     already capped small enough to fit fully resident (~6.4GB across 6
     shards on a 25.8GB machine), fully loading it removes the variance
     entirely instead of half-solving the memory problem and still eating
     mmap's unpredictable cold-page cost.
  3. Search every targeted shard (dense + sparse), pool all candidates by
     their real RRF score (comparable across shards — same formula,
     doesn't depend on shard-local corpus statistics the way a raw BM25
     score or IDF would), take the global top-k.

Real measured latency per shard (2026-08-22, this machine), WARM:
  - HNSW (usearch, threads=1 per call): ~0.7ms/shard
  - BM25 (bm25s):                        ~0.9ms/shard
COLD (a shard's first-ever search after process start, under the earlier
mmap/view design) was a different story: measured ~169ms for the first
HNSW search alone on a real production shard (real disk I/O + page faults
during graph traversal). An even earlier version of this module loaded
shards lazily on first request, which meant a live user's first query into
a not-yet-touched shard group could take 7-15+ SECONDS (48 shards x
~200ms cold each) -- a real, measured failure of the <200ms target, not a
theoretical one. Fixed in two steps: warm_all_shards() opens+searches
every shard during FastAPI lifespan startup (see api/app.py) before the
process reports ready, per CLAUDE.md's "load once at startup, never
per-request" rule; and switching from mmap/view to full .load() (above)
removed the remaining cold-page variance mmap alone couldn't fix.
IMPORTANT: always pass threads=1 to usearch's .search() for single-query
serving. Measured: default (multi-threaded) search cost ~30ms/call on a
500k-vector shard vs ~0.7-8ms with threads=1 explicitly -- the thread-pool
spin-up overhead dominates for a single query with nothing to parallelize.

EMBEDDING-CONSISTENCY CHECK — RESOLVED 2026-08-22: every segment's dense
vectors were built with embed_backend "mps_fp16_transformers" (see each
segment's manifest.json), NOT the int8 ONNX model
src/hhgoa_rag/retrieval/local_embedder.py uses to embed queries here — this
was flagged as an unresolved risk since 2026-08-21 (same bug *class* as
the earlier XLM-RoBERTa tokenizer offset bug: silent, no error). Checked
per POST_INDEXING_STEPS.md Step 3 via self-retrieval ground truth: sampled
24 real passages across all 6 indexed languages, queried with a natural
excerpt of each passage's own text, and checked whether the source passage
came back. Result: 15/24 at rank 1, 22/24 within top-5, only 2 genuine
misses. Passing, not perfect — real embeddings on real (sometimes
ambiguous/short) text are not expected to self-retrieve 100% even with a
perfectly matched embedder, but this rules out the "scrambled/incompatible
embedding space" failure mode the earlier tokenizer bug caused. Considered
verified-working, not "production ready" in some stronger unverified
sense — see CLAUDE.md's Honesty section.

Loaded/opened once per shard and cached for the process lifetime, per
CLAUDE.md (embedders/index clients must load once at startup, never
per-request) — shards are opened lazily on first use rather than eagerly
at startup so process boot doesn't have to touch all 112 files up front,
but each is cached after that.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SHARD_ROOT = Path("artifacts/full_local_index")
RRF_K = 60
_SEGMENT_NAME_RE = re.compile(r"^([a-z]{2})_(train|validation)_segment_(\d+)$")

# How many segments per language actually get loaded for LIVE serving.
# Measured (2026-08-22, this machine): one shard's serve-time footprint is
# ~1.07GB (842MB HNSW view + ~230MB BM25 compact structure). This machine
# has 25.8GB RAM total. hi alone has 32 segments (~34GB) -- more than
# total RAM -- so the OS page cache can never hold hi's whole working set
# at once; every query thrashes pages in and out regardless of "warming",
# and cold HNSW search on a real segment measured ~169ms (real disk I/O +
# random-access graph traversal, not a fixable software bug). Capping to a
# small, comfortably-cacheable number of segments per language keeps the
# warmed set small enough to actually STAY warm (see warm_all_shards),
# which is what makes the <200ms target real rather than aspirational.
# The full, uncapped 112-segment/54.25M-passage index still exists on disk
# and is the verified offline artifact -- this cap is a live-serving
# concession to this machine's RAM, not a change to what was built. See
# README's indexing-status section and CLAUDE.md's Honesty section.
#
# 2 segments/language (12 shards, ~12.8GB) was tried and measured to still
# thrash: `vm_stat` showed only ~9.3GB of truly reclaimable memory on this
# dev machine once Chrome/the IDE's own resident (non-reclaimable "active")
# memory is excluded -- so real available headroom for page cache is well
# under the naive 25.8GB total. 1 segment/language (~6.4GB) fits inside
# that real budget with margin.
MAX_SEGMENTS_PER_LANGUAGE = 1


import heapq

@dataclass
class _ShardHandles:
    name: str
    dir: Path
    hnsw: object
    bm25: object
    offsets: np.ndarray
    file_handle: object | None = None


_shard_cache: dict[str, _ShardHandles] = {}
_shards_by_lang: dict[str, list[str]] | None = None
_lock = threading.Lock()


def _discover_shards() -> dict[str, list[str]]:
    global _shards_by_lang
    if _shards_by_lang is not None:
        return _shards_by_lang

    by_lang: dict[str, list[str]] = {}
    for p in sorted(SHARD_ROOT.iterdir()):
        if not p.is_dir():
            continue
        m = _SEGMENT_NAME_RE.match(p.name)
        if not m:
            continue
        lang = m.group(1)
        by_lang.setdefault(lang, []).append(p.name)

    # Prefer train segments over validation (more representative of the
    # real corpus), then cap deterministically -- see MAX_SEGMENTS_PER_LANGUAGE.
    for lang, names in by_lang.items():
        names.sort(key=lambda n: (0 if "_train_" in n else 1, n))
        by_lang[lang] = names[:MAX_SEGMENTS_PER_LANGUAGE]

    _shards_by_lang = by_lang
    total = sum(len(v) for v in by_lang.values())
    logger.info(
        "Discovered shards, capped to %d/language for live serving: %d total across %s",
        MAX_SEGMENTS_PER_LANGUAGE,
        total,
        {k: len(v) for k, v in by_lang.items()},
    )
    return by_lang


def warm_all_shards(max_workers: int = 8) -> None:
    """Eagerly open + search every shard once, so the first REAL user query
    doesn't pay the cold cost (measured ~200ms/shard -- see module
    docstring). Call this once during FastAPI lifespan startup, not
    per-request. Parallelized across threads since usearch/bm25s's
    underlying calls release the GIL for their real work (confirmed:
    measured multi-shard parallel search is faster than serial, not just
    equal, in earlier fan-out timing tests)."""
    from concurrent.futures import ThreadPoolExecutor

    by_lang = _discover_shards()
    all_shard_names = [name for names in by_lang.values() for name in names]
    if not all_shard_names:
        return

    from hhgoa_rag.retrieval.local_embedder import EMBED_DIM

    dummy_vec = np.zeros(EMBED_DIM, dtype=np.float32)
    dummy_vec[0] = 1.0  # unit-ish vector, avoids any all-zero edge case in cosine search
    dummy_tokens = ["warmup"]

    def _warm_one(name: str) -> None:
        handles = _get_shard(name)
        try:
            # Use the SAME candidate_k as real queries (50) -- a k=1 search
            # was measured to under-warm the shard (usearch's graph
            # traversal explores fewer nodes/pages for a small k than the
            # real query later needs, so the real query still hit mostly-cold
            # pages despite "warmup" having run).
            handles.hnsw.search(dummy_vec, 50, threads=1)
        except Exception as exc:  # noqa: BLE001 -- warmup best-effort, real errors surface on real queries
            print(f"WARMUP hnsw failed for {name}: {exc}", flush=True)
        try:
            handles.bm25.retrieve([dummy_tokens], k=50, show_progress=False)
        except Exception as exc:  # noqa: BLE001
            print(f"WARMUP bm25 failed for {name}: {exc}", flush=True)

    import time

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        list(ex.map(_warm_one, all_shard_names))
    # A second pass: measured (2026-08-22) that a single k=50 warmup search
    # still left a handful of real queries hitting partially-cold pages
    # (P95/P100 of a 48-query benchmark right after a fresh restart showed
    # 207-306ms outliers despite every shard having been "warmed" once).
    # Re-running the same searches is cheap once genuinely warm (~sub-ms
    # each) and catches whatever the first pass's traversal path missed.
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        list(ex.map(_warm_one, all_shard_names))
    elapsed = time.monotonic() - t0
    print(f"WARMUP: {len(all_shard_names)} shards, 2 passes, in {elapsed:.2f}s ({max_workers} workers)", flush=True)
    logger.info("Warmed %d shards (2 passes) in %.1fs (%d workers)", len(all_shard_names), elapsed, max_workers)


def _get_shard(name: str) -> _ShardHandles:
    handles = _shard_cache.get(name)
    if handles is not None:
        return handles

    with _lock:
        handles = _shard_cache.get(name)
        if handles is not None:
            return handles

        from hhgoa_rag.retrieval.local_embedder import EMBED_DIM

        shard_dir = SHARD_ROOT / name

        import bm25s
        from usearch.index import Index

        # Fully load (not view/mmap) -- a real 120-query benchmark across
        # all 6 languages measured P100=286ms with view() + a single warmup
        # search per shard, because different real queries traverse
        # different regions of the HNSW graph than one dummy warmup query
        # touches, so some fraction of diverse traffic always hits a cold
        # page under mmap. MAX_SEGMENTS_PER_LANGUAGE was deliberately capped
        # to make the live-serving set small enough (~6.4GB across 6
        # shards) to fully fit in RAM -- so actually load it fully instead
        # of half-solving the memory problem AND still paying mmap's
        # variable cold-page cost.
        hnsw = Index(ndim=EMBED_DIM, metric="cos", dtype="f32")
        hnsw.load(str(shard_dir / "hnsw.usearch"))

        bm25 = bm25s.BM25.load(str(shard_dir / "bm25"), mmap=False, load_corpus=False)

        offsets = np.load(shard_dir / "passages_offsets.npy", mmap_mode="r")

        handles = _ShardHandles(name=name, dir=shard_dir, hnsw=hnsw, bm25=bm25, offsets=offsets)
        _shard_cache[name] = handles
        return handles


def _fetch_passage(handles: _ShardHandles, key: int) -> dict:
    offset = int(handles.offsets[key])
    f = handles.file_handle
    if f is None or getattr(f, "closed", True):
        handles.file_handle = open(handles.dir / "passages.jsonl", "rb")
        f = handles.file_handle
    f.seek(offset)
    line = f.readline()
    return json.loads(line)


def _bm25_search(handles: _ShardHandles, query_tokens: list[str], top_k: int) -> dict[int, int]:
    if not query_tokens:
        return {}
    results, _scores = handles.bm25.retrieve(
        [query_tokens], k=min(top_k, len(handles.offsets)), show_progress=False
    )
    return {int(key): rank + 1 for rank, key in enumerate(results[0])}


def _hnsw_search(handles: _ShardHandles, query_vector: np.ndarray, top_k: int) -> dict[int, int]:
    matches = handles.hnsw.search(query_vector, min(top_k, len(handles.offsets)), threads=1)
    return {int(key): rank + 1 for rank, key in enumerate(matches.keys)}


def search(
    query_text: str,
    query_vector: list[float],
    languages: list[str],
    top_k: int = 20,
) -> list[dict]:
    """Hybrid search fanned out across every shard for the given languages,
    fused via Reciprocal Rank Fusion across BOTH modalities AND shards.

    Returns hits shaped like PineconeStore.search_by_vector's SearchHit for
    drop-in compatibility: {id, score, fields}, where fields includes both
    "text" and "chunk_text" (the latter for TEXT_FIELD compatibility with
    the existing extract/ground/cite pipeline).
    """
    import hhgoa_rag.retrieval.local_embedder as le

    by_lang = _discover_shards()
    shard_names: list[str] = []
    for lang in languages:
        shard_names.extend(by_lang.get(lang, []))
    if not shard_names:
        logger.warning("No shards found for languages=%s", languages)
        return []

    query_tokens = le._sp.encode(query_text, out_type=str)
    query_vec_np = np.array(query_vector, dtype=np.float32)
    candidate_k = max(top_k * 3, 30)

    # (shard_name, local_key) -> summed RRF score. RRF's score formula
    # (1/(k+rank)) only depends on each list's own rank, not corpus-wide
    # statistics, so scores from different shards/modalities are directly
    # summable -- this is the same math as fusing two lists from one index,
    # just with more lists.
    rrf_scores: dict[tuple[str, int], float] = {}

    for shard_name in shard_names:
        handles = _get_shard(shard_name)
        bm25_ranks = _bm25_search(handles, query_tokens, candidate_k)
        hnsw_ranks = _hnsw_search(handles, query_vec_np, candidate_k)
        for key in set(bm25_ranks) | set(hnsw_ranks):
            score = 1.0 / (RRF_K + bm25_ranks.get(key, candidate_k + RRF_K)) + 1.0 / (
                RRF_K + hnsw_ranks.get(key, candidate_k + RRF_K)
            )
            rrf_scores[(shard_name, key)] = rrf_scores.get((shard_name, key), 0.0) + score

    ranked = heapq.nlargest(top_k, rrf_scores.items(), key=lambda kv: kv[1])

    hits = []
    for (shard_name, key), score in ranked:
        handles = _get_shard(shard_name)
        passage = _fetch_passage(handles, key)
        text = passage["text"]
        metadata = dict(passage.get("metadata", {}))
        metadata["chunk_text"] = text
        metadata["text"] = text
        hits.append({"id": f"{shard_name}:{key}", "score": score, "fields": metadata})
    return hits
