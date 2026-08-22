"""FastAPI application with local self-hosted hybrid retrieval lifespan
resource management. No managed vector DB anywhere in this path — the only
retrieval backend is the self-hosted BM25+HNSW sharded local index
(src/hhgoa_rag/retrieval/sharded_local_hybrid_store.py), per the
2026-08-22 team decision (see README's indexing-status section). The old
Pinecone-based serving path and the Pinecone-specific ingestion pipeline
it depended on were removed from the repo entirely the same day, not left
around unused.

The query embedder (local_embedder.py, int8 ONNX), the cross-encoder
relevance reranker (answer/reranker.py, int8 ONNX) that gates whether
extract_answer answers or declines, and every live-serving shard are all
warmed here at startup — loaded once, never per-request, per CLAUDE.md.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from hhgoa_rag.api.resources import get_resources
from hhgoa_rag.config.settings import get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    resources = get_resources()
    settings = get_settings()

    try:
        from hhgoa_rag.answer import reranker
        from hhgoa_rag.retrieval import local_embedder
        from hhgoa_rag.retrieval.sharded_local_hybrid_store import _discover_shards, warm_all_shards

        local_embedder._lazy_load()
        reranker._lazy_load()

        shards_by_lang = _discover_shards()
        total_shards = sum(len(v) for v in shards_by_lang.values())
        if total_shards == 0:
            resources.mark_not_ready(f"no shards found under {settings.local_index_root}")
            logger.warning("No local index shards found; serving will be degraded")
        else:
            # Eagerly open + search every shard now (measured ~200ms/shard
            # cold) so no live request pays that cost -- see
            # sharded_local_hybrid_store.warm_all_shards's docstring for why
            # lazy-on-first-request was measured to blow the 200ms budget by
            # 40-75x on a real query.
            warm_all_shards()
            resources.mark_ready()
            logger.info(
                "Lifespan startup complete — local hybrid index ready and warmed, %d shards across %s",
                total_shards,
                sorted(shards_by_lang.keys()),
            )

        resources.readiness_detail.update(
            {
                "retrieval_backend": "local_hybrid_sharded",
                "languages": sorted(shards_by_lang.keys()),
                "n_shards": total_shards,
            }
        )

    except Exception as e:
        resources.mark_not_ready(f"startup_error: {e}")
        logger.error("Lifespan startup error: %s", e)

    yield

    resources.ready = False
    logger.info("Lifespan shutdown complete")


from pathlib import Path
from fastapi.staticfiles import StaticFiles

from .routes import health, query, system, voice  # noqa: E402

app = FastAPI(title="HH Goa RAG API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(query.router)
app.include_router(system.router)
app.include_router(voice.router)

# Mount static web frontend
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
if not STATIC_DIR.exists():
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
