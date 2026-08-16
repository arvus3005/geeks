"""FastAPI application with Pinecone lifespan resource management.

PineconeStore is created once on startup using the integrated multilingual
embedding model. No local model is loaded; Pinecone handles embedding
server-side. The store is reused across all requests.
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
        if not settings.pinecone_api_key:
            resources.mark_not_ready("PINECONE_API_KEY not set")
            logger.warning("PINECONE_API_KEY not set; serving will be degraded")
            yield
            return

        from pinecone import Pinecone

        from hhgoa_rag.pinecone_lifecycle import validate_index
        from hhgoa_rag.pinecone_store import PineconeStore

        pc = Pinecone(api_key=settings.pinecone_api_key)
        index = pc.Index(settings.pinecone_index)
        store = PineconeStore(
            index=index,
            embed_model=settings.pinecone_embed_model,
            upsert_timeout=settings.pinecone_upsert_timeout_ms / 1000,
            search_timeout=settings.pinecone_search_timeout_ms / 1000,
        )

        # Validate index config against all canonical contract fields
        errors = validate_index(
            pc,
            settings.pinecone_index,
        )
        if errors:
            resources.mark_not_ready(f"index_config_errors: {errors}")
            logger.warning("Pinecone index config errors: %s", errors)
        else:
            # Probe — lightweight health check
            try:
                store.health()
                resources.pinecone_store = store
                resources.mark_ready()
                logger.info("Lifespan startup complete — Pinecone ready")
            except Exception as e:
                resources.mark_not_ready(f"pinecone_probe_failed: {e}")
                logger.warning("Pinecone probe failed: %s", e)

        resources.readiness_detail.update(
            {
                "pinecone_index": settings.pinecone_index,
                "pinecone_embed_model": settings.pinecone_embed_model,
                "pinecone_namespace": settings.pinecone_namespace,
            }
        )

    except Exception as e:
        resources.mark_not_ready(f"startup_error: {e}")
        logger.error("Lifespan startup error: %s", e)

    yield

    resources.pinecone_store = None
    resources.ready = False
    logger.info("Lifespan shutdown complete")


from .routes import health, query, system  # noqa: E402

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
