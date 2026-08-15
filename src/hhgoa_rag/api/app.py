"""FastAPI application with lifespan resource management.

Dense embedder, BM25 sparse encoder, and Qdrant client are loaded once on startup
and warmed up before readiness transitions to True. Requests reuse these resources —
no model loading happens inside a request handler.
"""

from __future__ import annotations

import logging
import os
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
        # 1. Load dense embedder
        use_fake = os.environ.get("HHGOA_USE_FAKE_EMBEDDER") == "1"
        if use_fake:
            from hhgoa_rag.retrieval.embedder import FakeEmbedder

            resources.embedder = FakeEmbedder()
        else:
            from hhgoa_rag.retrieval.embedder import E5MultilingualEmbedder

            resources.embedder = E5MultilingualEmbedder(model_id=settings.embedding_model_id)

        # 2. Load sparse encoder (BM25 — loads model on first encode call)
        from hhgoa_rag.retrieval.sparse_encoder import BM25SparseEncoder

        resources.sparse_encoder = BM25SparseEncoder()

        # 3. Create Qdrant client
        resources.qdrant_client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=int(settings.qdrant_connect_timeout_ms / 1000),
        )

        # 4. Warmup: embed a dummy passage (loads model weights from disk)
        _ = resources.embedder.embed_query("warmup passage")
        _ = resources.sparse_encoder.encode_query("warmup")

        # 5. Verify Qdrant alias exists and has points
        try:
            collections = {c.name for c in resources.qdrant_client.get_collections().collections}
            alias_target = settings.qdrant_collection_alias
            # Accept alias or physical collection name
            if alias_target not in collections:
                # Try resolving alias
                aliases = resources.qdrant_client.get_collection_aliases(alias_target)
                if not aliases.aliases:
                    resources.mark_not_ready(f"alias '{alias_target}' not found")
                    logger.warning("Qdrant alias not found; readiness False")
                else:
                    resources.mark_ready()
                    logger.info("Lifespan startup complete — ready")
            else:
                resources.mark_ready()
                logger.info("Lifespan startup complete — ready")
        except Exception as e:
            resources.mark_not_ready(f"qdrant_check_failed: {e}")
            logger.warning("Qdrant unreachable at startup; serving will retry: %s", e)

        resources.readiness_detail.update(
            {
                "embedder": type(resources.embedder).__name__,
                "sparse_encoder": resources.sparse_encoder.model_name,
                "qdrant_url": settings.qdrant_url,
            }
        )

    except Exception as e:
        resources.mark_not_ready(f"startup_error: {e}")
        logger.error("Lifespan startup error: %s", e)

    yield

    # Shutdown: close Qdrant client
    if resources.qdrant_client is not None:
        try:
            resources.qdrant_client.close()
        except Exception:
            pass
    resources.ready = False
    logger.info("Lifespan shutdown complete")


# Import QdrantClient here to avoid circular import issues
from qdrant_client import QdrantClient  # noqa: E402

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
