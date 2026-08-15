from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class LiveResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str
    corpus_mode: str
    index_manifest_id: str
    details: dict


@router.get("/health/live", response_model=LiveResponse)
async def liveness():
    return LiveResponse(status="ok")


@router.get("/health/ready", response_model=ReadyResponse)
async def readiness():
    from hhgoa_rag.api.resources import get_resources
    from hhgoa_rag.config.settings import get_settings

    s = get_settings()
    resources = get_resources()

    details = dict(resources.readiness_detail)
    if not resources.ready:
        details.setdefault("qdrant", "not_verified")
        details.setdefault("embedder", "not_loaded")

    return ReadyResponse(
        status="ready" if resources.ready else "initializing",
        corpus_mode=s.corpus_mode,
        index_manifest_id=s.index_manifest_id,
        details=details,
    )
