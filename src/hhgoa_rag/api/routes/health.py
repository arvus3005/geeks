from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

_ready = False


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
    from hhgoa_rag.config.settings import get_settings

    s = get_settings()
    return ReadyResponse(
        status="ready" if _ready else "initializing",
        corpus_mode=s.corpus_mode,
        index_manifest_id=s.index_manifest_id,
        details={"qdrant": "unknown", "embedder": "unknown"},
    )


def set_ready(value: bool):
    global _ready
    _ready = value
