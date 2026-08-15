from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class SystemResponse(BaseModel):
    corpus_mode: str
    index_manifest_id: str
    model_manifest_id: str
    supported_languages: list[str]
    warning: str | None


@router.get("/v1/system", response_model=SystemResponse)
async def system_info():
    from hhgoa_rag.config.settings import get_settings

    s = get_settings()
    warning = None
    if s.corpus_mode == "smoke":
        warning = "WARNING: SMOKE corpus only. Full-dataset and sub-200ms production claims are NOT yet verified."
    return SystemResponse(
        corpus_mode=s.corpus_mode,
        index_manifest_id=s.index_manifest_id,
        model_manifest_id=s.model_manifest_id,
        supported_languages=[
            "as",
            "bn",
            "gu",
            "hi",
            "kn",
            "ml",
            "mr",
            "ne",
            "or",
            "pa",
            "sa",
            "ta",
            "te",
            "ur",
            "en",
        ],
        warning=warning,
    )
