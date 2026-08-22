from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class SystemResponse(BaseModel):
    corpus_mode: str
    index_manifest_id: str
    model_manifest_id: str
    supported_languages: list[str]
    indexed_languages: list[str]
    warning: str | None


@router.get("/v1/system", response_model=SystemResponse)
async def system_info():
    from hhgoa_rag.config.settings import get_settings
    from hhgoa_rag.retrieval.language_routing import INDEXED_LANGUAGES, SUPPORTED_LANGUAGES

    s = get_settings()
    warning = None
    if s.corpus_mode == "smoke":
        warning = "WARNING: SMOKE corpus only. Full-dataset and sub-200ms production claims are NOT yet verified."
    elif len(INDEXED_LANGUAGES) < len(SUPPORTED_LANGUAGES):
        missing = sorted(SUPPORTED_LANGUAGES - INDEXED_LANGUAGES)
        warning = (
            f"Partial corpus: {len(INDEXED_LANGUAGES)} of {len(SUPPORTED_LANGUAGES)} "
            f"MSMARCO-XI language configs are indexed. Not yet built: {missing}. "
            "Queries in an un-indexed language fall back to English-only retrieval."
        )
    return SystemResponse(
        corpus_mode=s.corpus_mode,
        index_manifest_id=s.index_manifest_id,
        model_manifest_id=s.model_manifest_id,
        supported_languages=sorted(SUPPORTED_LANGUAGES),
        indexed_languages=sorted(INDEXED_LANGUAGES),
        warning=warning,
    )
