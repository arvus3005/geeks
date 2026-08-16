from typing import Literal

from fastapi import APIRouter

from hhgoa_rag.answer.extractive import extract_answer
from hhgoa_rag.config.settings import get_settings
from hhgoa_rag.guardrails.input_guards import check_input
from hhgoa_rag.guardrails.output_guards import verify_grounding
from hhgoa_rag.observability.timing import RequestTimer
from hhgoa_rag.pinecone_contract import TEXT_FIELD
from hhgoa_rag.retrieval.language_routing import get_language_filter
from hhgoa_rag.schemas.query import Citation, QueryRequest, QueryResponse, TimingsMs

router = APIRouter()


def _build_timings(stages: dict[str, float], total: float) -> TimingsMs:
    fields = TimingsMs.model_fields
    kwargs = {k: v for k, v in stages.items() if k in fields}
    kwargs["total_backend"] = total
    return TimingsMs(**kwargs)


def _error_response(
    req: QueryRequest, settings, timer, reason_code: str, detail: str
) -> QueryResponse:
    return QueryResponse(
        request_id=req.request_id,
        question=req.question,
        detected_language=None,
        answer=None,
        decision="error",
        reason_code=reason_code,
        grounded=False,
        confidence=0.0,
        citations=[],
        sources=[],
        retrieval_mode="none",
        index_manifest_id=settings.index_manifest_id,
        model_manifest_id=settings.model_manifest_id,
        timings_ms=_build_timings(timer.stages, timer.total_ms()),
        total_backend_ms=timer.total_ms(),
        error={"code": reason_code, "message": detail},
    )


@router.post("/v1/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    timer = RequestTimer()
    settings = get_settings()

    from hhgoa_rag.api.resources import get_resources

    resources = get_resources()

    # 1. Input guard
    with timer.stage("input_guard"):
        guard = check_input(req.question, settings.max_query_chars)

    if not guard.allowed:
        decision: Literal["refuse", "abstain"] = (
            "refuse"
            if guard.reason_code in ("unsafe", "credential_request", "prompt_injection")
            else "abstain"
        )
        return QueryResponse(
            request_id=req.request_id,
            question=req.question,
            detected_language=None,
            answer=None,
            decision=decision,
            reason_code=guard.reason_code,
            grounded=False,
            confidence=0.0,
            citations=[],
            sources=[],
            retrieval_mode="none",
            index_manifest_id=settings.index_manifest_id,
            model_manifest_id=settings.model_manifest_id,
            timings_ms=_build_timings(timer.stages, timer.total_ms()),
            total_backend_ms=timer.total_ms(),
        )

    # 2. Language detect
    detected_lang = "en"
    with timer.stage("language_detect"):
        try:
            from langdetect import detect

            detected_lang = detect(req.question)
        except Exception:
            detected_lang = req.language_hint or "en"
        lang_filter = get_language_filter(detected_lang, req.language_hint)

    # 3. Retrieve — Pinecone handles embedding server-side; no local vector computation
    if resources.pinecone_store is None:
        return _error_response(
            req, settings, timer, "index_unavailable", "Pinecone store not ready"
        )

    pinecone_filter: dict | None = None
    if lang_filter:
        pinecone_filter = {"language": {"$in": lang_filter}}

    try:
        with timer.stage("pinecone_retrieve"):
            hits = resources.pinecone_store.search(
                query_text=req.question,
                top_k=settings.retrieval_top_k,
                namespace=settings.pinecone_namespace,
                filter=pinecone_filter,
            )
    except Exception:
        return _error_response(
            req, settings, timer, "index_unavailable", "Vector index unavailable"
        )

    # Normalise hits to the passage dict format used downstream
    passages = [
        {
            "id": h.id,
            "score": h.score,
            "payload": h.fields,
        }
        for h in hits
    ]
    retrieval_mode = "pinecone_integrated_embed"

    # 4. Extract answer
    with timer.stage("answer_extract"):
        answer, evidence = extract_answer(passages, req.question)

    if not answer:
        return QueryResponse(
            request_id=req.request_id,
            question=req.question,
            detected_language=detected_lang,
            answer=None,
            decision="abstain",
            reason_code="weak_retrieval",
            grounded=False,
            confidence=0.0,
            citations=[],
            sources=[],
            retrieval_mode=retrieval_mode,
            index_manifest_id=settings.index_manifest_id,
            model_manifest_id=settings.model_manifest_id,
            timings_ms=_build_timings(timer.stages, timer.total_ms()),
            total_backend_ms=timer.total_ms(),
        )

    # 5. Grounding
    passage_texts = [p.get("payload", {}).get(TEXT_FIELD, "") for p in evidence]
    with timer.stage("grounding_verify"):
        grounded, confidence = verify_grounding(answer, passage_texts, settings.min_retrieval_score)

    if not grounded:
        return QueryResponse(
            request_id=req.request_id,
            question=req.question,
            detected_language=detected_lang,
            answer=None,
            decision="abstain",
            reason_code="ungrounded",
            grounded=False,
            confidence=confidence,
            citations=[],
            sources=[],
            retrieval_mode=retrieval_mode,
            index_manifest_id=settings.index_manifest_id,
            model_manifest_id=settings.model_manifest_id,
            timings_ms=_build_timings(timer.stages, timer.total_ms()),
            total_backend_ms=timer.total_ms(),
        )

    citations = [
        Citation(
            passage_id=p.get("id", ""),
            language=p.get("payload", {}).get("language", "en"),
            chunk_ordinal=p.get("payload", {}).get("chunk_ordinal", 0),
            text=p.get("payload", {}).get(TEXT_FIELD, "")[:200],
            score=p.get("score", 0.0),
        )
        for p in evidence
    ]

    total = timer.total_ms()
    return QueryResponse(
        request_id=req.request_id,
        question=req.question,
        detected_language=detected_lang,
        answer=answer,
        decision="allow",
        reason_code="answered",
        grounded=True,
        confidence=confidence,
        citations=citations,
        sources=[c.passage_id for c in citations],
        retrieval_mode=retrieval_mode,
        index_manifest_id=settings.index_manifest_id,
        model_manifest_id=settings.model_manifest_id,
        timings_ms=_build_timings(timer.stages, total),
        total_backend_ms=total,
    )
