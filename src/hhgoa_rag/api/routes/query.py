import logging
from typing import Literal

from fastapi import APIRouter

from hhgoa_rag.answer.extractive import extract_answer
from hhgoa_rag.config.settings import get_settings
from hhgoa_rag.guardrails.input_guards import check_input
from hhgoa_rag.guardrails.output_guards import verify_grounding
from hhgoa_rag.observability.timing import RequestTimer
from hhgoa_rag.pinecone_contract import TEXT_FIELD
from hhgoa_rag.retrieval.language_routing import detect_language, get_language_filter
from hhgoa_rag.retrieval.local_embedder import embed_query
from hhgoa_rag.schemas.query import Citation, QueryRequest, QueryResponse, TimingsMs

logger = logging.getLogger(__name__)

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
    with timer.stage("language_detect"):
        detected_lang = detect_language(req.question)
        lang_filter = get_language_filter(detected_lang, req.language_hint)

    # 3. Retrieve — self-hosted BM25+HNSW hybrid, sharded across the local
    # index (see sharded_local_hybrid_store's module docstring for why
    # sharded rather than one merged index). Not ready is only reported if
    # startup found zero shards at all — individual shard load errors
    # surface as the generic except below instead.
    if not resources.ready:
        return _error_response(
            req, settings, timer, "index_unavailable", "Local hybrid index not ready"
        )

    from hhgoa_rag.retrieval.sharded_local_hybrid_store import search as local_hybrid_search

    try:
        with timer.stage("query_embed"):
            query_vector = embed_query(req.question)
        with timer.stage("local_hybrid_retrieve"):
            hits = local_hybrid_search(
                query_text=req.question,
                query_vector=query_vector,
                languages=lang_filter,
                top_k=settings.retrieval_top_k,
            )
    except Exception as exc:
        logger.error("Retrieval failed for request_id=%s: %s", req.request_id, exc)
        return _error_response(
            req, settings, timer, "index_unavailable", f"Local hybrid index unavailable: {exc}"
        )

    # Normalise hits to the passage dict format used downstream
    passages = [
        {
            "id": h["id"],
            "score": h["score"],
            "payload": h["fields"],
        }
        for h in hits
    ]
    retrieval_mode = "local_hybrid_sharded_bm25_hnsw_rrf"

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
