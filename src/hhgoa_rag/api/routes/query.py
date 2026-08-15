from fastapi import APIRouter
from hhgoa_rag.schemas.query import QueryRequest, QueryResponse, TimingsMs, Citation
from hhgoa_rag.observability.timing import RequestTimer
from hhgoa_rag.guardrails.input_guards import check_input
from hhgoa_rag.guardrails.output_guards import verify_grounding
from hhgoa_rag.retrieval.language_routing import get_language_filter
from hhgoa_rag.answer.extractive import extract_answer
from hhgoa_rag.config.settings import get_settings

router = APIRouter()


def _get_embedder():
    # In tests HHGOA_USE_FAKE_EMBEDDER=1 env var triggers fake
    import os

    if os.environ.get("HHGOA_USE_FAKE_EMBEDDER") == "1":
        from hhgoa_rag.retrieval.embedder import FakeEmbedder

        return FakeEmbedder()
    from hhgoa_rag.retrieval.embedder import E5MultilingualEmbedder

    return E5MultilingualEmbedder()


def _build_timings(stages: dict[str, float], total: float) -> TimingsMs:
    fields = TimingsMs.model_fields
    kwargs = {k: v for k, v in stages.items() if k in fields}
    kwargs["total_backend"] = total
    return TimingsMs(**kwargs)


@router.post("/v1/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    timer = RequestTimer()
    settings = get_settings()

    # 1. Input guard
    with timer.stage("input_guard"):
        guard = check_input(req.question, settings.max_query_chars)

    if not guard.allowed:
        decision = (
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

    # 3. Embed query
    embedder = _get_embedder()
    with timer.stage("query_embed"):
        query_vec = embedder.embed_query(req.question)

    # 4. Retrieve (try Qdrant, fall back to error on failure)
    passages = []
    retrieval_mode = "none"
    try:
        from qdrant_client import QdrantClient
        from hhgoa_rag.retrieval.hybrid import HybridRetriever

        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=10)
        retriever = HybridRetriever(
            client,
            settings.qdrant_collection_alias,
            dense_k=settings.dense_prefetch_k,
            sparse_k=settings.sparse_prefetch_k,
            fused_k=settings.fused_k,
        )
        with timer.stage("qdrant_retrieve"):
            passages = retriever.retrieve(query_vec, req.question, lang_filter)
        retrieval_mode = "dense_sparse_rrf"
    except Exception:
        return QueryResponse(
            request_id=req.request_id,
            question=req.question,
            detected_language=detected_lang,
            answer=None,
            decision="error",
            reason_code="index_unavailable",
            grounded=False,
            confidence=0.0,
            citations=[],
            sources=[],
            retrieval_mode="none",
            index_manifest_id=settings.index_manifest_id,
            model_manifest_id=settings.model_manifest_id,
            timings_ms=_build_timings(timer.stages, timer.total_ms()),
            total_backend_ms=timer.total_ms(),
            error={"code": "index_unavailable", "message": "Vector index unavailable"},
        )

    # 5. Extract answer
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

    # 6. Grounding
    passage_texts = [p.get("payload", {}).get("text", "") for p in evidence]
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
            text=p.get("payload", {}).get("text", "")[:200],
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
