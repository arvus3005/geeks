"""Voice query, transcription, and speech synthesis routes."""

from __future__ import annotations

import base64
import logging
import time
from typing import Literal

from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel, Field

from hhgoa_rag.answer.extractive import extract_answer
from hhgoa_rag.config.settings import get_settings
from hhgoa_rag.guardrails.input_guards import check_input
from hhgoa_rag.guardrails.output_guards import verify_grounding
from hhgoa_rag.observability.timing import RequestTimer
from hhgoa_rag.retrieval.language_routing import detect_language, get_language_filter
from hhgoa_rag.retrieval.local_embedder import embed_query
from hhgoa_rag.retrieval_contract import TEXT_FIELD
from hhgoa_rag.schemas.query import Citation, TimingsMs
from hhgoa_rag.stt.sarvam import SarvamSTTAdapter
from hhgoa_rag.stt.tts import SarvamTTSAdapter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/voice", tags=["voice"])


class TranscribeResponse(BaseModel):
    transcript: str
    detected_language: str | None = None
    stt_latency_ms: float = 0.0
    error: str | None = None


class TTSRequest(BaseModel):
    text: str
    language: str = "en"


class TTSResponse(BaseModel):
    audio_base64: str | None = None
    language_code: str
    latency_ms: float = 0.0
    error: str | None = None


class VoiceQueryResponse(BaseModel):
    request_id: str
    transcript: str
    detected_language: str | None = None
    answer: str | None = None
    audio_base64: str | None = None
    decision: Literal["allow", "abstain", "refuse", "error"]
    reason_code: str
    grounded: bool
    confidence: float
    citations: list[Citation] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    retrieval_mode: str = "local_hybrid_sharded_bm25_hnsw_rrf"
    timings_ms: dict[str, float] = Field(default_factory=dict)
    stt_latency_ms: float = 0.0
    tts_latency_ms: float = 0.0
    total_backend_ms: float = 0.0
    error: dict | None = None


def _get_stt_adapter() -> SarvamSTTAdapter:
    settings = get_settings()
    return SarvamSTTAdapter(api_key=settings.sarvam_api_key or "", model=settings.sarvam_model)


def _get_tts_adapter() -> SarvamTTSAdapter:
    settings = get_settings()
    return SarvamTTSAdapter(api_key=settings.sarvam_api_key or "")


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    language_hint: str | None = Form(None),
):
    """Transcribe an audio file using Sarvam Saaras STT."""
    audio_bytes = await file.read()
    stt = _get_stt_adapter()

    if not stt.api_key:
        return TranscribeResponse(
            transcript="",
            detected_language=language_hint,
            error="SARVAM_API_KEY is not configured",
        )

    try:
        result = await stt.transcribe(audio_bytes, language_hint=language_hint)
        return TranscribeResponse(
            transcript=result.text,
            detected_language=result.language,
            stt_latency_ms=result.transcript_latency_ms,
        )
    except Exception as exc:
        logger.error("STT transcription error: %s", exc)
        return TranscribeResponse(
            transcript="",
            detected_language=language_hint,
            error=str(exc),
        )


@router.post("/tts", response_model=TTSResponse)
async def synthesize_speech(req: TTSRequest):
    """Synthesize text to speech using Sarvam Bulbul TTS."""
    tts = _get_tts_adapter()
    res = await tts.synthesize(req.text, language=req.language)
    return TTSResponse(
        audio_base64=res.audio_base64,
        language_code=res.language_code,
        latency_ms=res.latency_ms,
        error=res.error,
    )


@router.post("/query", response_model=VoiceQueryResponse)
async def voice_query_endpoint(
    file: UploadFile = File(...),
    language_hint: str | None = Form(None),
    generate_audio: bool = Form(True),
):
    """End-to-end Voice RAG endpoint: Audio -> STT -> RAG Pipeline -> TTS."""
    import uuid

    req_id = str(uuid.uuid4())
    audio_bytes = await file.read()
    settings = get_settings()
    stt = _get_stt_adapter()

    stt_latency_ms = 0.0
    transcript = ""
    detected_lang_stt = language_hint

    # 1. Speech-to-Text
    if stt.api_key:
        try:
            stt_res = await stt.transcribe(audio_bytes, language_hint=language_hint)
            transcript = stt_res.text
            detected_lang_stt = stt_res.language or language_hint
            stt_latency_ms = stt_res.transcript_latency_ms
        except Exception as exc:
            logger.warning("STT transcription failed: %s", exc)

    if not transcript:
        return VoiceQueryResponse(
            request_id=req_id,
            transcript="",
            detected_language=detected_lang_stt,
            answer=None,
            decision="error",
            reason_code="stt_failed",
            grounded=False,
            confidence=0.0,
            error={"code": "stt_failed", "message": "Voice transcription failed or no audio recorded."},
        )

    # 2. Run standard RAG pipeline with Timer
    timer = RequestTimer()

    # Input guard
    with timer.stage("input_guard"):
        guard = check_input(transcript, settings.max_query_chars)

    if not guard.allowed:
        decision = "refuse" if guard.reason_code in ("unsafe", "credential_request", "prompt_injection") else "abstain"
        return VoiceQueryResponse(
            request_id=req_id,
            transcript=transcript,
            detected_language=detected_lang_stt,
            answer=None,
            decision=decision,
            reason_code=guard.reason_code,
            grounded=False,
            confidence=0.0,
            timings_ms=timer.stages,
            stt_latency_ms=stt_latency_ms,
            total_backend_ms=timer.total_ms(),
        )

    # Language detect
    with timer.stage("language_detect"):
        detected_lang = detect_language(transcript)
        lang_filter = get_language_filter(detected_lang, language_hint or detected_lang_stt)

    # Retrieve
    from hhgoa_rag.retrieval.sharded_local_hybrid_store import search as local_hybrid_search

    try:
        with timer.stage("query_embed"):
            query_vector = embed_query(transcript)
        with timer.stage("local_hybrid_retrieve"):
            hits = local_hybrid_search(
                query_text=transcript,
                query_vector=query_vector,
                languages=lang_filter,
                top_k=settings.retrieval_top_k,
            )
    except Exception as exc:
        logger.error("Voice retrieval failed: %s", exc)
        return VoiceQueryResponse(
            request_id=req_id,
            transcript=transcript,
            detected_language=detected_lang,
            answer=None,
            decision="error",
            reason_code="index_unavailable",
            grounded=False,
            confidence=0.0,
            error={"code": "index_unavailable", "message": str(exc)},
            timings_ms=timer.stages,
            stt_latency_ms=stt_latency_ms,
            total_backend_ms=timer.total_ms(),
        )

    passages = [{"id": h["id"], "score": h["score"], "payload": h["fields"]} for h in hits]

    # Extract answer
    with timer.stage("answer_extract"):
        answer, evidence = extract_answer(passages, transcript)

    if not answer:
        return VoiceQueryResponse(
            request_id=req_id,
            transcript=transcript,
            detected_language=detected_lang,
            answer=None,
            decision="abstain",
            reason_code="weak_retrieval",
            grounded=False,
            confidence=0.0,
            timings_ms=timer.stages,
            stt_latency_ms=stt_latency_ms,
            total_backend_ms=timer.total_ms(),
        )

    # Grounding verify
    passage_texts = [p.get("payload", {}).get(TEXT_FIELD, "") for p in evidence]
    with timer.stage("grounding_verify"):
        grounded, confidence = verify_grounding(answer, passage_texts, settings.min_retrieval_score)

    if not grounded:
        return VoiceQueryResponse(
            request_id=req_id,
            transcript=transcript,
            detected_language=detected_lang,
            answer=None,
            decision="abstain",
            reason_code="ungrounded",
            grounded=False,
            confidence=confidence,
            timings_ms=timer.stages,
            stt_latency_ms=stt_latency_ms,
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

    total_backend_ms = timer.total_ms()

    # Optional TTS synthesis
    audio_b64 = None
    tts_latency_ms = 0.0
    if generate_audio and answer:
        tts = _get_tts_adapter()
        tts_res = await tts.synthesize(answer, language=detected_lang)
        audio_b64 = tts_res.audio_base64
        tts_latency_ms = tts_res.latency_ms

    return VoiceQueryResponse(
        request_id=req_id,
        transcript=transcript,
        detected_language=detected_lang,
        answer=answer,
        audio_base64=audio_b64,
        decision="allow",
        reason_code="answered",
        grounded=True,
        confidence=confidence,
        citations=citations,
        sources=[c.passage_id for c in citations],
        timings_ms=timer.stages,
        stt_latency_ms=stt_latency_ms,
        tts_latency_ms=tts_latency_ms,
        total_backend_ms=total_backend_ms,
    )
