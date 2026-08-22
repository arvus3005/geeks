import uuid
from typing import Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str
    language_hint: str | None = None
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    debug: bool = False


class Citation(BaseModel):
    passage_id: str
    language: str
    chunk_ordinal: int
    text: str
    score: float


class TimingsMs(BaseModel):
    input_guard: float = 0.0
    language_detect: float = 0.0
    query_embed: float = 0.0
    pinecone_retrieve: float = 0.0  # kept for old response compatibility; unused now
    local_hybrid_retrieve: float = 0.0
    rerank: float = 0.0
    evidence_check: float = 0.0
    answer_extract: float = 0.0
    grounding_verify: float = 0.0
    serialize: float = 0.0
    total_backend: float = 0.0


class QueryResponse(BaseModel):
    request_id: str
    question: str
    detected_language: str | None
    answer: str | None
    decision: Literal["allow", "abstain", "refuse", "error"]
    reason_code: str
    grounded: bool
    confidence: float
    citations: list[Citation]
    sources: list[str]
    retrieval_mode: str
    index_manifest_id: str
    model_manifest_id: str
    timings_ms: TimingsMs
    total_backend_ms: float
    error: dict | None = None
