"""Generator module for the HH Goa 2026 Task 2 eval harness
(rag-local-eval-loop, see docs/wiring-in-the-eval-loop.pdf and
TARGET_INTERFACE.md in that suite's own repo).

Wraps this project's REAL production answer pipeline — the same
extract_answer() + verify_grounding() call sequence
src/hhgoa_rag/api/routes/query.py runs for every live /v1/query request —
so the eval suite's Faithfulness/Correctness/Reliability checks score the
actual deployed guardrail behavior, not a stand-in reimplementation.

`.grounded` maps directly onto this project's own abstain decision: if
verify_grounding() would make the live API abstain (reason_code
"ungrounded"), this reports grounded=False and an empty answer text,
exactly mirroring what a real caller of /v1/query receives in that case.
Getting this signal right matters — the eval suite's own docs note a
generator that always reports grounded=True can never be caught
fabricating by the Reliability ("lying factor") check.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from hhgoa_rag.answer.extractive import extract_answer
from hhgoa_rag.config.settings import get_settings
from hhgoa_rag.guardrails.output_guards import verify_grounding

_MODEL_LABEL = "hhgoa-local-hybrid-extractive-v1"


@dataclass
class _EvalAnswer:
    text: str
    grounded: bool
    generation_ms: float
    model: str


def generate_answer(query: str, results: list) -> _EvalAnswer:
    t0 = time.monotonic()

    # eval/pipeline.py builds `results` as plain objects with .text/.source
    # (duck-typed, not this project's own passage dict shape) -- adapt into
    # the {"payload": {...}, "score": ...} shape extract_answer expects.
    passages = [
        {"id": getattr(r, "source", ""), "score": 1.0, "payload": {"chunk_text": r.text, "text": r.text}}
        for r in results
    ]

    answer, evidence = extract_answer(passages, query)
    if not answer:
        return _EvalAnswer(
            text="", grounded=False, generation_ms=(time.monotonic() - t0) * 1000, model=_MODEL_LABEL
        )

    passage_texts = [p["payload"]["chunk_text"] for p in evidence]
    settings = get_settings()
    grounded, _confidence = verify_grounding(answer, passage_texts, settings.min_retrieval_score)

    return _EvalAnswer(
        text=answer if grounded else "",
        grounded=grounded,
        generation_ms=(time.monotonic() - t0) * 1000,
        model=_MODEL_LABEL,
    )
