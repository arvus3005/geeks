"""Real test for extract_answer's reranker time-budget fallback -- forces
the slow path via monkeypatch rather than assuming the deadline check works."""

import time

from hhgoa_rag.answer import extractive


def _passage(text: str) -> dict:
    return {"payload": {"chunk_text": text}, "score": 0.03}


def test_declines_when_reranker_never_clears_deadline(monkeypatch):
    monkeypatch.setattr(extractive, "RERANKER_TIME_BUDGET_S", 0.05)

    def slow_low_score(query: str, text: str) -> float:
        time.sleep(0.06)  # exceeds the budget on the very first call
        return -10.0  # also below MIN_RERANKER_SCORE, so this isn't masking the real gate

    monkeypatch.setattr("hhgoa_rag.answer.reranker.score", slow_low_score)

    passages = [_passage("some passage text that is not obviously related")]
    answer, evidence = extractive.extract_answer(passages, "what is a corporation")

    assert answer is None
    assert evidence == []


def test_answers_normally_within_budget(monkeypatch):
    monkeypatch.setattr(extractive, "RERANKER_TIME_BUDGET_S", 5.0)

    def fast_high_score(query: str, text: str) -> float:
        return 5.0  # well above MIN_RERANKER_SCORE

    monkeypatch.setattr("hhgoa_rag.answer.reranker.score", fast_high_score)

    passages = [_passage("A corporation is a legal entity separate from its owners.")]
    answer, evidence = extractive.extract_answer(passages, "what is a corporation")

    assert answer is not None
    assert "corporation" in answer.lower()
    assert evidence != []
