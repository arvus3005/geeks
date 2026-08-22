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


def test_global_deadline_tightens_a_generous_local_budget(monkeypatch):
    """A slow upstream stage (retrieval, embedding) can already have eaten
    into the total request budget before extract_answer even starts -- the
    local RERANKER_TIME_BUDGET_S alone doesn't know that. global_deadline
    must win when it's tighter than "now + local budget", not be ignored."""
    monkeypatch.setattr(extractive, "RERANKER_TIME_BUDGET_S", 5.0)  # generous local budget

    def slow_low_score(query: str, text: str) -> float:
        time.sleep(0.06)
        return -10.0  # below MIN_RERANKER_SCORE, isn't masking the real gate

    monkeypatch.setattr("hhgoa_rag.answer.reranker.score", slow_low_score)

    passages = [_passage("some passage text that is not obviously related")]
    already_expired_deadline = time.monotonic() - 1.0  # simulates zero time left
    answer, evidence = extractive.extract_answer(
        passages, "what is a corporation", global_deadline=already_expired_deadline
    )

    assert answer is None
    assert evidence == []


def test_generous_global_deadline_does_not_shorten_a_tight_local_budget(monkeypatch):
    """global_deadline should only ever tighten the budget, never loosen the
    fixed local ceiling -- min(local, global), not just global."""
    monkeypatch.setattr(extractive, "RERANKER_TIME_BUDGET_S", 0.05)

    def slow_low_score(query: str, text: str) -> float:
        time.sleep(0.06)  # exceeds the tight LOCAL budget even though global is generous
        return -10.0

    monkeypatch.setattr("hhgoa_rag.answer.reranker.score", slow_low_score)

    passages = [_passage("some passage text that is not obviously related")]
    generous_global_deadline = time.monotonic() + 60.0
    answer, evidence = extractive.extract_answer(
        passages, "what is a corporation", global_deadline=generous_global_deadline
    )

    assert answer is None
    assert evidence == []
