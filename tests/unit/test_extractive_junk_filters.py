"""Real tests for the two answer-quality filters added after manually
reading actual eval-loop false-confidence examples (2026-08-22): website
navigation boilerplate scored as relevant by the reranker, and answer
sentences that are pure question-echoes."""

from hhgoa_rag.answer.extractive import (
    _content_tokens,
    _is_enumerator_stub,
    _is_navigation_junk,
    _is_pointer_sentence,
    _is_question_echo,
    _is_truncated_fragment,
    extract_answer,
)


def _passage(text: str) -> dict:
    return {"payload": {"chunk_text": text}, "score": 0.03}


def test_navigation_breadcrumb_is_junk():
    assert _is_navigation_junk(
        "You are here: Home / Products / Thermoplastic Elastomer (TPE) Compounds / Understanding TPEs"
    )


def test_normal_prose_is_not_junk():
    assert not _is_navigation_junk("A corporation is a legal entity separate from its owners.")


def test_pointer_sentence_is_flagged():
    assert _is_pointer_sentence("See the most popular majors at Clemson University.")
    assert _is_pointer_sentence("Learn more about thermoplastic elastomers here.")


def test_real_content_sentence_is_not_a_pointer():
    assert not _is_pointer_sentence("Clemson University is known for engineering and business.")


def test_pure_echo_sentence_is_flagged():
    query_tokens = _content_tokens("what type of photon has the greatest energy")
    assert _is_question_echo("Which photon has the greatest energy.", query_tokens)


def test_real_answer_with_new_information_is_not_flagged_as_echo():
    query_tokens = _content_tokens("what is the boiling point of water")
    assert not _is_question_echo("The boiling point of water is 100 degrees Celsius.", query_tokens)


def test_extract_answer_skips_navigation_junk_passage(monkeypatch):
    monkeypatch.setattr(
        "hhgoa_rag.answer.reranker.score", lambda query, text: 5.0
    )  # would clear MIN_RERANKER_SCORE if not filtered out first
    passages = [
        _passage("You are here: Home / Products / Elastomer Processing / Understanding TPEs"),
        _passage("Elastomer processing is the manufacturing method used to shape rubber-like polymers."),
    ]
    answer, evidence = extract_answer(passages, "what is elastomer processing")
    assert answer is not None
    assert "manufacturing method" in answer


def test_truncated_abbreviation_fragment_is_flagged():
    # Real eval example: "(e." left over from naive period-splitting of
    # "...other organisms (e.g., humans)..."
    assert _is_truncated_fragment("(e.")
    assert _is_truncated_fragment("i.")


def test_real_short_sentence_is_not_a_fragment():
    assert not _is_truncated_fragment("Water boils at 100 degrees.")


def test_enumerator_stub_is_flagged():
    # Real eval examples: a dangling list intro and a dangling MC-quiz stem.
    assert _is_enumerator_stub("In-text citations must be used in the following situations: 1.")
    assert _is_enumerator_stub("The New Deal did all of the following EXCEPT: A)")


def test_label_value_sentence_is_not_an_enumerator_stub():
    # Must not catch a real short label:value answer.
    assert not _is_enumerator_stub("Time: 5pm")
    assert not _is_enumerator_stub("Final score: 2-1")


def test_extract_answer_falls_back_when_every_sentence_echoes(monkeypatch):
    monkeypatch.setattr("hhgoa_rag.answer.reranker.score", lambda query, text: 5.0)
    # A single-sentence passage that IS an echo -- best_sentence must fall
    # back to the whole passage text rather than ever returning nothing.
    passages = [_passage("Which photon has the greatest energy.")]
    answer, evidence = extract_answer(passages, "what type of photon has the greatest energy")
    assert answer is not None  # falls back to best_text, doesn't crash or return None
