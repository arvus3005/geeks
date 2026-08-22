"""Real tests for verify_grounding() -- rewritten 2026-08-22 after the real
eval loop measured false_confidence_rate=0.88 despite the old answer-vs-
passage-only check almost always passing (near-tautological for an
extractive system: the answer is usually a literal substring of the
passage it came from). See src/hhgoa_rag/guardrails/output_guards.py's
module docstring for the calibration evidence."""

from hhgoa_rag.guardrails.output_guards import verify_grounding


def test_genuine_answer_is_grounded():
    grounded, _confidence = verify_grounding(
        "what is the boiling point of water",
        "The boiling point of water is 100 degrees Celsius.",
        ["The boiling point of water is 100 degrees Celsius at sea level."],
    )
    assert grounded


def test_answer_sharing_no_real_content_with_query_is_not_grounded():
    # Real 2026-08-22 eval example, shape preserved: a well-formed sentence
    # extracted from a passage on a genuinely different topic than what
    # was asked -- old check passed this (support ~1.0 against its own
    # source passage) despite sharing zero real content words with the
    # question. query_overlap is the signal that catches this one.
    passage = "Grand-maman is slightly less formal, and there are several informal terms, including gra-mere, mémère, mémé and mamé."
    grounded, _confidence = verify_grounding(
        "what is french name for mom",
        passage,
        [passage],
    )
    assert not grounded


def test_known_limitation_same_entity_wrong_specific_fact_still_passes():
    # Documents a real, deliberate limitation rather than hiding it: an
    # answer squarely about the right entity (shares >= MIN_QUERY_ANSWER_OVERLAP
    # of the query's content words) but not the specific fact asked for
    # still passes -- content-word overlap can't distinguish "about Erie
    # Insurance" from "is Erie Insurance's corporate address". Real
    # 2026-08-22 eval example (overlap measured 0.50, above the 0.25 floor
    # even though it was also uncaught at the more aggressive 0.40
    # Youden's-J optimum -- see output_guards.py's module docstring for why
    # 0.25 was picked anyway). Catching this class needs an answer-type/
    # entity check, not more content-word overlap.
    passage = "Erie Insurance Exchange has rolled down lots of different insurance roads since its founding."
    grounded, _confidence = verify_grounding(
        "erie insurance corporate address",
        passage,
        [passage],
    )
    assert grounded  # known gap, not a regression -- see docstring above


def test_enumerator_stub_answer_is_not_grounded_even_when_it_is_the_whole_passage():
    # Real 2026-08-22 eval example: extract_answer's own sentence filter
    # recognizes this as junk, but since it's the passage's only sentence,
    # extract_answer's documented fallback returns it anyway -- this is the
    # downstream gate that must still catch it.
    passage = "In-text citations must be used in the following situations: 1."
    grounded, _confidence = verify_grounding(
        "how to do citations in an essay",
        passage,
        [passage],
    )
    assert not grounded


def test_navigation_junk_answer_is_not_grounded():
    passage = "You are here: Home / Products / Understanding TPEs"
    grounded, _confidence = verify_grounding("what is elastomer processing", passage, [passage])
    assert not grounded


def test_empty_answer_or_passages_is_not_grounded():
    assert verify_grounding("a query", "", ["some passage"]) == (False, 0.0)
    assert verify_grounding("a query", "an answer", []) == (False, 0.0)
