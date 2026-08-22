"""Real tests for verify_grounding() -- rewritten 2026-08-22 after the real
eval loop measured false_confidence_rate=0.88 despite the old answer-vs-
passage-only check almost always passing (near-tautological for an
extractive system: the answer is usually a literal substring of the
passage it came from). See src/hhgoa_rag/guardrails/output_guards.py's
module docstring for the calibration evidence."""

from hhgoa_rag.guardrails.output_guards import _answer_type_mismatch, verify_grounding


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


# --- answer-type consistency gate (2026-08-22) -- see output_guards.py's
# module comment above _NUMERIC_QUESTION_RE for the failure class this
# targets: a topically-on-topic, query-word-reusing answer that is still
# the wrong TYPE of thing, which neither passage_support nor query_overlap
# can see since both only measure lexical overlap.


def test_numeric_question_without_a_number_in_the_answer_is_flagged():
    assert _answer_type_mismatch("how many legs does a spider have", "Spiders are arachnids, not insects.")


def test_numeric_question_with_a_digit_answer_is_not_flagged():
    assert not _answer_type_mismatch("how many legs does a spider have", "Spiders have eight legs.")


def test_numeric_question_with_a_spelled_out_number_is_not_flagged():
    assert not _answer_type_mismatch("how many legs does a spider have", "Spiders have eight legs, not six.")


def test_temporal_question_without_any_date_signal_is_flagged():
    assert _answer_type_mismatch("when was the eiffel tower built", "The Eiffel Tower is located in Paris, France.")


def test_temporal_question_with_a_year_is_not_flagged():
    assert not _answer_type_mismatch("when was the eiffel tower built", "The Eiffel Tower was completed in 1889.")


def test_when_used_as_a_conjunction_is_not_a_temporal_question():
    # Regression test: an earlier version of _TEMPORAL_QUESTION_RE matched
    # bare "when" anywhere in the query, which wrongly flagged this real
    # eval example -- "when" here is a subordinating conjunction ("what is
    # not present WHILE fermentation is used"), not a time question, and
    # the correct ground-truth answer ("Oxygen is not present...") contains
    # no date/number at all. Caught by inspecting the gate's own reject
    # list (eval/diagnose_skyline.py) before shipping.
    assert not _answer_type_mismatch(
        "what is not present when fermentation is used",
        "Alcoholic fermentation produces ethyl alcohol and carbon dioxide instead of lactic acid.",
    )


def test_non_numeric_non_temporal_question_is_never_flagged():
    assert not _answer_type_mismatch("what is the boiling point of water", "Water boils at 100 degrees Celsius.")
    assert not _answer_type_mismatch(
        "what is elastomer processing",
        "Elastomer processing is the manufacturing method used to shape rubber-like polymers.",
    )


def test_verify_grounding_declines_numeric_question_answered_with_no_number():
    passage = "Spiders are air-breathing arthropods with eight legs and chelicerae."
    grounded, _confidence = verify_grounding(
        "how many eyes does a spider have",
        "Spiders are air-breathing arthropods, not insects.",
        [passage],
    )
    assert not grounded
