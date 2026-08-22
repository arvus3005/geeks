import re

from hhgoa_rag.answer.extractive import (
    _content_tokens,
    _is_enumerator_stub,
    _is_navigation_junk,
    _is_truncated_fragment,
)

# verify_grounding's passage-support check alone is near-tautological for
# an extractive system: the answer is usually a literal substring of the
# passage it was extracted from, so support is ~1.0 almost by construction.
# Real 2026-08-22 eval measurement: false_confidence_rate=0.88 (system
# confidently answered an unanswerable query) despite this check almost
# always passing -- it was never checking whether the answer actually
# addresses the QUESTION, only whether it's lexically present in its own
# source text. query_overlap below is the added, genuinely different
# signal: does the answer share real content with the query itself.
#
# Calibrated via eval/diagnose_query_overlap.py against the real 100-example
# eval set: genuine answers (grounded=True on an answerable query) measured
# median overlap=0.500; fabrications (grounded=True on an unanswerable
# query) measured median=0.333 -- real separation, not a clean split. The
# Youden's-J-optimal threshold on that sweep was 0.40 (catches 56.8% of
# fabrications, but wrongly rejects 22.4% of genuine answers). Deliberately
# NOT using that optimum: this project shipped a different threshold
# (MIN_RERANKER_SCORE=0.4 in extractive.py) picked the same way, the same
# day, and found -- only after validating against the REAL production
# retriever instead of the eval loop's own isolated sample -- that it
# declined 75% of real answerable queries. 0.25 trades some fabrication
# catch (34.1% vs 56.8%) for a false-refusal cost roughly a quarter the
# size (8.2% vs 22.4% in the same sweep). This signal doesn't depend on
# corpus/retrieval noise the way a reranker score does (it's a plain
# query-string vs answer-string comparison, not a function of the 55M-
# passage index), so it's less likely to hide the same kind of eval-vs-
# production gap -- but it has NOT been separately validated against real
# production traffic the way the reranker threshold was, and should be
# watched with the same caution until it has.
MIN_QUERY_ANSWER_OVERLAP = 0.25

# Answer-type consistency check: a genuinely different, third signal from
# passage_support/query_overlap above -- both of those measure how much
# LEXICAL content the answer shares with the passage/query, so a passage
# that's topically on-topic and reuses the query's own words can still
# clear both even when it answers the wrong SPECIFIC thing. This is
# exactly the failure class this project's own test suite already
# documents as a known gap (see tests/unit/test_output_guards.py's
# test_known_limitation_same_entity_wrong_specific_fact_still_passes --
# "Erie Insurance" corporate-address query answered with unrelated Erie
# Insurance prose, right entity, wrong fact) and that a competing team's
# own adversarial testing independently confirmed no lexical/similarity
# signal catches (reference/VoiceRagAgent-main/docs/answerability/
# PHASE5_ANSWERABILITY_REPORT.md section 7: "Prime Minister of India"
# queries retrieve a passage about Ireland's government -- same domain,
# wrong entity -- and 0/13 of their gate variants rejected it; section 6
# separately found NUMERIC queries were their single worst category,
# FPR=1.0, because a passage merely containing SOME number scores as
# relevant regardless of whether it's the right one).
#
# Deliberately narrow: only the two question types where "the answer
# obviously lacks the expected type" is unambiguous and cheap to check
# (numeric, temporal) -- who/where questions are NOT checked here, since
# reliably telling "a person's name" or "a place" from ordinary prose
# needs real NER, and a cheap regex guessing wrong would just trade false
# confidence for false refusal, the exact tradeoff this project has
# already been burned by twice (see extractive.py's MIN_RERANKER_SCORE
# history). Each check is deliberately lenient about what COUNTS as
# satisfying the type (e.g. any digit at all counts as "temporal", since
# a real date answer might be "3 days later" not a bare year) -- the goal
# is only to catch the unambiguous case where the expected type is
# completely absent, not to also start rejecting borderline-but-correct
# answers.
_NUMERIC_QUESTION_RE = re.compile(
    r"\bhow (many|much)\b|\bhow old\b|\bwhat (percentage|percent|fraction)\b", re.IGNORECASE
)
_TEMPORAL_QUESTION_RE = re.compile(
    # "when" only counts as a temporal-question opener at the start of the
    # query or immediately before an auxiliary verb ("when did/was/is
    # ...") -- a bare \bwhen\b anywhere false-triggered on a real eval
    # example: "what is not present when fermentation is used" uses "when"
    # as a subordinating conjunction, not a time question, and got wrongly
    # declined by an earlier version of this regex (caught via
    # eval/diagnose_skyline.py inspecting the gate's own reject list
    # before shipping it).
    r"^when\b|\bwhen (did|was|were|is|are|does|do|will|won't|wasn't|weren't)\b|\bwhat (year|date|day|month|time)\b",
    re.IGNORECASE,
)
_DIGIT_RE = re.compile(r"\d")
_YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20\d{2}|21\d{2})\b")
_MONTH_OR_DAY_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)
_DATE_WORD_RE = re.compile(
    r"\b(century|decade|ad|bc|ce|bce|o'clock|am|pm|ago|yesterday|today|tomorrow)\b", re.IGNORECASE
)
_NUMBER_WORDS = frozenset(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy "
    "eighty ninety hundred thousand million billion dozen couple few several".split()
)


def _has_number(text: str) -> bool:
    if _DIGIT_RE.search(text):
        return True
    return bool(set(re.findall(r"[a-zA-Z]+", text.lower())) & _NUMBER_WORDS)


def _has_temporal_signal(text: str) -> bool:
    return bool(
        _DIGIT_RE.search(text)
        or _YEAR_RE.search(text)
        or _MONTH_OR_DAY_RE.search(text)
        or _DATE_WORD_RE.search(text)
    )


def _answer_type_mismatch(query: str, answer: str) -> bool:
    """True if the query clearly signals a numeric or temporal expected
    answer type and the answer contains neither digits nor the
    corresponding word forms. See the module comment above for scope and
    the real failure class this targets."""
    if _NUMERIC_QUESTION_RE.search(query):
        return not _has_number(answer)
    if _TEMPORAL_QUESTION_RE.search(query):
        return not _has_temporal_signal(answer)
    return False


def _support_and_overlap(query: str, answer: str, passages: list[str]) -> tuple[float, float]:
    """Raw (passage_support, query_overlap) pair, with no thresholding
    applied -- factored out of verify_grounding so eval/diagnose_skyline.py
    can sweep joint thresholds over real data using the exact formulas
    production uses, instead of a reimplementation that could silently
    drift from it."""
    clean_ans = answer.strip().rstrip(".?!।॥").lower()
    all_passage_text = " ".join(passages).lower()
    if clean_ans and clean_ans in all_passage_text:
        passage_support = 1.0
    else:
        answer_tokens = set(clean_ans.split())
        passage_tokens = set(all_passage_text.split())
        passage_support = len(answer_tokens & passage_tokens) / max(len(answer_tokens), 1)

    query_tokens = _content_tokens(query)
    if query_tokens:
        query_overlap = len(query_tokens & _content_tokens(answer)) / len(query_tokens)
    else:
        query_overlap = 1.0  # no content words to check against (rare) -- don't penalize
    return passage_support, query_overlap


def verify_grounding(
    query: str, answer: str, passages: list[str], threshold: float = 0.45
) -> tuple[bool, float]:
    """Check that answer text is supported by its source passages AND
    actually addresses the query -- not just lexically contained in the
    passage it was extracted from (see module docstring above)."""
    if not answer or not passages:
        return False, 0.0

    # Re-check the same junk-sentence patterns extract_answer's own
    # sentence-selection loop uses (hhgoa_rag.answer.extractive), against
    # the FINAL returned answer text. extract_answer's own filtering only
    # affects which sentence gets picked WITHIN an already-accepted
    # passage; when every sentence in a short (often single-sentence)
    # passage is junk, its documented, intentionally-tested fallback still
    # returns the whole passage text rather than declining (see
    # tests/unit/test_extractive_junk_filters.py's
    # test_extract_answer_falls_back_when_every_sentence_echoes -- correct
    # for extract_answer itself, since a real multi-sentence passage that
    # opens with junk still needs its later content considered). This is
    # the last real gate before that same junk pattern reaches a user. Real
    # 2026-08-22 eval example: "how to do citations in an essay" ->
    # "In-text citations must be used in the following situations: 1." --
    # extract_answer's own filter recognized this sentence as junk, but it
    # was the passage's only sentence, so its fallback returned it anyway;
    # this check catches it here instead.
    if _is_enumerator_stub(answer) or _is_truncated_fragment(answer) or _is_navigation_junk(answer):
        return False, 0.0

    # See _answer_type_mismatch's module comment: a third, genuinely
    # different signal from the two lexical-overlap ones below -- catches
    # a topically-correct, query-word-reusing answer that is still the
    # wrong TYPE of thing (e.g. "when" answered with no date/number at
    # all), which neither passage_support nor query_overlap can see.
    if _answer_type_mismatch(query, answer):
        return False, 0.0

    clean_ans = answer.strip().rstrip(".?!।॥").lower()
    if not clean_ans:
        return False, 0.0

    passage_support, query_overlap = _support_and_overlap(query, answer, passages)
    grounded = passage_support >= threshold and query_overlap >= MIN_QUERY_ANSWER_OVERLAP
    confidence = round(min(passage_support, query_overlap), 4)
    return grounded, confidence
