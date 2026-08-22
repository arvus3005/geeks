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

    clean_ans = answer.strip().rstrip(".?!।॥").lower()
    if not clean_ans:
        return False, 0.0
    all_passage_text = " ".join(passages).lower()
    if clean_ans in all_passage_text:
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

    grounded = passage_support >= threshold and query_overlap >= MIN_QUERY_ANSWER_OVERLAP
    confidence = round(min(passage_support, query_overlap), 4)
    return grounded, confidence
