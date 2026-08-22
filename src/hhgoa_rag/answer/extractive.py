from hhgoa_rag.retrieval_contract import TEXT_FIELD

# Common English function words, stripped before computing query<->passage
# overlap. Without this, a short question like "what is a corporation"
# tokenizes to {"what","is","a","corporation"} and a completely unrelated
# passage that merely contains "what" and "is" already scores 50% overlap
# -- inflating relevance on grammar, not content. Deliberately small/cheap
# (not a full NLTK-style list) since this only needs to remove the highest-
# frequency words that would otherwise dominate a 4-6 token query.
_STOPWORDS = frozenset(
    "a an the is are was were be been being of to in on at for and or "
    "what who whom whose which how why when where does do did can "
    "will would should could this that these those it its as with".split()
)

# Below this fraction of shared CONTENT words between the query and the
# best-matching passage, decline rather than extract an answer. Exists
# because grounding (extract_answer's caller, hhgoa_rag.guardrails.
# output_guards.verify_grounding) checks the answer against the passage it
# was extracted FROM -- for extractive answering that's nearly tautological
# (the "answer" is literally a sentence out of that passage, so it always
# "grounds"), so it can't catch a passage that's simply irrelevant to the
# question. Query<->passage content overlap is the signal that actually
# distinguishes "topically relevant" from "lexically coincidental" — real
# example this fixes, caught by rag-local-eval-loop's Reliability check
# (2026-08-22): query "hexadecimal numbers to binary numbers" against an
# irrelevant passage was answering anyway pre-fix (0% content overlap,
# should have declined).
MIN_QUERY_OVERLAP = 0.4


def _content_tokens(text: str) -> set[str]:
    return {t for t in text.lower().split() if t not in _STOPWORDS}


def extract_answer(passages: list[dict], query: str) -> tuple[str | None, list[dict]]:
    """Extract best-supported answer span from top passages. Declines
    (returns None) if even the best-scoring candidate doesn't share enough
    real content with the query -- see MIN_QUERY_OVERLAP."""
    if not passages:
        return None, []

    query_tokens = _content_tokens(query)

    best_passage = None
    best_score = -1.0
    best_overlap = 0.0

    for p in passages[:3]:
        payload = p.get("payload", {})
        # Local hybrid store sets TEXT_FIELD (chunk_text); fallback to legacy text
        text = payload.get(TEXT_FIELD, "") or payload.get("text", "")
        if not text:
            continue
        passage_tokens = _content_tokens(text)
        overlap = len(query_tokens & passage_tokens) / max(len(query_tokens), 1)
        retrieval_score = p.get("score", 0.0)
        combined = 0.7 * retrieval_score + 0.3 * overlap
        if combined > best_score:
            best_score = combined
            best_overlap = overlap
            best_passage = p

    if best_passage is None or best_overlap < MIN_QUERY_OVERLAP:
        return None, []

    payload = best_passage.get("payload", {})
    text = payload.get(TEXT_FIELD, "") or payload.get("text", "")
    sentences = text.split(". ")
    answer = sentences[0] if sentences else text[:300]
    if not answer.endswith("."):
        answer += "."

    return answer, passages[:3]
