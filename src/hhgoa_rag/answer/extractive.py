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
# question. Query<->passage content overlap is a fast, cheap first signal
# that distinguishes "topically relevant" from "grammatically coincidental"
# -- real example this fixes, caught by rag-local-eval-loop's Reliability
# check (2026-08-22): query "hexadecimal numbers to binary numbers" against
# an irrelevant passage was answering anyway pre-fix (0% content overlap,
# should have declined).
MIN_QUERY_OVERLAP = 0.4

# Below this cosine similarity between the query's own embedding and the
# best-matching passage's embedding, decline. A first attempt at this gate
# (2026-08-22, reverted -- see git history) was calibrated on hand-written,
# full-sentence English pairs ("What is a corporation?") and scored
# 0.88+ for genuinely relevant matches -- but real MSMARCO-style queries
# are short, informal fragments ("hexadecimal numbers to binary numbers"),
# and that mismatch in QUERY STYLE (not just language) miscalibrated the
# threshold so badly it drove the live production abstain rate to 80.8%.
#
# Redone properly: sampled 48 real passages across all 6 indexed languages
# (hi/bn/gu/ta/mr/ur), built SHORT natural-style queries from each (first
# ~6 words, lowercased -- matching real search-query shape, not a clean
# sentence), and measured real cosine similarity for (a) the guaranteed-
# relevant self-retrieval pair and (b) a same-language cross-pair (query
# paired with a DIFFERENT sampled passage, as a stand-in for "irrelevant").
# Per-language results (relevant min / irrelevant max): hi 0.802/0.794,
# bn 0.854/0.803, gu 0.834/0.805, ta 0.800/0.770, mr 0.826/0.807,
# ur 0.837/0.818 -- a real, consistent gap around 0.80 across every
# language, not just English. 0.80 is set conservatively at the low end of
# every language's relevant-pair range so it doesn't cost real answerable
# queries in any one language more than the others.
MIN_SEMANTIC_SIMILARITY = 0.80


def _content_tokens(text: str) -> set[str]:
    return {t for t in text.lower().split() if t not in _STOPWORDS}


def extract_answer(passages: list[dict], query: str) -> tuple[str | None, list[dict]]:
    """Extract best-supported answer span from top passages.

    Ranks the top-3 candidates by REAL embedding cosine similarity to the
    query (not lexical overlap -- an earlier version picked the
    lexical-overlap winner first and only used semantic similarity as a
    gate afterward, which meant a lexically-strong-but-semantically-wrong
    candidate could still get picked over a better one sitting right next
    to it in the same top-3). Declines (returns None) if even the best
    candidate doesn't clear MIN_QUERY_OVERLAP (cheap lexical pre-filter,
    catches zero-content-word cases before spending an embed call) or
    MIN_SEMANTIC_SIMILARITY (the real relevance gate).
    """
    if not passages:
        return None, []

    query_tokens = _content_tokens(query)
    candidates: list[tuple[dict, str, float]] = []  # (passage, text, overlap)

    for p in passages[:3]:
        payload = p.get("payload", {})
        # Local hybrid store sets TEXT_FIELD (chunk_text); fallback to legacy text
        text = payload.get(TEXT_FIELD, "") or payload.get("text", "")
        if not text:
            continue
        overlap = len(query_tokens & _content_tokens(text)) / max(len(query_tokens), 1)
        if overlap >= MIN_QUERY_OVERLAP:
            candidates.append((p, text, overlap))

    if not candidates:
        return None, []

    from hhgoa_rag.retrieval.local_embedder import embed_passage, embed_query

    query_vec = embed_query(query)
    best_passage = None
    best_text = ""
    best_similarity = -1.0

    for p, text, _overlap in candidates:
        passage_vec = embed_passage(text)
        # Both vectors are L2-normalized (see local_embedder._embed), so
        # cosine similarity is just the dot product.
        similarity = sum(a * b for a, b in zip(query_vec, passage_vec, strict=True))
        if similarity > best_similarity:
            best_similarity = similarity
            best_passage = p
            best_text = text

    if best_passage is None or best_similarity < MIN_SEMANTIC_SIMILARITY:
        return None, []

    sentences = best_text.split(". ")
    answer = sentences[0] if sentences else best_text[:300]
    if not answer.endswith("."):
        answer += "."

    return answer, passages[:3]
