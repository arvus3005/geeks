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
# best-matching SENTENCE's embedding (not the whole passage -- see below),
# decline. A first attempt at this gate (2026-08-22, reverted -- see git
# history) was calibrated on hand-written, full-sentence English pairs
# ("What is a corporation?") and scored 0.88+ for genuinely relevant
# matches -- but real MSMARCO-style queries are short, informal fragments
# ("hexadecimal numbers to binary numbers"), and that mismatch in QUERY
# STYLE (not just language) miscalibrated the threshold so badly it drove
# the live production abstain rate to 80.8%.
#
# Redone properly: sampled 48 real passages across all 6 indexed languages
# (hi/bn/gu/ta/mr/ur), built SHORT natural-style queries from each (first
# ~6 words, lowercased -- matching real search-query shape, not a clean
# sentence), and measured real cosine similarity for (a) the guaranteed-
# relevant self-retrieval pair's best-matching sentence and (b) the
# best-matching sentence within a DIFFERENT same-language passage (a
# stand-in for "irrelevant"). Per-language relevant-mean / irrelevant-mean:
# hi 0.878/0.782, bn 0.878/0.791, gu 0.888/0.795, ta 0.903/0.793,
# mr 0.867/0.793, ur 0.852/0.783 -- a real, consistent ~0.08-0.11 gap in
# every language. The tails overlap somewhat at n=8/language (e.g. mr's
# weakest relevant sample and ta's strongest irrelevant one land close
# together), so 0.80 is set conservatively near the low end of every
# language's relevant range rather than at the exact midpoint of the gap.
MIN_SEMANTIC_SIMILARITY = 0.80

# Passages are scored sentence-by-sentence, not as one block: a topically
# broad passage can have one sharply on-topic sentence among several
# irrelevant ones, which whole-passage scoring dilutes. Cap the number of
# sentences checked per candidate passage to bound embed-call latency (a
# real production passage rarely needs more than this to find its best
# match, and MSMARCO passages are themselves already short).
MAX_SENTENCES_PER_PASSAGE = 6


def _content_tokens(text: str) -> set[str]:
    return {t for t in text.lower().split() if t not in _STOPWORDS}


def _sentences(text: str) -> list[str]:
    parts = [s.strip() for s in text.split(". ") if s.strip()]
    return parts[:MAX_SENTENCES_PER_PASSAGE] if parts else [text]


def extract_answer(passages: list[dict], query: str) -> tuple[str | None, list[dict]]:
    """Extract best-supported answer span from top passages.

    Scores candidate SENTENCES, not whole passages: a topically broad
    passage can contain one sentence that actually answers the question
    among several that don't, which whole-passage scoring blurs together
    (found via a competing team's reference implementation, which had
    already measured this exact class of error -- see the commit this
    function was last changed in). Ranks by REAL embedding cosine
    similarity between the query and each candidate sentence, not lexical
    overlap (an earlier version picked the lexical-overlap winner first
    and only used semantic similarity as a gate afterward, letting a
    lexically-strong-but-semantically-wrong candidate beat a better one
    sitting right next to it). Declines (returns None) if even the best
    sentence doesn't clear MIN_QUERY_OVERLAP at the passage level (cheap
    lexical pre-filter, avoids spending embed calls on passages with zero
    real content overlap) or MIN_SEMANTIC_SIMILARITY (the real relevance
    gate, now checked at sentence granularity).
    """
    if not passages:
        return None, []

    query_tokens = _content_tokens(query)
    candidate_texts: list[str] = []

    for p in passages[:3]:
        payload = p.get("payload", {})
        # Local hybrid store sets TEXT_FIELD (chunk_text); fallback to legacy text
        text = payload.get(TEXT_FIELD, "") or payload.get("text", "")
        if not text:
            continue
        overlap = len(query_tokens & _content_tokens(text)) / max(len(query_tokens), 1)
        if overlap >= MIN_QUERY_OVERLAP:
            candidate_texts.append(text)

    if not candidate_texts:
        return None, []

    from hhgoa_rag.retrieval.local_embedder import embed_passage, embed_query

    query_vec = embed_query(query)
    best_sentence = ""
    best_similarity = -1.0

    for text in candidate_texts:
        for sentence in _sentences(text):
            sentence_vec = embed_passage(sentence)
            # Both vectors are L2-normalized (see local_embedder._embed), so
            # cosine similarity is just the dot product.
            similarity = sum(a * b for a, b in zip(query_vec, sentence_vec, strict=True))
            if similarity > best_similarity:
                best_similarity = similarity
                best_sentence = sentence

    if not best_sentence or best_similarity < MIN_SEMANTIC_SIMILARITY:
        return None, []

    answer = best_sentence if best_sentence.endswith(".") else best_sentence + "."
    return answer, passages[:3]
