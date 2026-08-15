def extract_answer(passages: list[dict], query: str) -> tuple[str | None, list[dict]]:
    """Extract best-supported answer span from top passages."""
    if not passages:
        return None, []

    query_tokens = set(query.lower().split())

    best_passage = None
    best_score = -1.0

    for p in passages[:3]:
        text = p.get("payload", {}).get("text", "")
        if not text:
            continue
        passage_tokens = set(text.lower().split())
        overlap = len(query_tokens & passage_tokens) / max(len(query_tokens), 1)
        qdrant_score = p.get("score", 0.0)
        combined = 0.7 * qdrant_score + 0.3 * overlap
        if combined > best_score:
            best_score = combined
            best_passage = p

    if best_passage is None:
        return None, []

    # Return first complete sentence or first 300 chars
    text = best_passage.get("payload", {}).get("text", "")
    sentences = text.split(". ")
    answer = sentences[0] if sentences else text[:300]
    if not answer.endswith("."):
        answer += "."

    return answer, passages[:3]
