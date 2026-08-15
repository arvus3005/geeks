def verify_grounding(
    answer: str, passages: list[str], threshold: float = 0.45
) -> tuple[bool, float]:
    """Check that answer text is supported by passages."""
    if not answer or not passages:
        return False, 0.0
    answer_tokens = set(answer.lower().split())
    if len(answer_tokens) < 3:
        return False, 0.0
    all_passage_text = " ".join(passages).lower()
    passage_tokens = set(all_passage_text.split())
    overlap = answer_tokens & passage_tokens
    support = len(overlap) / max(len(answer_tokens), 1)
    return support >= threshold, round(support, 4)
