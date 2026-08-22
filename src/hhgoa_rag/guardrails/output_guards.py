def verify_grounding(
    answer: str, passages: list[str], threshold: float = 0.45
) -> tuple[bool, float]:
    """Check that answer text is supported by passages."""
    if not answer or not passages:
        return False, 0.0
    clean_ans = answer.strip().rstrip(".?!।॥").lower()
    all_passage_text = " ".join(passages).lower()
    if clean_ans and clean_ans in all_passage_text:
        return True, 1.0
    answer_tokens = set(clean_ans.split())
    if not answer_tokens:
        return False, 0.0
    passage_tokens = set(all_passage_text.split())
    overlap = answer_tokens & passage_tokens
    support = len(overlap) / len(answer_tokens)
    return support >= threshold, round(support, 4)
