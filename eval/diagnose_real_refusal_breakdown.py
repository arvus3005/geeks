"""Fast breakdown of WHERE the real-production false-refusal rate (measured
30% in diagnose_answer_relevance_real_production.py, vs 12.5% on the clean
eval-mini-index) actually comes from: extract_answer's own gates (lexical
prefilter MIN_QUERY_OVERLAP, or the reranker floor MIN_RERANKER_SCORE) vs
verify_grounding's gates (junk filters, answer-type mismatch, passage
support, query overlap). Needed to pick a surgical fix instead of guessing
which threshold to touch with the deadline this close.
"""
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "/Users/suvra/Documents/hackerhouse-goa-task-2/src")

from eval.dataset import load_examples

examples = load_examples(num_answerable=40, num_unanswerable=0, seed=42)

from app.retriever import search, warmup

warmup()

from hhgoa_rag.answer.extractive import MIN_QUERY_OVERLAP, MIN_RERANKER_SCORE, _content_tokens
from hhgoa_rag.answer.extractive import extract_answer
from hhgoa_rag.answer.reranker import score as rerank_score
from hhgoa_rag.guardrails.output_guards import (
    MIN_QUERY_ANSWER_OVERLAP,
    _answer_type_mismatch,
    _is_enumerator_stub,
    _is_navigation_junk,
    _is_truncated_fragment,
    _support_and_overlap,
)

reasons = {"no_hits": 0, "extract_lexical_prefilter": 0, "extract_reranker_floor": 0,
           "verify_junk": 0, "verify_type": 0, "verify_support": 0, "verify_overlap": 0, "accepted": 0}
details = []

for ex in examples:
    resp = search(ex.query_en, top_k=5)
    if not resp.hits:
        reasons["no_hits"] += 1
        continue
    texts = [h.get("fields", {}).get("chunk_text", "") or h.get("fields", {}).get("text", "") for h in resp.hits]
    query_tokens = _content_tokens(ex.query_en)

    passages = [{"payload": {"chunk_text": t}} for t in texts]
    answer, evidence = extract_answer(passages, ex.query_en)

    if answer is None:
        # Was it the lexical prefilter (no candidate cleared MIN_QUERY_OVERLAP)
        # or the reranker floor (candidates existed but none cleared
        # MIN_RERANKER_SCORE)? Recompute the same way extract_answer does.
        survivors = [
            t for t in texts[:3]
            if t and len(query_tokens & _content_tokens(t)) / max(len(query_tokens), 1) >= MIN_QUERY_OVERLAP
        ]
        if not survivors:
            reasons["extract_lexical_prefilter"] += 1
            details.append((ex.query_en, "extract_lexical_prefilter"))
        else:
            best = max(rerank_score(ex.query_en, t) for t in survivors)
            reasons["extract_reranker_floor"] += 1
            details.append((ex.query_en, f"extract_reranker_floor (best={best:.2f} < {MIN_RERANKER_SCORE})"))
        continue

    if _is_enumerator_stub(answer) or _is_truncated_fragment(answer) or _is_navigation_junk(answer):
        reasons["verify_junk"] += 1
        details.append((ex.query_en, "verify_junk"))
        continue
    if _answer_type_mismatch(ex.query_en, answer):
        reasons["verify_type"] += 1
        details.append((ex.query_en, "verify_type"))
        continue

    passage_texts = [p["payload"]["chunk_text"] for p in evidence]
    support, overlap = _support_and_overlap(ex.query_en, answer, passage_texts)
    if support < 0.45:
        reasons["verify_support"] += 1
        details.append((ex.query_en, f"verify_support ({support:.2f} < 0.45)"))
        continue
    if overlap < MIN_QUERY_ANSWER_OVERLAP:
        reasons["verify_overlap"] += 1
        details.append((ex.query_en, f"verify_overlap ({overlap:.2f} < {MIN_QUERY_ANSWER_OVERLAP})"))
        continue

    reasons["accepted"] += 1

total = len(examples)
print(f"Breakdown of {total} real answerable queries against REAL production retrieval:\n")
for k, v in reasons.items():
    print(f"  {k:>28}: {v:>3}/{total} = {v / total:.1%}")

print("\nDetails for each declined query:")
for q, reason in details:
    print(f"  [{reason}]  {q!r}")
