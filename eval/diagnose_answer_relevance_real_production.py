"""Validates the candidate answer_relevance signal (eval/diagnose_answer_relevance.py)
against the REAL production retrieval path (app/retriever.py ->
hhgoa_rag.retrieval.sharded_local_hybrid_store), not eval/index_build.py's
isolated ~2391-chunk mini-index built only from the 100 sampled examples'
own candidate pools.

Same reason eval/diagnose_real_production_threshold.py exists: a signal
calibrated only against a clean/small eval sample can look great there and
still over-decline sharply against the real, much noisier production
retrieval distribution -- this project's own MIN_RERANKER_SCORE history is
the standing example (0.4 measured fine on the eval mini-index, then
measured to decline 75% of real traffic once checked here). Analysis only
-- no production thresholds changed by this script.

Runs real MSMARCO-XI ANSWERABLE queries through the ACTUAL retriever, then
the real extract_answer() + the candidate answer_relevance signal, and
reports what the false-refusal side of each candidate operating point
(Conservative/Moderate/Aggressive from the earlier eval-mini-index sweep)
would really cost against production retrieval.
"""
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "/Users/suvra/Documents/hackerhouse-goa-task-2/src")

from eval.dataset import load_examples

examples = load_examples(num_answerable=40, num_unanswerable=0, seed=42)
print(f"Loaded {len(examples)} real answerable queries.\n")

print("Warming up REAL production retriever (embedder + sharded index)...")
from app.retriever import search, warmup

warmup()
print("Warm.\n")

from hhgoa_rag.answer.extractive import extract_answer
from hhgoa_rag.answer.reranker import score as rerank_score
from hhgoa_rag.guardrails.output_guards import _answer_type_mismatch, _support_and_overlap

CANDIDATE_THRESHOLDS = {"Conservative": -1.70, "Moderate": -1.00, "Aggressive": -0.40}

rows = []
for i, ex in enumerate(examples):
    resp = search(ex.query_en, top_k=5)
    if not resp.hits:
        rows.append({"query": ex.query_en, "declined_by_extract": True, "answer_relevance": None})
        continue
    passages = [
        {"payload": {"chunk_text": h.get("fields", {}).get("chunk_text", "") or h.get("fields", {}).get("text", "")}}
        for h in resp.hits
    ]
    answer, evidence = extract_answer(passages, ex.query_en)
    if answer is None:
        rows.append({"query": ex.query_en, "declined_by_extract": True, "answer_relevance": None})
        continue
    passage_texts = [p["payload"]["chunk_text"] for p in evidence]
    support, overlap = _support_and_overlap(ex.query_en, answer, passage_texts)
    type_mismatch = _answer_type_mismatch(ex.query_en, answer)
    currently_grounded = (not type_mismatch) and support >= 0.45 and overlap >= 0.25
    answer_relevance = rerank_score(ex.query_en, answer) if currently_grounded else None
    rows.append(
        {
            "query": ex.query_en,
            "declined_by_extract": False,
            "currently_grounded": currently_grounded,
            "answer_relevance": answer_relevance,
            "answer": answer,
        }
    )
    if (i + 1) % 10 == 0:
        print(f"  {i + 1}/{len(examples)} real answerable queries processed against REAL production retrieval")

declined_by_extract = sum(1 for r in rows if r["declined_by_extract"])
currently_grounded = sum(1 for r in rows if not r["declined_by_extract"] and r.get("currently_grounded"))
print(f"\nOf {len(rows)} real answerable queries against the REAL production retriever:")
print(f"  declined before answer_relevance even applies (extract_answer or existing gates): "
      f"{len(rows) - currently_grounded}/{len(rows)} = {(len(rows) - currently_grounded) / len(rows):.1%}")
print(f"  reach the answer_relevance gate (currently grounded):                            "
      f"{currently_grounded}/{len(rows)} = {currently_grounded / len(rows):.1%}\n")

scored = [r for r in rows if r.get("answer_relevance") is not None]
if scored:
    import statistics

    vals = [r["answer_relevance"] for r in scored]
    print(f"answer_relevance on REAL production retrieval (n={len(vals)}): "
          f"mean={statistics.mean(vals):.2f} median={statistics.median(vals):.2f} "
          f"p10={sorted(vals)[len(vals) // 10]:.2f} p90={sorted(vals)[9 * len(vals) // 10]:.2f}\n")

for name, th in CANDIDATE_THRESHOLDS.items():
    additional_declines = sum(1 for r in scored if r["answer_relevance"] < th)
    total_declines = (len(rows) - currently_grounded) + additional_declines
    print(f"{name:>13} (answer_relevance>={th:>5}): would ADD {additional_declines}/{len(scored)} declines "
          f"on top of today's gates -- total decline rate on these real answerable queries: "
          f"{total_declines}/{len(rows)} = {total_declines / len(rows):.1%}")

print("\nWorst-scoring real-production answer_relevance among genuinely answerable queries "
      "(false-refusal candidates if the threshold were tightened):")
for r in sorted(scored, key=lambda r: r["answer_relevance"])[:10]:
    print(f"  relevance={r['answer_relevance']:>6.2f}  Q: {r['query']!r}")
    print(f"                A: {r['answer'][:120]!r}")
