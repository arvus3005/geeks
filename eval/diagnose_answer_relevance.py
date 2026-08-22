"""Calibration for a genuinely new signal: cross-encoder score of
(query, ANSWER) -- not (query, passage), which extract_answer's own
MIN_RERANKER_SCORE gate already checks, and not (answer, passage), which
would likely be near-tautological the same way passage_support already is
(see diagnose_skyline.py -- the answer is usually a literal substring of
its source passage).

Motivation: passage_support and query_overlap both measure LEXICAL token
overlap; neither can tell "the passage is about the right entity" from
"the extracted sentence actually addresses the specific question" (see
tests/unit/test_output_guards.py's
test_known_limitation_same_entity_wrong_specific_fact_still_passes --
"erie insurance corporate address" answered with unrelated Erie Insurance
prose passes both existing lexical gates). A cross-encoder run on
(query, answer) directly -- with real cross-attention between the two
texts, not a bag-of-words ratio -- is a semantically different signal,
and this project already has one loaded: hhgoa_rag.answer.reranker.score,
~5-6ms/call on CPU (see reranker.py's own module docstring), same
multilingual (XLM-R) coverage already trusted for retrieval. No new
model, no new dependency, no new startup cost -- reusing the singleton
already warmed at process start.

Same methodology as diagnose_margin.py: MSMARCO-XI's own candidate list
per example (ex.candidates_en), extract_answer() called directly so raw
signal values are available even for examples the current gates already
reject.
"""
import statistics
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "/Users/suvra/Documents/hackerhouse-goa-task-2/src")

from eval import target
from eval.dataset import load_examples

RAG_ROOT = "/Users/suvra/Documents/hackerhouse-goa-task-2"
target.load_target(RAG_ROOT)

from hhgoa_rag.answer.extractive import extract_answer
from hhgoa_rag.answer.reranker import score as rerank_score
from hhgoa_rag.guardrails.output_guards import _answer_type_mismatch, _support_and_overlap

examples = load_examples(num_answerable=50, num_unanswerable=50, seed=42)

rows = []
call_latencies_ms = []
for ex in examples:
    candidates = ex.candidates_en or ex.candidates_hi
    if not candidates:
        continue
    passages = [{"payload": {"chunk_text": c}} for c in candidates]
    answer, evidence = extract_answer(passages, ex.query_en)
    if answer is None:
        continue
    passage_texts = [p["payload"]["chunk_text"] for p in evidence]
    support, overlap = _support_and_overlap(ex.query_en, answer, passage_texts)
    type_mismatch = _answer_type_mismatch(ex.query_en, answer)

    t0 = time.perf_counter()
    answer_relevance = rerank_score(ex.query_en, answer)
    call_latencies_ms.append((time.perf_counter() - t0) * 1000)

    rows.append(
        {
            "answerable": ex.is_answerable,
            "support": support,
            "overlap": overlap,
            "type_mismatch": type_mismatch,
            "answer_relevance": answer_relevance,
            "currently_grounded": (not type_mismatch) and support >= 0.45 and overlap >= 0.25,
            "query": ex.query_en,
            "answer": answer,
        }
    )

print(f"Scored {len(rows)} examples\n")
print(f"answer_relevance call latency: mean={statistics.mean(call_latencies_ms):.2f}ms "
      f"p50={statistics.median(call_latencies_ms):.2f}ms p90={sorted(call_latencies_ms)[int(len(call_latencies_ms)*0.9)]:.2f}ms\n")

ans = [r for r in rows if r["answerable"]]
unans = [r for r in rows if not r["answerable"]]
print(f"answer_relevance:  answerable mean={statistics.mean(r['answer_relevance'] for r in ans):.3f} "
      f"median={statistics.median(r['answer_relevance'] for r in ans):.3f}  |  "
      f"unanswerable mean={statistics.mean(r['answer_relevance'] for r in unans):.3f} "
      f"median={statistics.median(r['answer_relevance'] for r in unans):.3f}\n")


def rates(rule):
    fn = sum(1 for r in ans if not rule(r))
    fp = sum(1 for r in unans if rule(r))
    return fn / max(len(ans), 1), fp / max(len(unans), 1)


thresholds = sorted({round(r["answer_relevance"], 2) for r in rows})
print(f"{'threshold':>10} {'false_refusal':>14} {'false_confidence':>18} {'youden_J':>10}")
best_j, best_th = -1.0, None
for th in thresholds:
    fr, fc = rates(lambda r, th=th: r["answer_relevance"] >= th)
    j = (1 - fr) + (1 - fc) - 1
    if j > best_j:
        best_j, best_th = j, th
    print(f"{th:>10.2f} {fr:>14.3f} {fc:>18.3f} {j:>10.3f}")
print(f"\nBest Youden's J threshold (answer_relevance alone): {best_th} (J={best_j:.3f})")

# --- Stacked onto everything currently shipping (type gate + support/overlap) ---
print("\n=== Stacked onto current production gates (type + support>=0.45 + overlap>=0.25) ===")
fr0, fc0 = rates(lambda r: r["currently_grounded"])
print(f"Current gates alone:                          false_refusal={fr0:.3f}  false_confidence={fc0:.3f}")
stacked_points = []
for th in [round(-2.5 + 0.1 * i, 2) for i in range(41)]:
    rule = lambda r, th=th: r["currently_grounded"] and r["answer_relevance"] >= th
    fr, fc = rates(rule)
    stacked_points.append((fr, fc, th))
    print(f"+ answer_relevance >= {th:>6.2f}:              false_refusal={fr:.3f}  false_confidence={fc:.3f}")

stacked_points.sort(key=lambda p: (p[0], p[1]))
skyline = []
best_fc = float("inf")
for fr, fc, th in stacked_points:
    if fc < best_fc:
        skyline.append((fr, fc, th))
        best_fc = fc
print(f"\nPareto frontier of stacked rule (non-dominated points):")
print(f"{'false_refusal':>14} {'false_confidence':>18} {'answer_relevance>=':>20}")
for fr, fc, th in skyline:
    print(f"{fr:>14.3f} {fc:>18.3f} {th:>20.2f}")

# --- Does it catch the documented known-limitation case and other
# currently-uncaught fabrications specifically? ---
print("\nCurrently-grounded UNANSWERABLE rows (i.e. current false_confidence cases) and their answer_relevance:")
for r in unans:
    if r["currently_grounded"]:
        print(f"  relevance={r['answer_relevance']:>6.2f}  Q: {r['query']!r}")
        print(f"                A: {r['answer'][:120]!r}")
