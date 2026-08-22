"""Real-data test: does a margin-based signal (top1 reranker score minus
top2) separate genuinely-answerable from genuinely-unanswerable queries
better than the current absolute-threshold-only approach (MIN_RERANKER_SCORE)?

For each of the same 100 eval examples (seed=42, matching the real eval
run), score the query against EVERY one of its own candidate passages with
the real production reranker (no short-circuit here -- need all scores to
compute a margin), then check which single-number rule (top1 alone,
margin alone, or both combined) best separates the two classes.
"""
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "/Users/suvra/Documents/hackerhouse-goa-task-2/src")

from eval import target
from eval.dataset import load_examples

RAG_ROOT = "/Users/suvra/Documents/hackerhouse-goa-task-2"
target.load_target(RAG_ROOT)

from hhgoa_rag.answer.reranker import score as rerank_score

examples = load_examples(num_answerable=50, num_unanswerable=50, seed=42)

rows = []
for ex in examples:
    candidates = ex.candidates_en or ex.candidates_hi
    if not candidates:
        continue
    scores = sorted((rerank_score(ex.query_en, c) for c in candidates), reverse=True)
    top1 = scores[0]
    top2 = scores[1] if len(scores) > 1 else float("-inf")
    margin = top1 - top2
    rows.append({"answerable": ex.is_answerable, "top1": top1, "margin": margin})

print(f"Scored {len(rows)} examples\n")


def evaluate_rule(name, predicate):
    tp = sum(1 for r in rows if r["answerable"] and predicate(r))
    fn = sum(1 for r in rows if r["answerable"] and not predicate(r))
    fp = sum(1 for r in rows if not r["answerable"] and predicate(r))
    tn = sum(1 for r in rows if not r["answerable"] and not predicate(r))
    false_refusal = fn / max(tp + fn, 1)
    false_confidence = fp / max(fp + tn, 1)
    print(f"{name:40s} false_refusal={false_refusal:.3f}  false_confidence={false_confidence:.3f}  (tp={tp} fn={fn} fp={fp} tn={tn})")


# Current production rule
evaluate_rule("top1 >= -2.0 (current MIN_RERANKER_SCORE)", lambda r: r["top1"] >= -2.0)

# Sweep margin-only thresholds
for m in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    evaluate_rule(f"margin >= {m}", lambda r, m=m: r["margin"] >= m)

# Sweep combined rules
for m in [0.5, 1.0, 1.5, 2.0]:
    evaluate_rule(f"top1 >= -2.0 AND margin >= {m}", lambda r, m=m: r["top1"] >= -2.0 and r["margin"] >= m)

# Distribution summary
import statistics
ans_top1 = [r["top1"] for r in rows if r["answerable"]]
unans_top1 = [r["top1"] for r in rows if not r["answerable"]]
ans_margin = [r["margin"] for r in rows if r["answerable"]]
unans_margin = [r["margin"] for r in rows if not r["answerable"]]
print(f"\ntop1 score:  answerable mean={statistics.mean(ans_top1):.2f} median={statistics.median(ans_top1):.2f}  |  unanswerable mean={statistics.mean(unans_top1):.2f} median={statistics.median(unans_top1):.2f}")
print(f"margin:      answerable mean={statistics.mean(ans_margin):.2f} median={statistics.median(ans_margin):.2f}  |  unanswerable mean={statistics.mean(unans_margin):.2f} median={statistics.median(unans_margin):.2f}")
