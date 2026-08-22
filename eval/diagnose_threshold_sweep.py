"""Real-data risk-coverage sweep for MIN_RERANKER_SCORE (extractive.py).

diagnose_margin.py already showed margin-based rules don't help. This
sweeps the plain top1 threshold itself across every score actually observed
in the real 50+50 eval set (not just the current -2.0 anecdote borrowed
from a different team's corpus), and reports false_confidence_rate vs.
false_refusal_rate at each one -- a risk-coverage curve, so the current
MIN_RERANKER_SCORE can be picked by evidence instead of by percentile-
eyeballing. reliability.py's own docstring calls false confidence the
worse failure of the two, so the right operating point trades some of the
(currently tiny, 2%) false-refusal budget for a lower false-confidence
rate, not a 50/50 split.
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
    top1 = max(rerank_score(ex.query_en, c) for c in candidates)
    rows.append({"answerable": ex.is_answerable, "top1": top1, "query": ex.query_en})

print(f"Scored {len(rows)} examples\n")

thresholds = sorted({round(r["top1"], 2) for r in rows})


def rates(th):
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    fn = sum(1 for r in ans if r["top1"] < th)
    fp = sum(1 for r in unans if r["top1"] >= th)
    false_refusal = fn / max(len(ans), 1)
    false_confidence = fp / max(len(unans), 1)
    return false_refusal, false_confidence


print(f"{'threshold':>10} {'false_refusal':>14} {'false_confidence':>18} {'youden_J':>10}")
best_j, best_th = -1.0, None
for th in thresholds:
    fr, fc = rates(th)
    j = (1 - fr) + (1 - fc) - 1  # Youden's J = sensitivity + specificity - 1
    marker = ""
    if j > best_j:
        best_j, best_th = j, th
    print(f"{th:>10.2f} {fr:>14.3f} {fc:>18.3f} {j:>10.3f}")

print(f"\nCurrent production MIN_RERANKER_SCORE = -2.0")
fr0, fc0 = rates(-2.0)
print(f"  at -2.0: false_refusal={fr0:.3f}  false_confidence={fc0:.3f}")
print(f"\nBest Youden's J threshold: {best_th} (J={best_j:.3f})")
fr1, fc1 = rates(best_th)
print(f"  at {best_th}: false_refusal={fr1:.3f}  false_confidence={fc1:.3f}")

# Since false confidence is the explicitly worse failure (reliability.py's
# own docstring), also show the highest threshold that keeps false_refusal
# at or below double its current measured rate (0.04) -- a cost-weighted
# operating point biased toward cutting false confidence harder.
print("\nCost-weighted sweep (cap false_refusal <= 0.04, minimize false_confidence):")
capped = [(th, *rates(th)) for th in thresholds]
capped = [(th, fr, fc) for th, fr, fc in capped if fr <= 0.04]
if capped:
    th, fr, fc = min(capped, key=lambda t: t[2])
    print(f"  best: threshold={th}  false_refusal={fr:.3f}  false_confidence={fc:.3f}")
else:
    print("  no threshold keeps false_refusal <= 0.04")
