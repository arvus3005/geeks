"""Real-data check: does the fraction of the QUERY's own content words found
in the ANSWER text separate genuine answers (grounded=True on an answerable
example) from fabrications (grounded=True on an unanswerable example)?

verify_grounding() currently only checks answer-vs-passage lexical overlap,
which is close to tautological for an extractive system (the answer is a
substring of the passage it came from, so support is ~1.0 almost always --
see this project's real measured false_confidence_rate=0.88 before today's
extractive.py fixes). It never looks at the query at all. This script
builds the missing signal from real data before picking a threshold for it,
same method as diagnose_threshold_sweep.py.
"""
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "/Users/suvra/Documents/hackerhouse-goa-task-2/src")

from eval import target
from eval.dataset import load_examples
from eval.index_build import build_index
from eval.pipeline import run as run_pipeline

RAG_ROOT = "/Users/suvra/Documents/hackerhouse-goa-task-2"
target.load_target(RAG_ROOT)

from hhgoa_rag.answer.extractive import _content_tokens

examples = load_examples(num_answerable=50, num_unanswerable=50, seed=42)
index, records = build_index(examples)
results = run_pipeline(examples, index, records, top_k=5, workers=6)

rows = []
for r in results:
    if r.error is not None or not r.answer_grounded:
        continue  # only examples the system currently answers confidently
    query_tokens = _content_tokens(r.example.query_en)
    if not query_tokens:
        continue
    answer_tokens = _content_tokens(r.answer_text)
    overlap = len(query_tokens & answer_tokens) / len(query_tokens)
    rows.append({"answerable": r.example.is_answerable, "overlap": overlap, "query": r.example.query_en, "answer": r.answer_text[:100]})

genuine = [r for r in rows if r["answerable"]]
fabricated = [r for r in rows if not r["answerable"]]
print(f"grounded=True and answerable (presumed genuine): {len(genuine)}")
print(f"grounded=True and unanswerable (fabrications):   {len(fabricated)}\n")

import statistics
if genuine:
    print(f"genuine overlap:     mean={statistics.mean(r['overlap'] for r in genuine):.3f}  median={statistics.median(r['overlap'] for r in genuine):.3f}")
if fabricated:
    print(f"fabricated overlap:  mean={statistics.mean(r['overlap'] for r in fabricated):.3f}  median={statistics.median(r['overlap'] for r in fabricated):.3f}")

thresholds = sorted({round(r["overlap"], 2) for r in rows})
print(f"\n{'threshold':>10} {'reject_genuine':>15} {'reject_fabricated':>18} {'youden_J':>10}")
best_j, best_th = -1.0, None
for th in thresholds:
    fn = sum(1 for r in genuine if r["overlap"] < th)   # genuine answers wrongly rejected
    tp = sum(1 for r in fabricated if r["overlap"] < th)  # fabrications correctly rejected
    reject_genuine_rate = fn / max(len(genuine), 1)
    reject_fabricated_rate = tp / max(len(fabricated), 1)
    j = (1 - reject_genuine_rate) + reject_fabricated_rate - 1
    if j > best_j:
        best_j, best_th = j, th
    print(f"{th:>10.2f} {reject_genuine_rate:>15.3f} {reject_fabricated_rate:>18.3f} {j:>10.3f}")

print(f"\nBest Youden's J threshold: {best_th} (J={best_j:.3f})")
print("\nFabricated examples NOT caught even at best threshold (overlap >= best_th):")
for r in fabricated:
    if r["overlap"] >= (best_th or 1.0):
        print(f"  Q: {r['query']!r}  overlap={r['overlap']:.2f}  A: {r['answer']!r}")
