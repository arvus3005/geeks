"""One-off diagnostic: for every unanswerable example the target answered
anyway (false confidence), check whether the chunk that grounded the
answer belongs to a DIFFERENT example's query_id (pooled-index
cross-contamination -- an artifact of this suite's shared mini-index, not
a real target failure) or the SAME example's own candidates (a real
grounding-calibration issue: the target accepted a candidate MSMARCO
itself labeled is_selected=0 for this exact query).

Run with the eval-tool's own isolated venv, same --rag-root pattern as a
normal eval.runner invocation, but importing the pipeline pieces directly
instead of going through runner.py's full report so this can inspect
retrieved_en/retrieved_hi's query_id per example.
"""
import sys

sys.path.insert(0, ".")

from eval import target
from eval.dataset import load_examples
from eval.index_build import build_index
from eval.pipeline import run as run_pipeline

RAG_ROOT = "/Users/suvra/Documents/hackerhouse-goa-task-2"

target.load_target(RAG_ROOT)
examples = load_examples(num_answerable=50, num_unanswerable=50, seed=42)
index, records = build_index(examples)
results = run_pipeline(examples, index, records, top_k=5, workers=6)

same_query_count = 0
cross_query_count = 0
no_hits_count = 0
details = []

for r in results:
    if r.error is not None or r.example.is_answerable:
        continue
    if not r.answer_grounded:
        continue  # correct abstention, not a false confidence
    all_hits = list(r.retrieved_en) + list(r.retrieved_hi)
    if not all_hits:
        no_hits_count += 1
        continue
    top_hit = max(all_hits, key=lambda h: h.score)
    same = top_hit.query_id == r.example.query_id
    if same:
        same_query_count += 1
    else:
        cross_query_count += 1
    details.append(
        {
            "query": r.example.query_en,
            "same_query_id": same,
            "top_hit_query_id": top_hit.query_id,
            "own_query_id": r.example.query_id,
            "top_hit_is_selected": top_hit.is_selected,
            "answer": r.answer_text[:150],
        }
    )

total = same_query_count + cross_query_count + no_hits_count
print(f"\nTotal false-confidence cases analyzed: {total}")
print(f"  Cross-query contamination (different example's passage): {cross_query_count}")
print(f"  Same-query (own candidate MSMARCO marked is_selected=0):  {same_query_count}")
print(f"  No hits somehow:                                         {no_hits_count}")

print("\n--- SAME-QUERY cases (real calibration concern, if any) ---")
for d in details:
    if d["same_query_id"]:
        print(f"  Q: {d['query']!r}")
        print(f"     top_hit_is_selected={d['top_hit_is_selected']}  answer={d['answer']!r}")

print("\n--- Sample CROSS-QUERY cases (the known artifact, if any) ---")
for d in details[:8]:
    if not d["same_query_id"]:
        print(f"  Q: {d['query']!r}  (own_id={d['own_query_id']}, hit_from_id={d['top_hit_query_id']})")
        print(f"     answer={d['answer']!r}")
