"""Validates MIN_RERANKER_SCORE against the REAL production retrieval path
(app/retriever.py -> hhgoa_rag.retrieval.sharded_local_hybrid_store, the
actual sharded index used by /v1/query), not eval/index_build.py's isolated
mini-index (~2391 chunks built only from 100 examples' own candidate
pools).

Why this exists: extractive.py's own history already documents that a
reranker-score threshold calibrated against a clean/small evaluation set
can look fine there and still over-decline sharply against the real, much
noisier production score distribution (MIN_RERANKER_SCORE=0.0 was
calibrated exactly this way once, then measured to decline ~80% of real
traffic). The 2026-08-22 recalibration to 0.4 was validated only against
eval/index_build.py's isolated index -- this script checks it against the
real thing before trusting it.

Uses real MSMARCO-XI answerable queries (their gold answer exists) as a
realistic query sample, run through the REAL retriever -- not against
their own dataset-provided candidate pool.
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

from hhgoa_rag.answer.reranker import score as rerank_score

rows = []
for i, ex in enumerate(examples):
    resp = search(ex.query_en, top_k=5)
    if not resp.hits:
        rows.append({"query": ex.query_en, "top1": float("-inf"), "n_hits": 0})
        continue
    texts = [h.get("fields", {}).get("chunk_text", "") or h.get("fields", {}).get("text", "") for h in resp.hits]
    scores = [rerank_score(ex.query_en, t) for t in texts if t]
    top1 = max(scores) if scores else float("-inf")
    rows.append({"query": ex.query_en, "top1": top1, "n_hits": len(resp.hits)})
    if (i + 1) % 10 == 0:
        print(f"  {i + 1}/{len(examples)} queries scored against REAL production retrieval")

import statistics

valid = [r["top1"] for r in rows if r["top1"] != float("-inf")]
print(f"\nScored {len(rows)} real queries against REAL production retrieval ({len(valid)} got real hits)")
if valid:
    print(f"top1 real-production score: mean={statistics.mean(valid):.2f} median={statistics.median(valid):.2f} "
          f"p10={sorted(valid)[len(valid)//10]:.2f} p90={sorted(valid)[9*len(valid)//10]:.2f}")

for th in [-2.0, 0.0, 0.4]:
    declined = sum(1 for r in rows if r["top1"] < th)
    print(f"threshold={th:>5.1f}  would decline {declined}/{len(rows)} = {declined / len(rows):.1%} of these real ANSWERABLE queries")

print("\nWorst-scoring real answerable queries (candidates for false refusal at threshold=0.4):")
for r in sorted(rows, key=lambda r: r["top1"])[:10]:
    print(f"  top1={r['top1']:>6.2f}  Q: {r['query']!r}")
