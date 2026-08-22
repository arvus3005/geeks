"""Does IDF-weighting the query<->answer overlap signal (verify_grounding's
query_overlap) separate genuine answers from fabrications better than the
current uniform (every content word counts equally) version?

Motivation from real remaining false-confidence examples (2026-08-22, after
the uniform query_overlap fix already shipped): several are a single rare/
specific query word swapped for a wrong one, buried among many correctly-
shared GENERIC words -- e.g. "...armed forces reserve army ribbon" answered
by a passage about an "Armed Forces Reserve Medal (AFRM)": overlap on
"armed"/"forces"/"reserve"/"authorized"/"wear" is high (~0.57), the ONE word
that actually matters ("ribbon") is silently outvoted. Uniform overlap
can't distinguish a rare, informative word from a generic one repeated
across many similar documents; IDF weighting can, if the IDF table itself
is real.

Methodology, deliberately different from diagnose_query_overlap.py's
mistake-that-almost-shipped (MIN_RERANKER_SCORE=0.4, calibrated and
measured only against the same 100-example set it was then shipped against,
and only caught by a SEPARATE real-production check): calibrate on seed=42
(same set used throughout this session), then VALIDATE on a held-out seed
that has never informed any threshold choice, before touching production
code at all.

IDF table itself is built from ~20,000 real MSMARCO-XI validation rows
(up to ~200,000 real English passages, document-frequency counted) --
independent of both eval seeds, and representative of the actual corpus's
own vocabulary (the production corpus IS this dataset), not a small
isolated sample.
"""
import math
import statistics
import sys
from collections import Counter

sys.path.insert(0, ".")
sys.path.insert(0, "/Users/suvra/Documents/hackerhouse-goa-task-2/src")

from eval import target
from eval.dataset import load_examples
from eval.index_build import build_index
from eval.msmarco import download_split, iter_rows
from eval.pipeline import run as run_pipeline

RAG_ROOT = "/Users/suvra/Documents/hackerhouse-goa-task-2"
target.load_target(RAG_ROOT)

from hhgoa_rag.answer.extractive import _content_tokens

print("Building real IDF table from MSMARCO-XI validation passages...")
path = download_split("hin", "validation")
df: Counter[str] = Counter()
n_docs = 0
seen_texts: set[str] = set()
for row in iter_rows(path, limit=20_000):
    for text in (row.get("passages") or {}).get("English_passages") or []:
        if not text or text in seen_texts:
            continue
        seen_texts.add(text)
        n_docs += 1
        for tok in _content_tokens(text):
            df[tok] += 1
print(f"IDF table built from {n_docs} unique real passages, {len(df)} distinct content words.\n")


def idf(word: str) -> float:
    return math.log(n_docs / (1 + df.get(word, 0)))


def weighted_overlap(query: str, answer: str) -> float:
    q_tokens = _content_tokens(query)
    if not q_tokens:
        return 1.0
    a_tokens = _content_tokens(answer)
    total = sum(idf(w) for w in q_tokens)
    if total <= 0:
        return 1.0
    shared = sum(idf(w) for w in q_tokens if w in a_tokens)
    return shared / total


def uniform_overlap(query: str, answer: str) -> float:
    q_tokens = _content_tokens(query)
    if not q_tokens:
        return 1.0
    a_tokens = _content_tokens(answer)
    return len(q_tokens & a_tokens) / len(q_tokens)


def measure(seed: int, label: str):
    examples = load_examples(num_answerable=50, num_unanswerable=50, seed=seed)
    index, records = build_index(examples)
    results = run_pipeline(examples, index, records, top_k=5, workers=6)

    rows = []
    for r in results:
        if r.error is not None or not r.answer_grounded:
            continue
        rows.append({
            "answerable": r.example.is_answerable,
            "uniform": uniform_overlap(r.example.query_en, r.answer_text),
            "weighted": weighted_overlap(r.example.query_en, r.answer_text),
        })

    genuine = [r for r in rows if r["answerable"]]
    fabricated = [r for r in rows if not r["answerable"]]
    print(f"=== seed={seed} ({label}) === grounded genuine={len(genuine)} grounded fabricated={len(fabricated)}")

    for key in ("uniform", "weighted"):
        g = [r[key] for r in genuine]
        f = [r[key] for r in fabricated]
        if g:
            print(f"  {key:9s} genuine:    mean={statistics.mean(g):.3f} median={statistics.median(g):.3f}")
        if f:
            print(f"  {key:9s} fabricated: mean={statistics.mean(f):.3f} median={statistics.median(f):.3f}")

        thresholds = sorted({round(r[key], 2) for r in rows})
        best_j, best_th, best_stats = -1.0, None, None
        for th in thresholds:
            fn = sum(1 for r in genuine if r[key] < th)
            tp = sum(1 for r in fabricated if r[key] < th)
            reject_genuine = fn / max(len(genuine), 1)
            reject_fabricated = tp / max(len(fabricated), 1)
            j = (1 - reject_genuine) + reject_fabricated - 1
            if j > best_j:
                best_j, best_th, best_stats = j, th, (reject_genuine, reject_fabricated)
        rg, rf = best_stats if best_stats else (None, None)
        print(f"  {key:9s} best J={best_j:.3f} at threshold={best_th}  (reject_genuine={rg:.3f}, reject_fabricated={rf:.3f})")
    print()
    return rows


train_rows = measure(42, "calibration seed, already used throughout this session")
holdout_rows = measure(123, "HELD-OUT seed, never used to pick anything")
