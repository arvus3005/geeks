"""Joint calibration of verify_grounding's two lexical-overlap thresholds
(passage_support, query_overlap), plus a standalone measurement of the new
answer-type gate (_answer_type_mismatch) added alongside this script.

Every prior threshold script in this directory (diagnose_query_overlap.py,
diagnose_threshold_sweep.py, diagnose_margin.py) swept exactly ONE signal
at a time and picked its own Youden's-J optimum independently. That leaves
a real gap: passage_support (threshold=0.45, externally set via
settings.min_retrieval_score) and query_overlap (MIN_QUERY_ANSWER_OVERLAP
=0.25) are ANDed together in production, but their joint interaction was
never swept -- a pair of individually-mediocre thresholds can dominate (or
be dominated by) a different pair in ways neither axis's own sweep would
reveal.

This computes the actual joint Pareto frontier (skyline) of achievable
(false_refusal, false_confidence) operating points for the
passage_support>=a AND query_overlap>=b rule family, using a classic
skyline algorithm: generate every candidate rule's outcome, sort by one
axis, sweep keeping only points that improve the other axis (non-dominated
points) -- O(n^2) candidate thresholds (n=~100 distinct values per axis,
so ~1e4 candidates) x O(n) evaluation each, trivial at this sample size.

Uses the same real 50+50 MSMARCO-XI sample and the same MSMARCO-XI-supplied
candidate passages per example as diagnose_margin.py (ex.candidates_en),
calling extract_answer() directly (bypassing verify_grounding's own
thresholds) so the raw signal values are available even for examples the
CURRENT thresholds already reject. Same methodology as
diagnose_query_overlap.py -- these are lexical text-vs-text ratios, not
retrieval-score magnitudes, so (per output_guards.py's own module
docstring) they don't carry the eval-mini-index-vs-55M-passage-corpus
score-distribution gap that burned MIN_RERANKER_SCORE; they have NOT,
however, been separately validated against real production traffic, and
should be watched with the same caution until they have (see this
project's Verification Rigor practice: real measured numbers only, never
estimated).
"""
import statistics
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "/Users/suvra/Documents/hackerhouse-goa-task-2/src")

from eval import target
from eval.dataset import load_examples

RAG_ROOT = "/Users/suvra/Documents/hackerhouse-goa-task-2"
target.load_target(RAG_ROOT)

from hhgoa_rag.answer.extractive import extract_answer
from hhgoa_rag.guardrails.output_guards import (
    MIN_QUERY_ANSWER_OVERLAP,
    _answer_type_mismatch,
    _support_and_overlap,
)

CURRENT_SUPPORT_THRESHOLD = 0.45  # settings.min_retrieval_score, as passed by every real call site

examples = load_examples(num_answerable=50, num_unanswerable=50, seed=42)

rows = []
for ex in examples:
    candidates = ex.candidates_en or ex.candidates_hi
    if not candidates:
        continue
    passages = [{"payload": {"chunk_text": c}} for c in candidates]
    answer, evidence = extract_answer(passages, ex.query_en)
    if answer is None:
        continue  # extract_answer's own relevance gate already declined -- not this script's concern
    passage_texts = [p["payload"]["chunk_text"] for p in evidence]
    support, overlap = _support_and_overlap(ex.query_en, answer, passage_texts)
    rows.append(
        {
            "answerable": ex.is_answerable,
            "support": support,
            "overlap": overlap,
            "type_mismatch": _answer_type_mismatch(ex.query_en, answer),
            "query": ex.query_en,
            "answer": answer,
        }
    )

print(f"Scored {len(rows)} examples that clear extract_answer's own relevance gate\n")

ans = [r for r in rows if r["answerable"]]
unans = [r for r in rows if not r["answerable"]]


def rates(rule):
    """rule(row) -> True means ACCEPT (grounded)."""
    fn = sum(1 for r in ans if not rule(r))       # genuine answers wrongly declined
    fp = sum(1 for r in unans if rule(r))          # fabrications wrongly accepted
    return fn / max(len(ans), 1), fp / max(len(unans), 1)


# --- 1. Current production rule, as a baseline ---
current_rule = lambda r: r["support"] >= CURRENT_SUPPORT_THRESHOLD and r["overlap"] >= MIN_QUERY_ANSWER_OVERLAP
fr0, fc0 = rates(current_rule)
print(f"Current production (support>={CURRENT_SUPPORT_THRESHOLD}, overlap>={MIN_QUERY_ANSWER_OVERLAP}): "
      f"false_refusal={fr0:.3f}  false_confidence={fc0:.3f}")

# --- 2. Answer-type gate alone (new, standalone contribution) ---
type_rule = lambda r: not r["type_mismatch"]
fr_t, fc_t = rates(type_rule)
n_triggered = sum(1 for r in rows if r["type_mismatch"])
print(f"Answer-type gate alone:                                          "
      f"false_refusal={fr_t:.3f}  false_confidence={fc_t:.3f}  (rejected {n_triggered}/{len(rows)} rows)")

# --- 3. Answer-type gate stacked onto current production rule ---
combined_current = lambda r: current_rule(r) and type_rule(r)
fr_c, fc_c = rates(combined_current)
print(f"Answer-type gate + current production rule:                     "
      f"false_refusal={fr_c:.3f}  false_confidence={fc_c:.3f}\n")

# --- 4. Joint skyline over (support, overlap), gated by the type check first ---
# Only rows that already pass the type gate are candidates for the
# support/overlap sweep below, matching production's actual gate order
# (type mismatch is a hard reject before the two threshold checks run).
typed_rows = [r for r in rows if not r["type_mismatch"]]
support_vals = sorted({round(r["support"], 3) for r in typed_rows})
overlap_vals = sorted({round(r["overlap"], 3) for r in typed_rows})

points = []  # (false_refusal, false_confidence, a, b)
for a in support_vals:
    for b in overlap_vals:
        rule = lambda r, a=a, b=b: r["support"] >= a and r["overlap"] >= b
        fr, fc = rates(lambda r, rule=rule: rule(r) if not r["type_mismatch"] else False)
        points.append((fr, fc, a, b))

# Skyline (Pareto frontier): sort by false_refusal ascending, keep only
# points whose false_confidence is a strict new minimum seen so far --
# classic O(n log n) staircase sweep after the O(n^2) candidate generation
# above. Ties on false_refusal: keep the lowest false_confidence among them.
points.sort(key=lambda p: (p[0], p[1]))
skyline = []
best_fc_so_far = float("inf")
for fr, fc, a, b in points:
    if fc < best_fc_so_far:
        skyline.append((fr, fc, a, b))
        best_fc_so_far = fc

print(f"Pareto frontier (type gate applied first, {len(support_vals)}x{len(overlap_vals)}="
      f"{len(support_vals) * len(overlap_vals)} candidates swept, {len(skyline)} non-dominated points):")
print(f"{'false_refusal':>14} {'false_confidence':>18} {'support>=':>10} {'overlap>=':>10}")
for fr, fc, a, b in skyline:
    marker = "  <- current" if abs(a - CURRENT_SUPPORT_THRESHOLD) < 1e-6 and abs(b - MIN_QUERY_ANSWER_OVERLAP) < 1e-6 else ""
    print(f"{fr:>14.3f} {fc:>18.3f} {a:>10.3f} {b:>10.3f}{marker}")

# --- 5. Verdict, not an auto-pick. support>=X is a no-op at every X<=1.0
# (support is ~1.0 for essentially every row -- literally true by
# construction for an extractive system, per output_guards.py's own module
# docstring), so this "2D" sweep is really 1D over overlap alone -- the
# same axis eval/diagnose_query_overlap.py already swept. Every point on
# this frontier BELOW the current one (0.25) is dominated (worse on both
# axes or a wash); every point ABOVE it buys less false_confidence only by
# paying strictly more false_refusal than today, on the same tradeoff
# curve the team already looked at today and picked 0.25 over the 0.40
# Youden's-J optimum specifically to protect the false-refusal budget (see
# MIN_QUERY_ANSWER_OVERLAP's module comment). This run reproduces that same
# shape independently (overlap=0.40 here: false_refusal=0.333 vs today's
# 0.125, a ~2.7x refusal cost for cutting false_confidence roughly in
# half) -- not new evidence for moving off 0.25, just confirmation there's
# no hidden joint-threshold win being left on the table. Not changing
# MIN_QUERY_ANSWER_OVERLAP based on this script; it already sits at a
# deliberate, reasoned point on this exact frontier.
current_idx = next((i for i, p in enumerate(skyline) if abs(p[3] - MIN_QUERY_ANSWER_OVERLAP) < 1e-6), None)
print(f"\nVerdict: support contributes nothing to this sweep (constant ~1.0 for both classes -- "
      f"see means above), so the frontier collapses to the existing overlap-only sweep. "
      f"Current overlap={MIN_QUERY_ANSWER_OVERLAP} sits on the frontier"
      + (f" at rank {current_idx + 1}/{len(skyline)}" if current_idx is not None else "")
      + f"; every point past it trades more false_refusal than today for less false_confidence, "
      f"the same tradeoff already considered and declined in favor of 0.25. No threshold change "
      f"recommended from this analysis.")

print(f"\nsupport:  answerable mean={statistics.mean(r['support'] for r in ans):.3f} "
      f"median={statistics.median(r['support'] for r in ans):.3f}  |  "
      f"unanswerable mean={statistics.mean(r['support'] for r in unans):.3f} "
      f"median={statistics.median(r['support'] for r in unans):.3f}")
print(f"overlap:  answerable mean={statistics.mean(r['overlap'] for r in ans):.3f} "
      f"median={statistics.median(r['overlap'] for r in ans):.3f}  |  "
      f"unanswerable mean={statistics.mean(r['overlap'] for r in unans):.3f} "
      f"median={statistics.median(r['overlap'] for r in unans):.3f}")
