# Post-Indexing Steps

**What this doc is:** a forward-looking checklist for what happens *after*
every language shard is indexed — merging, verifying, and switching the
live app over to the self-hosted index. It does not contain project
history (see the root `README.md` for that); it only contains what's left
to do, in order, so anyone can pick this up without guessing.

**Who this is for:** whoever is running the indexing effort once shards
start coming back from teammates, or a single person finishing the last
few languages themselves.

---

## 1. Confirm every shard actually finished correctly

Before merging anything, check each shard folder for real completion —
don't assume a shard is done just because someone said so.

| Check | How | Pass condition |
|---|---|---|
| Every segment finalized | Look for `manifest.json` inside each `*_segment_NNNN/` folder | No segment folder is missing `manifest.json` |
| Correct embedding backend | Read `manifest.json`'s `embed_backend` field in every segment | All segments across all shards report the **same** backend (e.g. all `mps_fp16_transformers`, or all `cpu_int8_onnx`) — a mix is a real problem, see Step 3 |
| No forbidden fields leaked | `grep` shard's `passages.jsonl` for `query`, `Answer`, `Eng_Query`, `Eng_Answer`, `query_type`, `is_selected` | Zero matches |
| Language counts look right | Compare `manifest.json` passage counts against what the contributor was assigned | Roughly matches (large discrepancies mean something was interrupted and never resumed) |

## 2. Merge all shards into one index

```bash
uv run python -m scripts.merge_local_indexes \
    --shards <shard-dir-1> <shard-dir-2> ... \
    --output-dir artifacts/merged_local_index
```

- Deduplicates by `(language, content_hash)` across every shard — this
  matters because every language shares the same underlying English
  passages, so most shards will contain identical English content that
  should collapse to one copy, not be stored N times.
- Rebuilds one combined HNSW (dense) index and one combined BM25
  (keyword) index from the deduplicated set.
- **Sanity-check the output**: total passage count after merging should
  be noticeably less than the sum of all shards' individual counts
  (because of the English overlap above) — if it's exactly equal to the
  sum, deduplication silently didn't work and needs investigating before
  going further.

## 3. Resolve the embedding-consistency risk before trusting the merged index

Different machines may have used different embedding paths (GPU vs CPU,
different precision). This has **not been verified safe** to mix, and
this project has already had one real, silent, no-error bug caused by
exactly this class of mismatch — assume it matters until proven
otherwise.

Run a retrieval consistency check:

1. Pick a handful of real queries with an obviously-correct expected
   answer (e.g. a query about a specific topic that should retrieve
   passages clearly about that topic, not something unrelated).
2. Embed each query using the same model/precision the live app will use
   to embed queries.
3. Search the merged index and read the top results.
4. Confirm the results are actually relevant — not just "some result came
   back," but "this is obviously the right passage for this question."

**If this check fails or looks shaky**, don't proceed to Step 4 yet — the
fix is either re-embedding the mismatched shards with a consistent
model/precision, or matching the query-side embedder to whatever the
index was built with (which may reopen a memory/footprint tradeoff worth
re-checking, not just assumed to be fine).

## 4. Benchmark the merged index for real

Spot-checking a few queries (Step 3) proves *correctness*, not *quality*.
Before calling this production-ready, run a systematic evaluation — the
kind of measured, real numbers this project has consistently insisted on
elsewhere, not an estimate:

- Latency: P50 / P70 / P95 / P100 over a real query set, end-to-end
  (embed + search + fuse), matching the methodology already used for the
  currently-deployed system.
- Retrieval quality: a proper metric (e.g. MRR@k or Recall@k) against a
  held-out query set with known-correct passages, not just spot checks.
- Memory and disk footprint of the loaded index at serving time, measured
  directly, not estimated from file size alone.

## 5. Wire the local hybrid index into the live serving API

This is the step that actually makes any of the above matter to a real
user. Concretely:

1. Add a way to select the retrieval backend at startup (a setting/flag),
   defaulting to whatever is currently safest, so this can be tested
   without breaking the existing working deployment.
2. Load the merged HNSW + BM25 index once at process startup (per the
   project's own latency rules — never load a heavy resource per-request).
3. Replace (or run alongside, for comparison) the current search step
   with: embed query → search HNSW → search BM25 → fuse with Reciprocal
   Rank Fusion → continue into the existing grounding/guardrail pipeline
   unchanged.
4. Re-run the full latency benchmark against this new path before
   switching it on for real traffic, the same way the current deployment's
   numbers were verified live, not assumed from a local test.

## 6. After switching over, update the story

Once the switch in Step 5 is live and verified:

- Update the root `README.md`'s status tables to reflect that the
  self-hosted index is now what's actually serving traffic (not just
  "built, not yet wired in").
- Record the real, measured latency/quality numbers from Step 4 — not
  projected or estimated ones.
- Keep whatever the previous serving setup was available as a fallback
  until the new path has run stable for a reasonable stretch of real
  traffic.

## Known open risks going into this (don't skip re-checking these)

- **BM25 build time/memory at full scale is unmeasured.** The BM25
  library used here needs the entire tokenized corpus in memory at
  build time, unlike the incremental dense index. This has only been
  tested at much smaller scale — watch memory closely the first time this
  step runs against the full merged corpus, and be ready for it to need
  more memory than expected.
- **Not every contributor's machine has been verified identical in
  software version.** `manifest.json`'s backend label is a smell test,
  not proof — someone could be on a modified or stale script version.
  Confirm shard contributors were on the same commit before fully
  trusting merged quality.
- **Telugu has no training data in the source dataset at all** (only
  validation) — this is a real gap in the data itself, not something
  missed during indexing. Don't treat a small Telugu shard as a bug.
