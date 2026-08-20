# Full Local Index — Handoff / Continuation Notes

**Purpose of this doc**: a self-contained account of the full-corpus local
indexing effort (`scripts/build_full_local_index.py`, branch
`feat/self-hosted-hybrid-retrieval`) — what was tried, what was measured,
what was decided and why, and what's left — so anyone (a fresh session, a
different model, a teammate) can continue without re-deriving any of this.
Written 2026-08-21. If something here conflicts with the current state of
the code or a running process, trust the code/process — this doc is a
snapshot, not a source of truth.

## Where this sits in the project

CLAUDE.md requires the full MSMARCO-XI corpus, not a sample, for anything
presented as final. The live Pinecone deployment
(`https://hhgoa-rag-d3fw.onrender.com`, index `msmarco-xi-e5small`) only has
~57k pilot passages and is **not** affected by anything in this doc — it
stays the safe, working submission fallback regardless of how the work
below turns out. This doc is about the separate, in-progress effort to
build a much bigger self-hosted local index (BM25 + HNSW, no Pinecone) per
the team's 2026-08-20 decision to go full-corpus and self-hosted. See
`README.md`'s "Indexing Status" section and the `feedback-verification-rigor`
memory for that decision's context.

## What "full corpus" actually means here — the shared-English discovery

The original ingestion pipeline (`src/hhgoa_rag/ingestion/engine.py`,
`src/hhgoa_rag/dataset/discovery.py`) assumed MSMARCO-XI exposes one
HuggingFace **config per language** (`hi`, `bn`, `gu`, ...), each an
independent corpus. That assumption is now known wrong in two ways:

1. **The dataset repo's own loading script is broken.** `ms_marco_translations.py`
   in `ai4bharat/MSMARCO-XI` points at `train/{lang}train.jsonl` /
   `validation/{lang}val.jsonl` paths that no longer exist. `datasets.get_dataset_config_names()`
   therefore returns only `["default"]`, and HuggingFace's auto parquet-conversion
   fallback throws `pyarrow.lib.ArrowNotImplementedError: Nested data conversions
   not implemented for chunked array outputs` when you try to stream it anyway.
   **Fix**: read the real parquet files directly via `huggingface_hub.hf_hub_download`
   + `pyarrow.parquet.ParquetFile.iter_batches()`, bypassing `datasets` entirely.
   The real files use 3-letter prefixes, mapped in `CONFIG_TO_PARQUET_PREFIX` in
   `build_full_local_index.py`: `hi->hin, bn->ben, gu->guj, ta->tam, mr->mar, te->tel`.
   Full file listing (`HfApi().list_repo_files`) showed:
   `train/{asm,ben,guj,hin,kan,mal,mar,nep,ori,pan,san,tam,urd}train.parquet` and
   `validation/{asm,ben,guj,hin,kan,mal,mar,nep,ori,pan,san,tam,tel,urd}val.parquet`.
   **Telugu ("te") has no train split at all** — confirmed via a live 404 on
   `train/teltrain.parquet`. Only Telugu validation exists. This is a real gap
   in the source dataset, not a bug in our code; any "full 14-language" claim
   needs to account for Telugu being validation-only.

2. **Every language config shares the exact same English passage pool.**
   Measured directly (`artifacts/logs/dedup_rate_check.log`,
   `artifacts/logs/cross_lang_dedup_check.log`): seeded a dedup table from
   `hi`'s first 5,000-20,000 rows, then streamed `bn`, `gu`, `ta`, `mr` —
   **100% of their English occurrences were exact duplicates** of what `hi`
   already produced (0 new English passages from any of them). This makes
   sense in retrospect: MSMARCO-XI is "MS MARCO translated to N languages"
   from one shared base English MS MARCO query/passage set, not N
   independent datasets. The unique-English count this converges to
   (~9.9M, extrapolated from a ~9.9-passages/row rate) lines up almost
   exactly with MS MARCO's real published ~8.8M passage corpus — a strong
   sanity check that this conclusion is right, not a measurement artifact.

**Implication for corpus sizing** (all extrapolated from real per-row
sampling, not the original README's linear-extrapolation-from-57k guess,
which is now known to be wrong — see below). Storage is a MEASURED
per-passage rate (6,601 bytes/passage: passages.jsonl + bm25_tokens.jsonl
+ hnsw.usearch + the compiled bm25/ directory + embeddings.npy, from a
real finalized 500,212-passage segment), not the earlier percentage-based
guess this table used to show -- that guess (~75-100GB / ~360-450GB) was
meaningfully too low and has been corrected here:

| Scope | Unique passages (extrapolated) | Storage (measured per-passage rate) |
|---|---|---|
| hi + bn + en (current target) | ~25.9M | ~131GB (~171GB if exporting embeddings.npy for every segment) |
| Full 14-language (13 with train+val, Telugu val-only) | ~122.5M | ~620GB (~809GB with embeddings.npy) |

The README's earlier "~24.87M vectors extrapolated" figure for the *full
14-language* corpus was based on linear-scaling the 57k Pinecone pilot's
row-to-vector ratio and is now superseded — it's actually in the right
ballpark for hi+bn+en *alone*, not the full 14 languages, because it didn't
account for the true ~9.9-passages/row yield or the shared-English
structure. **This README figure should be corrected** by whoever picks this
back up; it wasn't fixed in this session because the immediate priority was
getting the index build itself running.

## Why hi+bn (not a separate "en" config)

There is no standalone `en` config. Streaming `hi` and `bn` each yields both
`English_passages` and `Translated_passages` per MSMARCO-XI's schema (see
`src/hhgoa_rag/dataset/parser.py::parse_record`), so processing `hi`+`bn`
alone already produces English + Hindi + Bengali coverage. `dedup_en`
(a `ContentDeduplicator` keyed on `content_hash`) collapses English across
both configs so the shared passages aren't stored twice.

## Leakage boundary — unaffected, verified

`parse_record` only ever reads `passages.English_passages` and
`passages.Translated_passages` from each row; the forbidden fields
(`query`, `Answer`, `Eng_Query`, `Eng_Answer`, `query_type`, `is_selected` —
`FORBIDDEN_FIELDS` in `parser.py`) are never touched, regardless of where
they live in the parquet schema (`is_selected` in this schema actually
lives *inside* the `passages` struct, not top-level — doesn't matter, code
never reads it). Verified by grepping the smoke-test output for these
literal field names — zero hits. This logic is untouched by any of the
performance work below.

## Disk capacity: the real blocker for full 14-language

This machine (a MacBook Pro, Apple M4 Pro, 12 cores) has a single internal
disk, ~214-223GB free (fluctuates as segments write), no external volumes
mounted at time of writing. The user has an external SSD but hadn't
plugged it in yet as of this doc. **hi+bn+en (~131GB measured) fits, but
with less headroom than earlier estimates suggested -- roughly 80-90GB of
slack, not 115-140GB.** If embeddings.npy gets exported for every segment
(needed for merging), that adds up to ~171GB total, tightening the margin
further. Full 14-language (~620GB, ~809GB with embeddings.npy) does not
fit without the external SSD, and by a wider margin than previously
thought. This is a storage constraint, not a compute constraint — see the
timing numbers below, which show compute finishing
well within the 2026-08-22 deadline even for the full scope.

## Performance journey — what was tried, in order, with real numbers

All numbers below are measured on real MSMARCO-XI passage text pulled from
the actual `hi/train` parquet file, not synthetic data, unless stated. The
full trail exists in `artifacts/logs/*.log` from this session if anyone
wants to re-verify.

1. **Original single-thread ONNX int8 CPU** (`local_embedder.py`'s
   production embedder, `intra_op_num_threads=1`, unsorted stream order):
   **~41.6 passages/sec** (measured: 39,706 passages in 954.4s). At this
   rate, `hi/train` alone (778,638 rows) projected to ~103 hours — the
   original trigger for the whole optimization effort.

2. **Multiprocessing, ONNX int8, unsorted** (`ProcessPoolExecutor`, workers
   pinned to 1 ONNX thread each): tried 12, then 8, then 6 workers.
   **Got WORSE, not better, as more processes were added**: 12 workers →
   27.9 texts/sec, 8 workers → 11.3/sec, 6 workers → 10.7/sec — all worse
   than the single-process baseline. Root cause: this Mac (M4 Pro) has 8
   Performance cores + 4 Efficiency cores (`sysctl hw.perflevel0/1.logicalcpu`).
   Spawning more workers than P-cores pushes work onto much slower E-cores,
   and running many processes at full CPU concurrently likely also triggers
   thermal throttling on a laptop chassis (each successive test in the same
   session got monotonically worse regardless of the parameter being
   varied, consistent with cumulative heat build-up, not a worker-count
   effect specifically). **Conclusion: multiprocessing is net-negative for
   this model size on this hardware. Removed entirely from the final script.**

3. **Root-cause isolation**: profiled the main-process pipeline stages in
   isolation to find where time actually went. Parquet decode: 2.13ms/row.
   `parse_record`: 0.47ms/row. Full parse+dedup+chunk pipeline: 0.48ms/row.
   All fast — nowhere near the observed per-row cost. The bottleneck was
   confirmed to be purely the embedding step itself, isolated via a direct
   `ProcessPoolExecutor` timing: 1024 texts across 12 workers took 36.7s
   (27.9 texts/sec) — consistent with the P/E-core oversubscription theory.

4. **Length-sorted batching, single process, ONNX int8 CPU**: real MSMARCO
   passage lengths range from 19 to 4,759+ tokens (avg 94.3) — padding every
   text in a batch to the batch's longest passage wastes massive compute
   when batch order is arbitrary stream order. Sorting texts by token length
   before batching (so each batch is length-uniform) took the SAME
   single-thread config from **36.5 texts/sec (unsorted) to 87.9 texts/sec
   (sorted) — a 2.4x win from sorting alone**, reproduced twice
   (87.9, then 87.8 on a repeat).

5. **Intra-op CPU threading, WITH sorted batches** (single process, no
   multiprocessing): this scales far better than it did with unsorted data
   (which only got ~1.5x from 9 threads earlier). Sweep: 1 thread=87.9/sec,
   2=122.9, 4=171.6, **8=229.9 (peak, matches the 8 P-cores exactly)**,
   10=206.5 (worse), 12=187.3 (worse). Confirmed reproducible (repeat at 8
   threads: 216.4, consistent). **230 texts/sec is the CPU ceiling on this
   machine**, achieved via `HHGOA_ONNX_INTRA_THREADS=8` (an env var added to
   `local_embedder.py::_build()` — production serving path still defaults
   to 1 thread, unaffected).

6. **CoreML Execution Provider (via ONNX Runtime), same int8 model**: only
   542 of 889 graph nodes could be offloaded to CoreML (`GetCapability` log
   line), the rest falls back to CPU. Small-batch test: 62.6 texts/sec
   (worse than plain CPU's 165 texts/sec on the same short synthetic
   batch). A retry with a larger, length-sorted, real-data batch crashed
   with SIGKILL (exit 137) — not investigated further since the user had
   already redirected to MPS by that point and the partial-offload ceiling
   made it structurally unlikely to compete anyway.

7. **PyTorch MPS (Apple GPU), fp32, original naive loop**: **568.1
   texts/sec** — already 2.5x past the CPU ceiling. But this uses the
   *original* fp32 `transformers.AutoModel` weights, NOT the int8-quantized
   ONNX model (`Xenova/multilingual-e5-small`) that the production query
   embedder uses. **This is the source of the unresolved correctness risk
   — see below.**

8. **MPS optimization sweep**: tested fp16 vs fp32, batch sizes 64-1024,
   deferring `.cpu()` sync to the end of a pool instead of per-batch, and
   `torch.inference_mode()` vs `torch.no_grad()`.
   - fp16 > fp32 (Apple GPUs get real wins from half precision).
   - Deferred sync matters a lot — calling `.cpu()` inside the batch loop
     forces a synchronous round-trip every batch and stalls the async MPS
     queue; collecting all batch results as GPU tensors and only calling
     `.cpu()` once at the end lets the queue stay full.
   - batch=128 was the sweet spot across every pool size tried (4096 and
     8192); 256/512/1024 were all slower.
   - `inference_mode()` beat `no_grad()` by ~12-17% (lower autograd
     bookkeeping overhead).
   - Confirmed via `PYTORCH_MPS_LOG_FALLBACK=1` that **zero ops fall back to
     CPU** — the whole model runs natively on the GPU, no hidden bottleneck.
   - **Final result: 1434.8 texts/sec** (fp16, batch=128, 8192-item sort
     pool, deferred sync, inference_mode). Reproducible.

9. **Native CoreML/ANE conversion** (via `coremltools`, NOT going through
   ONNX Runtime's partial-offload path): traced the PyTorch model with
   `torch.jit.trace`, converted with `ct.convert(..., compute_units=ct.ComputeUnit.ALL)`,
   targeting a fixed shape (batch=64, seq_len=128) since CoreML performs
   best with static shapes. This was a genuine, fair, dedicated-conversion
   test, not the crippled ONNX-EP path. **Result: 694.7 texts/sec — still
   ~2x slower than MPS.** For a model this small, the ANE's per-call
   dispatch overhead plus CoreML's fixed-shape padding tax outweigh
   whatever efficiency advantage the ANE has. **MPS via PyTorch remains the
   best result found, by a wide margin, across every approach tried.**

### Summary table

| Approach | Texts/sec | vs. baseline |
|---|---|---|
| ONNX int8 CPU, 1 thread, unsorted (original) | 41.6 | 1x |
| ONNX int8 CPU, 1 thread, sorted | 87.9 | 2.1x |
| ONNX int8 CPU, 8 threads, sorted (CPU ceiling) | 230 | 5.5x |
| Multiprocessing, any config (6-12 workers) | 11-28 | **worse than 1x** |
| CoreML via ONNX Runtime EP | ~63, crashed on retry | worse |
| CoreML via native `coremltools` (ANE+GPU) | 694.7 | 16.7x |
| MPS fp32, naive | 568.1 | 13.7x |
| **MPS fp16, sorted, batch=128, deferred sync, inference_mode (FINAL)** | **1434.8** | **34.5x** |

Important caveat on that 1434.8 number: it's the embedding step measured
**in isolation**, on text that was already parsed and sitting in memory.
The real end-to-end run (parquet decode + parse_record + SQLite dedup +
chunking + file writes, all on CPU, sequential with the GPU step, not
overlapped) only sustained **~700-720 passages/sec** in practice. That gap
motivated the CPU/GPU overlap attempt below.

### CPU/GPU overlap: tried and reverted

Reasoned that splitting embedding into a non-blocking "submit" phase
(dispatch all of a pool's GPU batches, return without the `.cpu()` sync
that forces a wait) and a separate "drain" phase (the actual sync, called
one pool later) should let CPU parsing of pool N+1 overlap with GPU
embedding of pool N -- the same "submit before drain" pattern already
proven correct in the (removed) multiprocessing version, just adapted to
MPS's async queue instead of separate worker processes. Implemented it
(`_mps_submit()` / `_mps_drain()`, checkpoint logic adjusted for the
one-pool lag), verified it was still CORRECT via the same
interrupt-and-resume test used elsewhere in this doc (exact match: 15,917
passages, zero duplicates).

**But it measured slower, not faster, in production**: ~505-520
passages/sec sustained, vs. ~700-720 for the plain synchronous version.
Reverted immediately (`git checkout -- scripts/build_full_local_index.py`
before the change was ever committed) rather than debug a regression
under deadline pressure with an unclear root cause. Two live confounds
worth knowing about if anyone revisits this:

1. **The comparison itself may have been contaminated.** The overlap
   version was benchmarked in the same wall-clock window as the still-
   running production job (both fighting for the same GPU), so its number
   might be worse than the true isolated cost of the code path.
2. **After reverting back to the known-good synchronous code, the SAME
   ~520/sec regression persisted** -- i.e., the previously-fast simple
   version was ALSO running slow post-revert, strongly suggesting
   cumulative thermal throttling from ~45+ minutes of near-continuous
   heavy GPU/CPU load (many restarts, tests, and the overlap experiment
   itself, all back-to-back with no cooldown) was the real explanation
   for both "regressions" -- not the overlap code specifically. This was
   never conclusively separated from a genuine code regression; the user
   chose to let the run continue rather than spend more wall-clock time
   on a cooldown-and-retest cycle. **If revisiting the overlap idea**,
   test it after a real cooldown period (machine fully idle 15-20+
   minutes) and in isolation (nothing else touching the GPU), not
   back-to-back with other GPU work the way this session did it.

## THE UNRESOLVED RISK — read this before treating the output as production-ready

The final, fastest embedding path (#9 above, MPS fp16 via
`transformers.AutoModel`) is a **different model artifact** than the one
`src/hhgoa_rag/retrieval/local_embedder.py` uses to embed **queries** in
the live API (ONNX int8, `Xenova/multilingual-e5-small`). Both derive from
the same base weights (`intfloat/multilingual-e5-small`), but:

- int8 quantization perturbs weights one way.
- fp16 casting perturbs them a different way.
- Neither perturbation is verified to leave cosine similarity rankings
  intact relative to the fp32 original, and the two perturbations were
  never verified compatible *with each other*.

This project already had one real, silent correctness bug from exactly
this class of issue (the missing XLM-RoBERTa fairseq +1 offset — see
`local_embedder.py`'s module docstring and the 2026-08-19 fix in git log)
that produced no error, just semantically scrambled embeddings, caught only
by manually testing a live query. **Do not assume this is fine by
default.**

This was a **deliberate, explicit user decision** ("we need MPS/GPU no
other option... 568 or more we need"), made after the tradeoff was
explained, prioritizing speed over verified precision-consistency, under
real deadline pressure. That's a legitimate call for the user to make. What
hasn't happened yet is the follow-up verification. **Before this local
index is wired into anything serving real queries (i.e., before it
replaces or supplements the live Pinecone deployment), someone needs to
run a retrieval consistency check**:

1. Take a handful of real queries with known-good expected passages (the
   same kind of spot-check used to catch the fairseq offset bug: "what is a
   corporation?" should retrieve corporation-related passages, not
   something unrelated).
2. Embed the query via the production path (`local_embedder.embed_query`,
   int8 ONNX).
3. Search against this MPS-fp16-built HNSW index.
4. Confirm the top results are still semantically relevant, and ideally
   compare rank order against the SAME queries run through the existing
   57k Pinecone e5-small index (which used the exact matching int8 model on
   both sides) as a reference for "what good looks like."

If that check fails or shows meaningful degradation, the fix is either (a)
re-embed the corpus with the exact int8 ONNX model instead (slower, ~230
texts/sec ceiling, but zero mismatch risk), or (b) switch the *query*
embedder to match (fp16 MPS or fp32) — which reopens the memory-budget
problem that made int8 ONNX the production choice in the first place (see
`local_embedder.py`'s docstring: fp32/torch measured 1.5-2GB, didn't fit
the 512MB Render container).

## Current script architecture (`scripts/build_full_local_index.py`)

- Single process, no multiprocessing (proven net-negative on this hardware).
- Streams parquet directly via `huggingface_hub` + `pyarrow`, not `datasets`.
- Reuses the leakage-safe `parse_record` / `ContentDeduplicator` /
  `get_chunker("passage_native")` pipeline unchanged from the original
  Pinecone ingestion engine.
- Accumulates passages into pools of `POOL_SIZE=8192` before processing (a
  tunable tradeoff between length-sort quality and checkpoint granularity —
  not deeply tuned beyond this).
- `_mps_embed_and_tokenize()`: sorts the pool by token length, embeds in
  GPU batches of `MPS_BATCH=128` with deferred `.cpu()` sync, and
  BM25-tokenizes each text (via the same SentencePiece model used
  elsewhere in the pipeline) in the same pass.
- HNSW (`usearch`) is incremental — `.add()` per pool, saved at the end.
- BM25 (`bm25s`) is **not** incremental — the library requires the full
  tokenized corpus in memory at index-build time, so token lists are
  streamed to `bm25_tokens.jsonl` as they're produced and only loaded back
  + indexed in one shot at the very end of the whole run. **This has not
  yet been tested at the ~25.9M-passage scale** — it's a real unknown for
  time and memory, not yet measured. If this run gets to that step, watch
  for it.
- Resumable per `(config, split)` shard via
  `artifacts/full_index_checkpoints/*.json`, granularity = one pool.
- **`torch` is an optional dependency** (`pyproject.toml`'s
  `[project.optional-dependencies] gpu-index`), deliberately NOT in the
  base dependency list — the production API must never import torch (see
  `local_embedder.py`'s docstring on why: measured ~1.5-2GB footprint,
  doesn't fit the 512MB Render deployment). This script is offline-only.

## Timing

Measured live at the start of the current run (see "Current run state"
below): **~697-703 passages/sec sustained**, end-to-end (parse + dedup +
chunk + embed + tokenize + write) — noticeably below the isolated 1434.8
texts/sec embedding-only benchmark, because parse/dedup/chunk time is
**additive, not overlapped** with GPU embedding time in this single-process
design (a possible future optimization: overlap CPU-side parsing of the
NEXT pool with GPU embedding of the CURRENT pool via a background thread,
similar in spirit to the double-buffering approach tried and abandoned for
the multiprocessing version — not implemented here for lack of time before
the deadline).

At ~700 passages/sec sustained:
- hi+bn (~25.9M passages): **~10.3 hours** (embed+parse combined; BM25
  build time on top of this is still unmeasured at scale).
- Full 14-language (~122.5M passages, blocked on disk anyway): **~48.6
  hours** at this same rate — would need the external SSD AND likely some
  further optimization (e.g. the CPU/GPU overlap idea above) to be
  comfortable against a tight deadline.

## Current run state (as of writing this doc)

A real (uncapped) `hi`+`bn` run was launched in the background:
```
nohup uv run python -m scripts.build_full_local_index --configs hi bn > artifacts/logs/full_local_index_mps_hi_bn.log 2>&1 &
```
Check `artifacts/logs/full_local_index_mps_hi_bn.log` for progress, or
`artifacts/full_index_checkpoints/*.json` for exact resume position per
shard. It is resumable — if it dies or is killed, re-running the same
command picks up from the last completed pool per shard rather than
restarting. **Do not resume it using a DIFFERENT embedding backend than
what's already in the output** (i.e., don't mix ONNX-CPU-embedded and
MPS-embedded passages in the same `artifacts/full_local_index/` — if the
script or embedding approach changes again, clear `artifacts/full_local_index/`,
`artifacts/full_index_checkpoints/`, and `artifacts/full_index_dedup/` and
restart clean, the same way this run itself started clean after an earlier
CPU-based partial run was discarded for exactly this reason).

## Distributed indexing across multiple machines

Since compute (not disk) was the binding constraint for a single machine's
full-14-language scope (~48.6h estimated at ~700 passages/sec — see
Timing above), and since indexing is trivially parallel across languages
(each `--configs X` run is independent), the plan is to have contributors
run `build_full_local_index.py` on their own Apple Silicon Macs for a
subset of the remaining languages, then merge everyone's output centrally.
The full contributor-facing walkthrough is `docs/FRIEND_INDEXING_GUIDE.md`
— written to be handed to someone with zero context on this project.

Two new scripts support this:

- **`scripts/export_local_index_vectors.py`**: usearch's HNSW format
  doesn't support merging two independently-built indexes, and
  `passages.jsonl` never stored the embedding vectors themselves (only
  text + metadata) — so there was no portable way to combine shards
  without re-embedding everything from scratch. This script reads a
  shard's `hnsw.usearch`, exports all vectors as `embeddings.npy` (shape
  `[N, 384]`, row `i` = passage `key` `i`). **Important implementation
  detail found while writing this**: `Index.vectors` (the naive "give me
  everything" property) is NOT returned in key order — usearch stores
  vectors by internal HNSW graph position, not insertion/key order,
  confirmed empirically (shuffled-key test: `idx.vectors[i]` did not match
  `key=i`'s original vector). `Index.get(keys_array)` IS correctly ordered
  by whatever keys you pass it, and is vectorized/fast (1000-vector batch
  retrieval measured at <1ms) — use `get()`, not `.vectors`, for any
  key-ordered export.

- **`scripts/merge_local_indexes.py`**: takes N shard directories, dedupes
  by `(passage_language, content_hash)` across all of them (not
  `content_hash` alone — this keeps English's cross-shard dedup working
  correctly, since every shard's English is expected to be identical, per
  the shared-English-pool finding above, while not accidentally colliding
  translated passages from different languages, though real hash
  collisions between different-language text are effectively impossible
  anyway; the tuple key is just a free extra safety margin), rebuilds one
  combined HNSW index and one combined BM25 index. **Verified correct** by
  merging two identical copies of the 11,950-passage smoke-test shard:
  correctly deduped to exactly 11,950 (not 23,900), per-language counts
  matched (en=3984, hi=3982, bn=3984), both HNSW and BM25 rebuilt
  successfully.

**Not verified, deliberately, by the tooling**: that every contributor
actually used the identical embedding backend/precision. `merge_local_indexes.py`
prints each shard's `manifest.json` `embed_backend` field as a smell test,
but a match there is not proof — someone could still run a stale or
modified script version. **Whoever runs the merge should manually confirm**
every contributor was on the same commit of `build_full_local_index.py`
before trusting the merged output's retrieval quality (on top of the
already-flagged fp16-MPS vs int8-ONNX risk above, which applies identically
regardless of how many machines contributed).

**Language code → parquet filename mapping was extended** to cover all 14
`INDIC_LANGUAGE_CODES` (previously only `hi`, `bn`, `gu`, `ta`, `mr`, `te`
were mapped, from ad-hoc testing) — `CONFIG_TO_PARQUET_PREFIX` in
`build_full_local_index.py` now has every language so any contributor can
be assigned any code.

### Non-Apple-Silicon contributors: CPU and CUDA paths

The script originally hard-required MPS (`torch.backends.mps.is_available()`
or crash) since that was the only path built/tested. Two of the actual
contributors (Prasun, Souvik) turned out to be on Windows, GPU presence
unconfirmed at time of writing -- so the script now supports `--device
{auto,mps,cuda,cpu}` (default `auto`, tries MPS then CUDA then falls back
to CPU):

- **CPU path** (`_load_cpu_model()` / `_cpu_embed_and_tokenize()`): reuses
  `hhgoa_rag.retrieval.local_embedder` verbatim -- the exact ONNX int8
  model the production API already uses for query embedding, cross-platform
  (onnxruntime + sentencepiece run on Windows/Linux, no torch/CUDA needed),
  with the same length-sort-before-batching trick that helped on the MPS
  path applied here too (measured 2.1x on CPU specifically: unsorted
  41.6/sec -> sorted 87.9/sec, before thread-count tuning -- see the
  Performance journey section above for the full CPU benchmark trail up to
  the 230/sec ceiling found on this M4 Pro's 8 P-cores). Contributors on
  this path don't need `uv sync --extra gpu-index` at all -- base
  dependencies are enough. **Interesting side effect**: CPU-path shards are
  actually MORE consistent with the live production query embedder (exact
  same model) than this machine's own MPS fp16 shards are, inverting the
  usual "GPU path is better" assumption for this specific project.
  `manifest.json`'s `embed_backend` field distinguishes `cpu_int8_onnx`
  from `mps_fp16_transformers`/`cuda_fp16_transformers`, so
  `merge_local_indexes.py`'s existing smell-test can at least flag someone
  merging mismatched backends even though it can't block it outright.

- **CUDA path**: same code as MPS (`_load_gpu_model(device)` /
  `_gpu_embed_and_tokenize(..., device)` now take the device string instead
  of hardcoding `"mps"`), just targeting `cuda` instead. The fp16 /
  sorted-batching / `inference_mode()` / deferred-sync approach was tuned
  and measured on MPS specifically -- untested on any actual NVIDIA GPU as
  of this writing. `MPS_BATCH=128` is a reasonable starting point, not a
  verified-optimal one for CUDA; a contributor with an NVIDIA GPU could
  likely get more by sweeping batch size once real hardware is available
  to test on.

Both paths were verified via real smoke tests (isolated output directories,
not the live production run): correct device auto-detection, correct
`embed_backend` label in `manifest.json`, segments finalize correctly,
clean leakage boundary. Neither was benchmarked for real throughput on
non-Apple-Silicon hardware -- there wasn't a Windows/CUDA machine available
to test on during this session, so the "how long will this take" numbers in
`docs/FRIEND_INDEXING_GUIDE.md` are deliberately vague for those paths.

## Immediate next steps for whoever continues this

1. Let the current hi+bn run finish (or check on it — `tail -f
   artifacts/logs/full_local_index_mps_hi_bn.log`), watching in particular
   for how long the final BM25 build step takes (unmeasured at this scale).
2. Run the retrieval consistency check described above before treating the
   output as anything more than "implemented, not quality-verified."
3. Correct the README's "~24.87M vectors extrapolated" full-corpus figure
   — it's now known to be roughly right for hi+bn+en alone and a
   significant undercount (by ~5x) for the true full 14-language scope.
4. If pursuing full 14-language: get the external SSD mounted, verify its
   free space against the corrected ~620GB (~809GB with embeddings.npy)
   estimate -- not the earlier ~360-450GB guess, which was based on a
   percentage-of-embedding-size approximation rather than a measured
   per-passage rate. A CPU/GPU pipeline-overlap optimization (parsing pool
   N+1 while GPU embeds pool N) was attempted and REVERTED after live
   testing showed it was slower, not faster, than the simple synchronous
   version (~505-520 texts/sec vs ~700-720) -- see "CPU/GPU overlap: tried
   and reverted" below before attempting this again. Given the current
   ~48.6h single-machine full-14-language estimate used ~700 passages/sec,
   and the run in progress plateaued lower (~520/sec, likely thermal, not
   yet confirmed to recover) -- treat that hour figure as optimistic, not
   a floor. Distributing across contributors'
   machines (see "Distributed indexing" above), but still worth knowing
   about if that falls through.
5. As contributors' shards come back (per `docs/FRIEND_INDEXING_GUIDE.md`),
   run `scripts/merge_local_indexes.py --shards <dir1> <dir2> ... --output-dir
   artifacts/merged_local_index` to combine them. Confirm every shard's
   `manifest.json` reports the same `embed_backend` before trusting the
   result (see "Distributed indexing" above for why this matters and isn't
   automatically enforced).
6. Wire a config flag to actually let the live API query this local
   hybrid index instead of (or alongside) Pinecone — not done yet, per the
   existing README's "Self-Hosted Hybrid Retrieval" section, which predates
   all of the work in this doc.
