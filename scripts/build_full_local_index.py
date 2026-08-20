"""Build the self-hosted BM25 + HNSW hybrid index directly from HuggingFace,
for the real full MSMARCO-XI corpus — not the 57k Pinecone-metadata subset
that scripts/build_local_hybrid_index.py produces.

Per CLAUDE.md, the full corpus is required; this script never truncates a
config unless --max-rows-per-config is explicitly passed, and any capped run
must be labeled smoke/pilot/experiment by the caller, not treated as final.

Phase 1 (this run): configs "hi" and "bn". Each MSMARCO-XI config carries
BOTH English_passages and Translated_passages, so streaming just these two
configs yields English + Hindi + Bengali coverage — no separate "en" config
exists. English is deduplicated once across configs via dedup_en so the same
English passage appearing under both "hi" and "bn" rows is stored once.
MEASURED (5k-row cross-language sample against hi/bn/gu/ta/mr): every
MSMARCO-XI language config shares the exact same underlying English passage
pool (100% overlap) -- this is one shared MS MARCO corpus (~9.9M unique
English passages, matching MS MARCO's real published ~8.8M passage count)
translated into 14 targets, not 14 independent corpora. A full 14-language
build would therefore be ~122M unique passages (~360-450GB with HNSW
overhead) -- does not fit this machine's 214GB free disk. hi+bn+en fits
comfortably on disk (~26M passages, ~75-100GB+, though see the SEGMENTING
section below for why disk was never actually the binding constraint).
See docs/FULL_INDEX_HANDOFF.md.

Reuses the exact leakage-safe parse/normalize/dedup/chunk pipeline from
hhgoa_rag.ingestion.engine (parse_record drops query/Answer/query_type/
is_selected before anything downstream ever sees them -- see
hhgoa_rag.dataset.parser.FORBIDDEN_FIELDS).

SEGMENTING -- READ THIS BEFORE CHANGING SEGMENT_SIZE:
An earlier version of this script built ONE growing HNSW index and ONE
growing BM25 token list for an entire (config, split) run, both held
entirely in RAM, only touching disk once at the very end. This was found
mid-run to be a real, imminent crash risk: this machine has 24GB total RAM
(`sysctl hw.memsize`), and live process RSS growth (measured: 1.9GB at
623,241 passages, tracking vector count almost exactly) projected to
~52-60GB by the ~25.9M-passage hi+bn+en target -- more than double
available RAM. Worse, because the HNSW index was never saved until
success, a crash partway through would have discarded EVERY embedding
computed so far while the checkpoint files still claimed those rows were
"done" -- a silent, unrecoverable data-loss bug, not just a performance
one. BM25 has the same shape of problem: `bm25s.BM25.index()` needs the
full tokenized corpus materialized as Python objects, and Python's
per-object overhead on millions of nested string lists is large relative
to the tokens' raw disk size.

The fix: output is split into SEGMENT_SIZE-passage segments, each with its
own directory, own HNSW index, and own BM25 index, finalized (written to
disk, memory released) as soon as it fills up -- not batched together
across the entire run. This bounds peak RAM to roughly one segment's
footprint regardless of how large the full corpus gets, and means a crash
loses at most one in-progress segment (a few minutes of work), not
everything. Segments are NOT automatically combined into one final index
-- see scripts/merge_local_indexes.py, and note its own docstring caveat
that a merge covering enough segments to approach this same RAM ceiling
has the identical BM25-materialization problem. For hi+bn+en at full
scale, a single final merged BM25 index may not fit in 24GB RAM either --
this is a genuinely open problem, not yet solved, and either needs a
bigger machine for the final merge or a federated/sharded serving
architecture that queries segments separately. Flagging honestly per
CLAUDE.md rather than pretending the segmenting fix solves everything.

EMBEDDING BACKEND: MPS (Apple GPU via torch/transformers), fp16, NOT the
ONNX int8 CPU model that src/hhgoa_rag/retrieval/local_embedder.py uses to
embed QUERIES in the live API. This was a deliberate speed choice (measured
~1435 texts/sec vs ~230 texts/sec for the fastest safe all-CPU-int8 config)
made after the user explicitly chose GPU speed over precision-matching.
IMPORTANT UNRESOLVED RISK: fp16-MPS passage vectors and int8-ONNX query
vectors are NOT guaranteed to be numerically/semantically compatible --
int8 quantization and fp16 both perturb the embedding space, in different
directions, from the fp32 original. This index must NOT be treated as
production-ready / swapped in for live serving until someone runs a
retrieval consistency check (embed a known query both ways, confirm top-k
results still make sense) -- see docs/FULL_INDEX_HANDOFF.md for the full
writeup and what check to run. Never claim this is "production ready"
without that check, per CLAUDE.md's Honesty section.

Why not stay on ONNX int8 (the safe option)? Measured throughput ceiling:
  - single-thread CPU, unsorted batches (original code): ~41.6 passages/sec
  - single-thread CPU, LENGTH-SORTED batches:            ~87.9 passages/sec (2.1x)
  - 8 intra-op threads (= 8 P-cores on this M4 Pro), sorted: ~230 texts/sec (peak;
    9-12 threads is WORSE -- spills onto slower E-cores, confirmed reproducible)
  - multiprocessing (any worker count, 6-12 tested): WORSE than single-process,
    ranged 11-28 texts/sec -- P/E-core oversubscription + IPC pickling overhead
    dominates for a model this small; do not reintroduce multiprocessing here.
  - CoreML EP (same int8 model): only 542/889 graph nodes offload to CoreML,
    rest falls back to CPU; measured slower than plain CPU (~62.6 texts/sec on
    a small batch) and crashed (SIGKILL) on a larger sorted-batch retry.
  - Native CoreML/ANE conversion (coremltools, fixed-shape): 694.7 texts/sec --
    a fair, dedicated test, still ~2x slower than MPS.
  - MPS fp16, batch=128, sorted, DEFERRED cpu() sync, inference_mode(): 1434.8
    texts/sec (measured, real corpus text, reproducible) -- the winner.

Usage:
    uv run python -m scripts.build_full_local_index --configs hi bn
    uv run python -m scripts.build_full_local_index --configs hi bn --max-rows-per-config 5000  # smoke only
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATASET_REPO = "ai4bharat/MSMARCO-XI"
SPLITS = ["train", "validation"]

# The dataset repo's custom loading script (ms_marco_translations.py) references
# train/{lang}train.jsonl / validation/{lang}val.jsonl paths that no longer exist,
# so `datasets.load_dataset(..., streaming=True)` cannot discover per-language
# configs (get_dataset_config_names returns only "default") and HF's auto parquet
# fallback throws ArrowNotImplementedError on these nested list columns when
# streamed. The real files are parquet, with 3-letter language prefixes:
#   train/{prefix}train.parquet, validation/{prefix}val.parquet
# We read them directly via pyarrow.parquet instead of `datasets`.
# NOTE: "te" (Telugu) has NO train split on this dataset revision (confirmed:
# 404 on train/teltrain.parquet) -- validation-only.
CONFIG_TO_PARQUET_PREFIX = {
    "as": "asm",
    "bn": "ben",
    "gu": "guj",
    "hi": "hin",
    "kn": "kan",
    "ml": "mal",
    "mr": "mar",
    "ne": "nep",
    "or": "ori",
    "pa": "pan",
    "sa": "san",
    "ta": "tam",
    "te": "tel",  # validation only -- no train split exists upstream
    "ur": "urd",
}


def _parquet_filename(config: str, split: str) -> str:
    prefix = CONFIG_TO_PARQUET_PREFIX[config]
    suffix = "train" if split == "train" else "val"
    return f"{split}/{prefix}{suffix}.parquet"


def _iter_parquet_rows(config: str, split: str, revision: str | None):
    """Stream rows of one MSMARCO-XI parquet shard as plain dicts.

    Downloads (or reuses the cached copy of) the parquet file via
    huggingface_hub, then reads it in row-group/record batches via pyarrow so
    the whole file is never materialized as a single in-memory table. This
    sidesteps both HF issues: (1) `datasets` cannot discover per-language
    configs because the repo's custom loading script points at files that no
    longer exist, and (2) HF's auto parquet-conversion streaming fallback
    throws ArrowNotImplementedError on these nested list<string> columns.
    """
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    filename = _parquet_filename(config, split)
    local_path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=filename,
        repo_type="dataset",
        revision=revision,
    )
    pf = pq.ParquetFile(local_path)

    def _rows():
        for batch in pf.iter_batches(batch_size=1024):
            for record in batch.to_pylist():
                yield record

    return _rows()


OUTPUT_DIR = Path("artifacts/full_local_index")
CHECKPOINT_DIR = Path("artifacts/full_index_checkpoints")
DEDUP_DIR = Path("artifacts/full_index_dedup")
LOG_EVERY_ROWS = 2000
EMBED_DIM = 384

# Accumulate this many passages before sorting-by-length + embedding as one
# GPU unit. Bigger pools -> better length-bucketing (less padding waste) but
# more RAM held at once mid-pool. 8192 is a middle ground.
POOL_SIZE = 8192
# GPU forward-pass batch size. Measured sweep on real corpus text (fp16,
# sorted, deferred sync, inference_mode): b128=1434.8 texts/sec (best of
# 64/128/256/512/1024 tried).
MPS_BATCH = 128
# Passages per finalized segment (own directory, own HNSW, own BM25). See the
# module docstring's SEGMENTING section for why this exists and how it was
# sized: measured ~2.0-2.3KB/passage for HNSW in RAM, conservatively assumed
# ~7-9KB/passage for BM25's temporary token-list materialization (not
# directly measured at scale -- treat as an estimate, not a verified number).
# 500,000 passages/segment -> an estimated ~5-6GB peak per segment, well
# under this machine's 24GB, with real headroom for the model/OS/other apps.
SEGMENT_SIZE = 500_000


def _checkpoint_path(config: str, split: str) -> Path:
    return CHECKPOINT_DIR / f"{config}_{split}.json"


def _load_checkpoint(config: str, split: str) -> tuple[int, int]:
    """Returns (next_row, next_segment_idx). Both only ever advance together,
    at the moment a segment is fully finalized (saved to disk) -- see the
    module docstring. A crash between finalizations replays from the last
    safely-finalized segment, discarding the in-progress one."""
    p = _checkpoint_path(config, split)
    if not p.exists():
        return 0, 0
    d = json.loads(p.read_text())
    return d["next_row"], d.get("next_segment_idx", 0)


def _save_checkpoint(config: str, split: str, next_row: int, next_segment_idx: int) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    p = _checkpoint_path(config, split)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({"next_row": next_row, "next_segment_idx": next_segment_idx}))
    tmp.rename(p)


def _mark_shard_done(config: str, split: str) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    p = _checkpoint_path(config, split)
    p.write_text(json.dumps({"next_row": -1, "next_segment_idx": -1, "status": "complete"}))


def _shard_done(config: str, split: str) -> bool:
    p = _checkpoint_path(config, split)
    if not p.exists():
        return False
    return json.loads(p.read_text()).get("status") == "complete"


def _segment_dir(config: str, split: str, segment_idx: int) -> Path:
    return OUTPUT_DIR / f"{config}_{split}_segment_{segment_idx:04d}"


def _load_mps_model():
    """Loads the ORIGINAL fp32 HF model (not the production ONNX int8 one),
    cast to fp16, on the MPS device. See module docstring for the precision-
    mismatch risk this introduces vs. the int8 query embedder."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS not available on this machine -- this script requires an Apple "
            "Silicon Mac with Metal support for the GPU embedding path."
        )
    tok = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
    model = AutoModel.from_pretrained("intfloat/multilingual-e5-small", dtype=torch.float16)
    model.eval()
    model = model.to("mps")
    return tok, model


def _mps_embed_and_tokenize(
    tok, model, sp, texts: list[str]
) -> tuple[list[list[float]], list[list[str]]]:
    """Embeds `texts` (passages, no prefix) via MPS fp16 and tokenizes them
    for BM25 via the same SentencePiece model the production embedder uses
    (so lexical search stays consistent with the rest of the pipeline).

    Two things matter for the measured ~1435 texts/sec:
      1. Sort by token length before batching. Padding every text in a batch
         to the batch's longest passage wastes GPU compute on real MSMARCO
         data (lengths range ~19-4700+ tokens) -- sorting first cut time by
         2.4x on its own.
      2. Defer `.cpu()` sync until ALL batches are queued, not once per
         batch. MPS ops are async; calling `.cpu()` inside the loop forces a
         round-trip sync every batch and stalls the GPU queue.
      3. `torch.inference_mode()` over `torch.no_grad()` -- lower autograd
         bookkeeping overhead, ~12-17% faster on top of the above. Batch
         sizes above 128 (256/512/1024) were all slower even at larger sort
         pools -- 128 is a real optimum, not an artifact of pool size.
         Confirmed zero CPU-fallback ops (PYTORCH_MPS_LOG_FALLBACK showed
         nothing) -- the whole model runs natively on MPS.
    """
    import torch

    lens = [len(tok.encode(t, truncation=True, max_length=512)) for t in texts]
    order = sorted(range(len(texts)), key=lambda i: lens[i])

    gpu_results: list[tuple[list[int], "torch.Tensor"]] = []
    bm25_tokens: list[list[str] | None] = [None] * len(texts)

    with torch.inference_mode():
        for start in range(0, len(order), MPS_BATCH):
            chunk_idx = order[start : start + MPS_BATCH]
            chunk_texts = [texts[i] for i in chunk_idx]
            enc = tok(
                [f"passage: {t}" for t in chunk_texts],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to("mps")
            out = model(**enc)
            last_hidden = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(last_hidden.dtype)
            pooled = (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            normed = torch.nn.functional.normalize(pooled, p=2, dim=1)
            gpu_results.append((chunk_idx, normed))  # NOT synced yet
            for j, oi in enumerate(chunk_idx):
                bm25_tokens[oi] = sp.encode(chunk_texts[j], out_type=str)

    embeddings: list[list[float] | None] = [None] * len(texts)
    for chunk_idx, tensor in gpu_results:
        arr = tensor.to(torch.float32).cpu().numpy()  # single sync point per pool
        for j, oi in enumerate(chunk_idx):
            embeddings[oi] = arr[j].tolist()

    assert all(e is not None for e in embeddings)
    assert all(t is not None for t in bm25_tokens)
    return embeddings, bm25_tokens  # type: ignore[return-value]


class _SegmentWriter:
    """Owns one segment's in-progress HNSW index + output files. Finalizing
    writes everything to disk, builds that segment's own BM25 index (bounded
    to this segment's token count, not the whole run's), and releases the
    in-memory HNSW/token state so the next segment starts with a clean RAM
    footprint."""

    def __init__(self, config: str, split: str, segment_idx: int):
        from usearch.index import Index

        self.config = config
        self.split = split
        self.segment_idx = segment_idx
        self.dir = _segment_dir(config, split, segment_idx)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.hnsw = Index(ndim=EMBED_DIM, metric="cos", dtype="f32")
        self.next_key = 0
        self.passages_f = open(self.dir / "passages.jsonl", "w")
        self.tokens_f = open(self.dir / "bm25_tokens.jsonl", "w")

    def add(self, embeddings, bm25_token_lists, texts, meta_list):
        import numpy as np

        n = len(texts)
        keys = np.arange(self.next_key, self.next_key + n)
        self.hnsw.add(keys, np.array(embeddings, dtype=np.float32))
        for i, (text, meta, pieces) in enumerate(zip(texts, meta_list, bm25_token_lists, strict=True)):
            key = self.next_key + i
            self.passages_f.write(json.dumps({"key": key, "text": text, "metadata": meta}) + "\n")
            self.tokens_f.write(json.dumps(pieces) + "\n")
        self.passages_f.flush()
        self.tokens_f.flush()
        self.next_key += n

    def finalize(self) -> None:
        import bm25s

        self.passages_f.close()
        self.tokens_f.close()
        self.hnsw.save(str(self.dir / "hnsw.usearch"))

        corpus_tokens = []
        with open(self.dir / "bm25_tokens.jsonl") as f:
            for line in f:
                corpus_tokens.append(json.loads(line))
        bm25 = bm25s.BM25()
        bm25.index(corpus_tokens, show_progress=False)
        bm25.save(str(self.dir / "bm25"))

        manifest = {
            "config": self.config,
            "split": self.split,
            "segment_idx": self.segment_idx,
            "n_passages": self.next_key,
            "embed_dim": EMBED_DIM,
            "embed_backend": (
                "mps_fp16_transformers (NOT the production int8 ONNX query "
                "embedder -- see build_full_local_index.py module docstring)"
            ),
            "bm25_backend": "bm25s",
            "hnsw_backend": "usearch",
            "status": "complete",
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (self.dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        logger.info("Finalized segment %s (%d passages)", self.dir, self.next_key)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["hi", "bn"])
    ap.add_argument("--dataset-revision", default=None)
    ap.add_argument(
        "--max-rows-per-config",
        type=int,
        default=None,
        help="Caps rows per (config, split). Leave unset for the real full corpus; "
        "if set, the caller MUST label the resulting artifact smoke/pilot/experiment per CLAUDE.md.",
    )
    args = ap.parse_args()

    from hhgoa_rag.dataset.parser import parse_record
    from hhgoa_rag.ingestion.chunkers import get_chunker
    from hhgoa_rag.ingestion.dedup import ContentDeduplicator
    import sentencepiece as spm
    from huggingface_hub import hf_hub_download

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEDUP_DIR.mkdir(parents=True, exist_ok=True)

    dedup_en = ContentDeduplicator(DEDUP_DIR / "dedup_en.sqlite")
    dedup_lang = ContentDeduplicator(DEDUP_DIR / "dedup_lang.sqlite")
    chunker = get_chunker("passage_native")

    logger.info("Loading MPS embedding model (fp16) + SentencePiece tokenizer…")
    tok, model = _load_mps_model()
    sp_model_path = hf_hub_download("intfloat/multilingual-e5-small", "sentencepiece.bpe.model")
    sp = spm.SentencePieceProcessor(model_file=sp_model_path)
    logger.info("Ready.")

    t_start = time.monotonic()
    # On a resumed run, count passages already sitting in finalized segments
    # from prior invocations -- otherwise this only tracks what THIS process
    # writes and undercounts the true total in progress/completion logs
    # (the on-disk data itself is unaffected either way; this is a reporting
    # fix, not a data-integrity one).
    grand_total_indexed = 0
    for manifest_path in OUTPUT_DIR.glob("*/manifest.json"):
        try:
            grand_total_indexed += json.loads(manifest_path.read_text()).get("n_passages", 0)
        except (json.JSONDecodeError, OSError):
            continue
    if grand_total_indexed:
        logger.info("Resuming: %d passages already in finalized segments on disk", grand_total_indexed)

    for config in args.configs:
        for split in SPLITS:
            if _shard_done(config, split):
                logger.info("Shard %s/%s already complete, skipping", config, split)
                continue

            start_row, segment_idx = _load_checkpoint(config, split)
            logger.info(
                "Streaming %s/%s from row %d (segment %d) …", config, split, start_row, segment_idx
            )

            try:
                ds = _iter_parquet_rows(config, split, args.dataset_revision)
            except Exception as e:
                logger.warning("Split %s/%s unavailable (%s), skipping", config, split, e)
                _mark_shard_done(config, split)
                continue

            segment = _SegmentWriter(config, split, segment_idx)
            row_idx = -1
            emitted_this_shard = 0
            pending_texts: list[str] = []
            pending_meta: list[dict] = []

            def flush_pool() -> None:
                nonlocal pending_texts, pending_meta
                if not pending_texts:
                    return
                embeddings, bm25_token_lists = _mps_embed_and_tokenize(tok, model, sp, pending_texts)
                segment.add(embeddings, bm25_token_lists, pending_texts, pending_meta)
                pending_texts, pending_meta = [], []

            for row_idx, record in enumerate(ds):
                if row_idx < start_row:
                    continue

                occurrences, _rejected = parse_record(
                    record,
                    config_language=config,
                    split=split,
                    source_shard=config,
                    source_row=row_idx,
                    dataset_revision=args.dataset_revision or "unknown",
                )

                for occ in occurrences:
                    dedup = dedup_en if occ.is_original_english else dedup_lang
                    if dedup.is_duplicate(occ.content_hash):
                        continue
                    dedup.mark_seen(occ.content_hash, occ.content_hash)

                    chunks = chunker.chunk(occ.normalized_text, occ.content_hash)
                    for chunk in chunks:
                        pending_texts.append(chunk.text)
                        pending_meta.append(
                            {
                                "language": occ.passage_language,
                                "config_language": occ.config_language,
                                "split": split,
                                "source_row": occ.source_row,
                                "passage_position": occ.passage_position,
                                "content_hash": occ.content_hash,
                                "chunk_ordinal": chunk.chunk_ordinal,
                            }
                        )

                if len(pending_texts) >= POOL_SIZE:
                    flush_pool()
                    dedup_en.flush()
                    dedup_lang.flush()

                    if segment.next_key >= SEGMENT_SIZE:
                        segment.finalize()
                        grand_total_indexed += segment.next_key
                        segment_idx += 1
                        _save_checkpoint(config, split, row_idx + 1, segment_idx)
                        segment = _SegmentWriter(config, split, segment_idx)

                emitted_this_shard += 1
                if emitted_this_shard % LOG_EVERY_ROWS == 0:
                    elapsed = time.monotonic() - t_start
                    total_so_far = grand_total_indexed + segment.next_key
                    rate = total_so_far / elapsed if elapsed > 0 else 0
                    logger.info(
                        "%s/%s: %d source rows, %d passages indexed so far (%.1fs elapsed, %.1f passages/sec, segment %d has %d)",
                        config,
                        split,
                        emitted_this_shard,
                        total_so_far,
                        elapsed,
                        rate,
                        segment_idx,
                        segment.next_key,
                    )

                if args.max_rows_per_config is not None and row_idx + 1 >= args.max_rows_per_config:
                    # Absolute row position, NOT relative to start_row -- a
                    # relative check would let a resumed capped run process
                    # extra rows beyond the original cap (found via testing:
                    # resuming from row 90 with max_rows=400 processed rows
                    # 90-489 instead of stopping at row 399). Only affects
                    # this smoke-test flag; the real uncapped run never
                    # takes this branch.
                    break

            if pending_texts:
                flush_pool()
                dedup_en.flush()
                dedup_lang.flush()

            if segment.next_key > 0:
                segment.finalize()
                grand_total_indexed += segment.next_key
                segment_idx += 1

            _save_checkpoint(config, split, row_idx + 1, segment_idx)
            _mark_shard_done(config, split)
            logger.info(
                "Completed %s/%s: %d source rows this run, %d segments, %d passages total indexed",
                config,
                split,
                emitted_this_shard,
                segment_idx,
                grand_total_indexed,
            )

    logger.info(
        "Done. %d passages across all segments. Segments are NOT auto-merged -- "
        "run scripts/merge_local_indexes.py to combine them (see its docstring "
        "for the RAM caveat on merging many/large segments).",
        grand_total_indexed,
    )


if __name__ == "__main__":
    main()
