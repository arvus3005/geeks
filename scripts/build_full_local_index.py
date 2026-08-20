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
comfortably (~26M passages, ~75-100GB). See docs/FULL_INDEX_HANDOFF.md.

Reuses the exact leakage-safe parse/normalize/dedup/chunk pipeline from
hhgoa_rag.ingestion.engine (parse_record drops query/Answer/query_type/
is_selected before anything downstream ever sees them -- see
hhgoa_rag.dataset.parser.FORBIDDEN_FIELDS).

Resumable: progress is checkpointed per (config, split) to
artifacts/full_index_checkpoints/*.json, granularity = one POOL_SIZE pool.

EMBEDDING BACKEND: MPS (Apple GPU via torch/transformers), fp16, NOT the
ONNX int8 CPU model that src/hhgoa_rag/retrieval/local_embedder.py uses to
embed QUERIES in the live API. This was a deliberate speed choice (measured
~1227 texts/sec vs ~230 texts/sec for the fastest safe all-CPU-int8 config,
~5.3x) made after the user explicitly chose GPU speed over precision-
matching. IMPORTANT UNRESOLVED RISK: fp16-MPS passage vectors and int8-ONNX
query vectors are NOT guaranteed to be numerically/semantically compatible
-- int8 quantization and fp16 both perturb the embedding space, in
different directions, from the fp32 original. This index must NOT be
treated as production-ready / swapped in for live serving until someone
runs a retrieval consistency check (embed a known query both ways, confirm
top-k results still make sense) -- see docs/FULL_INDEX_HANDOFF.md for the
full writeup and what check to run. Never claim this is "production ready"
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
  - MPS fp16, batch=128, sorted, DEFERRED cpu() sync (see _mps_embed_and_tokenize
    docstring for why deferring sync matters): ~1227 texts/sec (measured, real
    corpus text, reproducible).

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
# 404 on train/teltrain.parquet) -- validation-only. Not in scope for phase 1.
CONFIG_TO_PARQUET_PREFIX = {
    "hi": "hin",
    "bn": "ben",
    "gu": "guj",
    "ta": "tam",
    "mr": "mar",
    "te": "tel",
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
# unit. Bigger pools -> better length-bucketing (less padding waste) but
# coarser checkpoint/resume granularity and more RAM held at once. 8192 is a
# middle ground; not deeply tuned, "take more storage/CPU" per user request.
POOL_SIZE = 8192
# GPU forward-pass batch size. Measured sweep on real corpus text (fp16,
# sorted, deferred sync): b64=954, b128=1227 (best), b256=1104 texts/sec.
MPS_BATCH = 128


def _checkpoint_path(config: str, split: str) -> Path:
    return CHECKPOINT_DIR / f"{config}_{split}.json"


def _load_checkpoint(config: str, split: str) -> int:
    p = _checkpoint_path(config, split)
    if not p.exists():
        return 0
    return json.loads(p.read_text())["next_row"]


def _save_checkpoint(config: str, split: str, next_row: int) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    p = _checkpoint_path(config, split)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({"next_row": next_row}))
    tmp.rename(p)


def _mark_shard_done(config: str, split: str) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    p = _checkpoint_path(config, split)
    p.write_text(json.dumps({"next_row": -1, "status": "complete"}))


def _shard_done(config: str, split: str) -> bool:
    p = _checkpoint_path(config, split)
    if not p.exists():
        return False
    return json.loads(p.read_text()).get("status") == "complete"


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

    Two things matter for the measured ~1227 texts/sec:
      1. Sort by token length before batching. Padding every text in a batch
         to the batch's longest passage wastes GPU compute on real MSMARCO
         data (lengths range ~19-4700+ tokens) -- sorting first cut time by
         2.4x on its own (measured: 36.5s -> 23.35s for the same 2048 texts).
      2. Defer `.cpu()` sync until ALL batches are queued, not once per
         batch. MPS ops are async; calling `.cpu()` inside the loop forces a
         round-trip sync every batch and stalls the GPU queue. Deferring
         sync to the end let the queue stay full (measured: 4.47s -> matches
         the batch=64 fp32 numbers; the b128/fp16 combo hit 3.34s for the
         same 4096 texts, i.e. ~1227 texts/sec).
      3. `torch.inference_mode()` over `torch.no_grad()` -- lower autograd
         bookkeeping overhead. Measured on real corpus text, 8192-item sort
         pool, batch=128: no_grad=1279 texts/sec, inference_mode=1435
         texts/sec (~12% more, on top of the above). Batch sizes above 128
         (256/512/1024) were all slower even at this larger pool size --
         128 is a real optimum here, not an artifact of a small sort pool.
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
    from usearch.index import Index
    import numpy as np

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

    passages_path = OUTPUT_DIR / "passages.jsonl"
    hnsw_path = OUTPUT_DIR / "hnsw.usearch"
    tokens_path = OUTPUT_DIR / "bm25_tokens.jsonl"  # one JSON array of token strings per line

    hnsw = Index(ndim=EMBED_DIM, metric="cos", dtype="f32")
    if hnsw_path.exists():
        hnsw.load(str(hnsw_path))
        next_key = len(hnsw)
        logger.info("Resuming HNSW index with %d existing vectors", next_key)
    else:
        next_key = 0

    total_indexed = next_key
    t_start = time.monotonic()

    passages_f = open(passages_path, "a")
    tokens_f = open(tokens_path, "a")

    def flush_pool(pending_texts: list[str], pending_meta: list[dict]) -> None:
        nonlocal next_key, total_indexed
        if not pending_texts:
            return
        embeddings, bm25_token_lists = _mps_embed_and_tokenize(tok, model, sp, pending_texts)
        n = len(pending_texts)
        keys = np.arange(next_key, next_key + n)
        hnsw.add(keys, np.array(embeddings, dtype=np.float32))
        for i, (text, meta, pieces) in enumerate(
            zip(pending_texts, pending_meta, bm25_token_lists, strict=True)
        ):
            key = next_key + i
            passages_f.write(json.dumps({"key": key, "text": text, "metadata": meta}) + "\n")
            tokens_f.write(json.dumps(pieces) + "\n")
        passages_f.flush()
        tokens_f.flush()
        next_key += n
        total_indexed += n

    try:
        for config in args.configs:
            for split in SPLITS:
                if _shard_done(config, split):
                    logger.info("Shard %s/%s already complete, skipping", config, split)
                    continue

                start_row = _load_checkpoint(config, split)
                logger.info("Streaming %s/%s from row %d …", config, split, start_row)

                try:
                    ds = _iter_parquet_rows(config, split, args.dataset_revision)
                except Exception as e:
                    logger.warning("Split %s/%s unavailable (%s), skipping", config, split, e)
                    _mark_shard_done(config, split)
                    continue

                row_idx = -1
                emitted_this_shard = 0
                pending_texts: list[str] = []
                pending_meta: list[dict] = []

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
                        flush_pool(pending_texts, pending_meta)
                        pending_texts, pending_meta = [], []
                        dedup_en.flush()
                        dedup_lang.flush()
                        _save_checkpoint(config, split, row_idx + 1)

                    emitted_this_shard += 1
                    if emitted_this_shard % LOG_EVERY_ROWS == 0:
                        elapsed = time.monotonic() - t_start
                        rate = total_indexed / elapsed if elapsed > 0 else 0
                        logger.info(
                            "%s/%s: %d source rows, %d passages indexed so far (%.1fs elapsed, %.1f passages/sec)",
                            config,
                            split,
                            emitted_this_shard,
                            total_indexed,
                            elapsed,
                            rate,
                        )

                    if args.max_rows_per_config is not None and (
                        row_idx - start_row + 1
                    ) >= args.max_rows_per_config:
                        break

                if pending_texts:
                    flush_pool(pending_texts, pending_meta)
                    dedup_en.flush()
                    dedup_lang.flush()
                    _save_checkpoint(config, split, row_idx + 1)

                _mark_shard_done(config, split)
                logger.info(
                    "Completed %s/%s: %d source rows this run, %d passages total indexed",
                    config,
                    split,
                    emitted_this_shard,
                    total_indexed,
                )

        hnsw.save(str(hnsw_path))
        logger.info("Saved HNSW index (%d vectors) to %s", len(hnsw), hnsw_path)

    finally:
        passages_f.close()
        tokens_f.close()

    # ── Final BM25 build (requires full in-memory token corpus — see docstring) ──
    logger.info("Loading full token corpus for BM25 build …")
    import bm25s

    corpus_tokens = []
    with open(tokens_path) as f:
        for line in f:
            corpus_tokens.append(json.loads(line))
    logger.info("Loaded %d token sequences, building BM25 index …", len(corpus_tokens))
    t0 = time.monotonic()
    bm25 = bm25s.BM25()
    bm25.index(corpus_tokens)
    bm25.save(str(OUTPUT_DIR / "bm25"))
    logger.info("BM25 index built + saved in %.1fs", time.monotonic() - t0)

    manifest = {
        "source": f"{DATASET_REPO} (direct HF stream, not Pinecone metadata)",
        "configs": args.configs,
        "n_passages": total_indexed,
        "embed_dim": EMBED_DIM,
        "embed_backend": "mps_fp16_transformers (NOT the production int8 ONNX query embedder -- see module docstring for the unresolved precision-consistency risk)",
        "bm25_backend": "bm25s",
        "hnsw_backend": "usearch",
        "tokenization": "sentencepiece_pieces",
        "max_rows_per_config": args.max_rows_per_config,
        "label": "smoke" if args.max_rows_per_config is not None else "full_phase1_hi_bn",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("Done: %s", manifest)


if __name__ == "__main__":
    main()
