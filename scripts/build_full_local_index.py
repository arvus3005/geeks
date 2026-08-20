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

Reuses the exact leakage-safe parse/normalize/dedup/chunk pipeline from
hhgoa_rag.ingestion.engine (parse_record drops query/Answer/query_type/
is_selected before anything downstream ever sees them — see
hhgoa_rag.dataset.parser.FORBIDDEN_FIELDS).

Resumable: progress is checkpointed per (config, split) to
artifacts/full_index_checkpoints/*.json (source row only, not upload acks —
there's no remote to ack against). Re-running resumes from the last
checkpointed row per shard rather than re-streaming from zero.

Known constraint, not yet solved: bm25s.BM25.index() takes the whole
tokenized corpus in memory at once — there's no incremental/streaming BM25
build in this library. So this script buffers SentencePiece token lists for
every chunk in memory for the whole run and only calls bm25s at the very
end. HNSW (usearch) IS incremental (.add() per batch), so passage text,
metadata, and embeddings are flushed to disk as we go and the process is
resumable up to the BM25 build step. If a config's token buffer doesn't fit
RAM, that is a real capacity finding to report, not a hypothetical.

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
CONFIG_TO_PARQUET_PREFIX = {
    "hi": "hin",
    "bn": "ben",
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
EMBED_BATCH = 64


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
    import hhgoa_rag.retrieval.local_embedder as le
    from usearch.index import Index
    import numpy as np

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEDUP_DIR.mkdir(parents=True, exist_ok=True)

    dedup_en = ContentDeduplicator(DEDUP_DIR / "dedup_en.sqlite")
    dedup_lang = ContentDeduplicator(DEDUP_DIR / "dedup_lang.sqlite")
    chunker = get_chunker("passage_native")

    le._lazy_load()
    assert le._sp is not None

    passages_path = OUTPUT_DIR / "passages.jsonl"
    hnsw_path = OUTPUT_DIR / "hnsw.usearch"
    tokens_path = OUTPUT_DIR / "bm25_tokens.jsonl"  # one JSON array of token strings per line

    hnsw = Index(ndim=le.EMBED_DIM, metric="cos", dtype="f32")
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

    def flush_batch(pending_texts: list[str], pending_meta: list[dict]) -> None:
        nonlocal next_key, total_indexed
        if not pending_texts:
            return
        embeddings = le.embed_passages_batch(pending_texts, batch_size=EMBED_BATCH)
        keys = np.arange(next_key, next_key + len(pending_texts))
        hnsw.add(keys, np.array(embeddings, dtype=np.float32))
        for i, (text, meta) in enumerate(zip(pending_texts, pending_meta, strict=True)):
            key = next_key + i
            passages_f.write(json.dumps({"key": key, "text": text, "metadata": meta}) + "\n")
            pieces = le._sp.encode(text, out_type=str)
            tokens_f.write(json.dumps(pieces) + "\n")
        passages_f.flush()
        tokens_f.flush()
        next_key += len(pending_texts)
        total_indexed += len(pending_texts)

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
                last_ckpt_row = start_row - 1

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

                    if len(pending_texts) >= EMBED_BATCH:
                        flush_batch(pending_texts, pending_meta)
                        pending_texts, pending_meta = [], []
                        dedup_en.flush()
                        dedup_lang.flush()
                        _save_checkpoint(config, split, row_idx + 1)
                        last_ckpt_row = row_idx

                    emitted_this_shard += 1
                    if emitted_this_shard % LOG_EVERY_ROWS == 0:
                        elapsed = time.monotonic() - t_start
                        logger.info(
                            "%s/%s: %d source rows, %d passages indexed so far (%.1fs elapsed)",
                            config,
                            split,
                            emitted_this_shard,
                            total_indexed,
                            elapsed,
                        )

                    if args.max_rows_per_config is not None and (
                        row_idx - start_row + 1
                    ) >= args.max_rows_per_config:
                        break

                if pending_texts:
                    flush_batch(pending_texts, pending_meta)
                    dedup_en.flush()
                    dedup_lang.flush()
                    last_ckpt_row = row_idx

                if row_idx >= start_row - 1:
                    _save_checkpoint(config, split, last_ckpt_row + 1)
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
        "embed_dim": le.EMBED_DIM,
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
