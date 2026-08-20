"""Merge multiple independently-built local index shards (e.g. from
different machines/languages, each produced by build_full_local_index.py)
into one combined BM25 + HNSW index.

Each shard directory must contain: passages.jsonl, hnsw.usearch (an
embeddings.npy is exported from it automatically if missing), and
bm25_tokens.jsonl. manifest.json is read if present for a smell-test
warning (not a guarantee) about embedding-backend consistency.

CRITICAL PRECONDITION, NOT ENFORCED BY THIS SCRIPT: every shard must have
been built with the EXACT SAME embedding backend and precision (this
codebase's MPS fp16 pipeline in build_full_local_index.py, as of
2026-08-21 -- see docs/FULL_INDEX_HANDOFF.md). Mixing shards embedded with
different backends/precisions silently corrupts cosine similarity across
the merged index -- there is no automated way to detect this from the
vectors alone, so it must be verified out of band (confirm every
contributor ran the identical script version on an Apple Silicon Mac).

Dedup: every MSMARCO-XI language config shares the same underlying English
passage pool (measured, see docs/FULL_INDEX_HANDOFF.md) -- English
passages are deduped across ALL shards here. Dedup key is
(passage_language, content_hash), not content_hash alone, so this can't
accidentally collide translated passages from different languages that
happen to share text (extremely unlikely, but free to guard against). If
two shards cover the SAME language, their translated passages will also
correctly dedup against each other -- but assign distinct --configs per
contributor; don't rely on this as your dedup strategy.

Usage:
    uv run python -m scripts.merge_local_indexes \\
        --shards artifacts/full_local_index artifacts/shard_gu artifacts/shard_ta \\
        --output-dir artifacts/merged_local_index
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EMBED_DIM = 384


def _load_shard(shard_dir: Path):
    """Yields (text, metadata, embedding, bm25_tokens) for one shard, in
    local key order (0..N-1), reading passages.jsonl/bm25_tokens.jsonl
    line-aligned with embeddings.npy row order."""
    from scripts.export_local_index_vectors import export_vectors

    passages_path = shard_dir / "passages.jsonl"
    tokens_path = shard_dir / "bm25_tokens.jsonl"
    embeddings_path = shard_dir / "embeddings.npy"

    if not passages_path.exists() or not tokens_path.exists():
        raise FileNotFoundError(f"{shard_dir} is missing passages.jsonl or bm25_tokens.jsonl")

    if not embeddings_path.exists():
        logger.info("No embeddings.npy in %s, exporting from hnsw.usearch...", shard_dir)
        export_vectors(shard_dir)

    embeddings = np.load(embeddings_path)

    manifest_path = shard_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        logger.info(
            "%s manifest: configs=%s n_passages=%s embed_backend=%s",
            shard_dir,
            manifest.get("configs"),
            manifest.get("n_passages"),
            manifest.get("embed_backend", "UNKNOWN -- verify this matches other shards!"),
        )

    with (
        open(passages_path) as pf,
        open(tokens_path) as tf,
    ):
        for i, (pline, tline) in enumerate(zip(pf, tf, strict=True)):
            record = json.loads(pline)
            tokens = json.loads(tline)
            if record["key"] != i:
                raise RuntimeError(
                    f"{shard_dir}: passages.jsonl key {record['key']} != line position {i} "
                    "-- file may be corrupt or out of order."
                )
            if i >= len(embeddings):
                raise RuntimeError(
                    f"{shard_dir}: more passage lines than embedding rows "
                    f"({len(embeddings)}) -- shard may be from an in-progress run; "
                    "re-export embeddings.npy after the shard finishes."
                )
            yield record["text"], record["metadata"], embeddings[i], tokens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    args = ap.parse_args()

    from usearch.index import Index

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_passages = open(args.output_dir / "passages.jsonl", "w")
    out_tokens = open(args.output_dir / "bm25_tokens.jsonl", "w")

    seen: set[tuple[str, str]] = set()  # (passage_language, content_hash)
    all_embeddings: list[np.ndarray] = []
    per_language_counts: dict[str, int] = {}
    global_key = 0
    t_start = time.monotonic()

    for shard_dir in args.shards:
        logger.info("Merging shard %s ...", shard_dir)
        shard_new = 0
        shard_dup = 0
        for text, meta, embedding, tokens in _load_shard(shard_dir):
            dedup_key = (meta.get("language", "unknown"), meta["content_hash"])
            if dedup_key in seen:
                shard_dup += 1
                continue
            seen.add(dedup_key)

            out_passages.write(
                json.dumps({"key": global_key, "text": text, "metadata": meta}) + "\n"
            )
            out_tokens.write(json.dumps(tokens) + "\n")
            all_embeddings.append(embedding)
            per_language_counts[meta.get("language", "unknown")] = (
                per_language_counts.get(meta.get("language", "unknown"), 0) + 1
            )
            global_key += 1
            shard_new += 1

        logger.info("Shard %s: %d new passages, %d cross-shard duplicates", shard_dir, shard_new, shard_dup)

    out_passages.close()
    out_tokens.close()
    logger.info(
        "Merged %d unique passages from %d shards in %.1fs. Per-language: %s",
        global_key,
        len(args.shards),
        time.monotonic() - t_start,
        per_language_counts,
    )

    logger.info("Building combined HNSW index...")
    t0 = time.monotonic()
    hnsw = Index(ndim=EMBED_DIM, metric="cos", dtype="f32")
    embeddings_arr = np.array(all_embeddings, dtype=np.float32)
    hnsw.add(np.arange(global_key), embeddings_arr)
    hnsw.save(str(args.output_dir / "hnsw.usearch"))
    np.save(args.output_dir / "embeddings.npy", embeddings_arr)
    logger.info("HNSW index built (%d vectors) in %.1fs", global_key, time.monotonic() - t0)

    logger.info("Building combined BM25 index (full corpus in memory -- see build_full_local_index.py's docstring on this constraint)...")
    import bm25s

    corpus_tokens = []
    with open(args.output_dir / "bm25_tokens.jsonl") as f:
        for line in f:
            corpus_tokens.append(json.loads(line))
    t0 = time.monotonic()
    bm25 = bm25s.BM25()
    bm25.index(corpus_tokens)
    bm25.save(str(args.output_dir / "bm25"))
    logger.info("BM25 index built in %.1fs", time.monotonic() - t0)

    manifest = {
        "source_shards": [str(s) for s in args.shards],
        "n_passages": global_key,
        "per_language_counts": per_language_counts,
        "embed_dim": EMBED_DIM,
        "bm25_backend": "bm25s",
        "hnsw_backend": "usearch",
        "merged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "warning": (
            "This merge does NOT verify all source shards used the same "
            "embedding backend/precision. See docs/FULL_INDEX_HANDOFF.md."
        ),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("Done: %s", manifest)


if __name__ == "__main__":
    main()
