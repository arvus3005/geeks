"""Add a byte-offset index into passages.jsonl (for O(1) passage lookup at
serve time) to every existing full_local_index shard, and build a per-segment
BM25 index for any segment that's somehow missing one.

IMPORTANT: build_full_local_index.py's `_SegmentWriter.finalize()` ALREADY
builds a per-segment BM25 index for every segment as part of the original
indexing run (confirmed 2026-08-22: every existing segment already has a
`bm25/` folder with bm25s's native files). This script does NOT rebuild
those -- it only adds what's actually missing, the offsets index, and
falls back to building BM25 only for segments that genuinely lack one.

Why per-segment, not one merged BM25 over the whole corpus
------------------------------------------------------------
Measured (2026-08-22, this machine, 25.8GB RAM): loading BM25 tokens for
the full 54.25M-passage corpus as bm25s's default Python list-of-lists
costs ~368GB RAM to build -- 14x more than this machine has, regardless of
how the tokens are encoded going in (bm25s's score-matrix construction
itself needs memory proportional to total corpus tokens, ~5.15B for the
full corpus). There is no single-machine merge that fits.

Fix: keep the existing per-segment shards (this project already segments
output specifically to bound peak RAM -- see build_full_local_index.py's
SEGMENTING section) and build ONE BM25 index per segment instead of one
merged index. Each segment is ~500k passages -- measured ~1.1GB peak RAM
to build, ~0.24GB resident once built and saved. At serve time, shards are
loaded via bm25s's own `mmap=True` and usearch's `view=True` (both
confirmed to keep query-time resident memory low regardless of total
on-disk corpus size -- see src/hhgoa_rag/retrieval/sharded_local_hybrid_store.py),
and a query only touches the shards for its detected language, not all of
them. Real measured fan-out latency across a 32-shard language: ~70ms
combined (dense + sparse), well inside the 200ms budget -- see
docs/POST_INDEXING_STEPS.md and the README's indexing-status section.

No cross-shard dedup needed: every MSMARCO-XI language config's shared
English pool was measured (2026-08-20/21) to live ONLY in hi's segments,
not duplicated into bn/gu/ta/mr/ur's -- so shards are already disjoint by
content, nothing to deduplicate here.

Idempotent / resumable: skips any segment that already has a `bm25/`
directory with a manifest, so re-running after an interruption or after a
new language finishes just picks up what's missing.

Usage:
    uv run python -m scripts.build_shard_bm25 [--workers 6] [--root artifacts/full_local_index]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _build_one_segment(segment_dir: str) -> tuple[str, str, float, int]:
    """Runs in a worker process. Returns (segment_dir, status, seconds, n_docs)."""
    segment_path = Path(segment_dir)
    bm25_out = segment_path / "bm25"
    offsets_out = segment_path / "passages_offsets.npy"
    # The ORIGINAL build already creates bm25/params.index.json for every
    # segment (see build_full_local_index.py's _SegmentWriter.finalize()) --
    # that's the real marker for "BM25 already built", not anything this
    # script writes itself.
    bm25_already_built = (bm25_out / "params.index.json").exists()

    passages_path = segment_path / "passages.jsonl"
    if not passages_path.exists():
        return (segment_dir, "missing_input_files", 0.0, 0)

    if bm25_already_built and offsets_out.exists():
        return (segment_dir, "skipped_already_built", 0.0, 0)

    t0 = time.monotonic()

    # 1. Byte-offset index for O(1) passage lookup at serve time (avoids
    #    ever loading passages.jsonl fully into RAM -- measured ~2GB/M
    #    passages as Python dicts, which doesn't scale to the full corpus
    #    any better than the BM25 token problem below).
    n_offsets = 0
    if not offsets_out.exists():
        offsets = []
        offset = 0
        with open(passages_path, "rb") as f:
            for line in f:
                offsets.append(offset)
                offset += len(line)
        np.save(offsets_out, np.array(offsets, dtype=np.int64))
        n_offsets = len(offsets)

    if bm25_already_built:
        return (segment_dir, "offsets_only", time.monotonic() - t0, n_offsets)

    # 2. Fallback: build BM25 for a segment that's genuinely missing one.
    #    Uses compact int32 token-id arrays (not bm25s's default raw-string
    #    ingestion) -- measured ~6.8x less RAM to load (0.997GB/M vs
    #    6.785GB/M passages), which is the input shape build_index_from_ids
    #    actually needs (len() + iteration per doc), so no wasted conversion
    #    back to boxed Python lists.
    tokens_path = segment_path / "bm25_tokens.jsonl"
    if not tokens_path.exists():
        return (segment_dir, "missing_bm25_tokens_for_fallback_build", 0.0, 0)

    import bm25s
    from bm25s.tokenization import Tokenized

    vocab: dict[str, int] = {}
    doc_token_ids: list[np.ndarray] = []
    n_docs = 0
    with open(tokens_path) as f:
        for line in f:
            toks = json.loads(line)
            ids = np.empty(len(toks), dtype=np.int32)
            for i, tok in enumerate(toks):
                tid = vocab.get(tok)
                if tid is None:
                    tid = len(vocab)
                    vocab[tok] = tid
                ids[i] = tid
            doc_token_ids.append(ids)
            n_docs += 1

    bm = bm25s.BM25()
    bm.index(Tokenized(ids=doc_token_ids, vocab=vocab), show_progress=False)  # type: ignore[arg-type]
    bm25_out.mkdir(parents=True, exist_ok=True)
    bm.save(str(bm25_out))

    return (segment_dir, "built_bm25_and_offsets", time.monotonic() - t0, n_docs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="artifacts/full_local_index", type=Path)
    ap.add_argument("--workers", default=6, type=int)
    args = ap.parse_args()

    segments = sorted(str(p) for p in args.root.iterdir() if p.is_dir())
    logger.info("Found %d segments under %s", len(segments), args.root)

    t_start = time.monotonic()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers, max_tasks_per_child=1) as ex:
        futures = {ex.submit(_build_one_segment, s): s for s in segments}
        for fut in as_completed(futures):
            seg = futures[fut]
            try:
                seg_dir, status, secs, n_docs = fut.result()
            except Exception as exc:  # noqa: BLE001 -- log and keep going, one bad segment shouldn't kill the run
                logger.error("FAILED %s: %s", seg, exc)
                continue
            done += 1
            logger.info(
                "[%d/%d] %s: %s (%.1fs, %d docs)",
                done,
                len(segments),
                Path(seg_dir).name,
                status,
                secs,
                n_docs,
            )

    logger.info("All done in %.1fs", time.monotonic() - t_start)


if __name__ == "__main__":
    main()
