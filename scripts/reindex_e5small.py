"""One-time migration: re-embed the live pilot_v1 corpus with the local
e5-small embedder and upsert into a new, dimension-compatible Pinecone index.

Why this exists
----------------
The original 57,240 passages were embedded via Pinecone's server-side
integrated multilingual-e5-large (1024-dim). Query-time embedding moved
local (see src/hhgoa_rag/retrieval/local_embedder.py) to escape both the
account's embedding-token quota and, eventually, e5-large's ~1.5-2GB RAM
footprint — settling on e5-small (384-dim), which is NOT vector-space
compatible with the existing e5-large index. This script re-embeds the same
passage text (pulled from the live index's own metadata — not re-downloaded
from HuggingFace) with e5-small and writes it into a new raw-vector index.

Safety
------
- Source index/namespace are read-only here — nothing is deleted or
  modified in msmarco-xi/pilot_v1.
- The destination index is newly created, never an in-place recreate of a
  production or serving-alias index (CLAUDE.md Vector Store Safety).
- `query`/`Answer`/etc. leakage-boundary fields cannot leak here: the source
  metadata was already stripped of them at original indexing time, and this
  script only ever reads and re-writes that same metadata unchanged.

Concurrency
-----------
A sequential first pass measured ~10.5s per 100-passage batch (fetch 3.3s +
embed 4.4s + upsert 2.8s — all three genuinely load-bearing, not one clear
bottleneck), projecting to ~3 hours for 57,240 passages. Fetch/upsert are
network-bound and ONNX Runtime sessions are safe for concurrent calls from
multiple threads, so batches are processed by a thread pool instead of one
at a time — CPU has 12 cores available on the machine this was developed on.

Usage:
    PINECONE_API_KEY=... uv run python -m scripts.reindex_e5small \\
        --dest-index msmarco-xi-e5small --execute
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SOURCE_INDEX = "msmarco-xi"
SOURCE_NAMESPACE = "pilot_v1"
TEXT_FIELD = "chunk_text"
FETCH_BATCH = 100  # Pinecone's fetch endpoint takes IDs as GET query params —
# larger batches (e.g. 500 UUIDs) overflow the URL length limit (414)
EMBED_BATCH = 32
UPSERT_BATCH = 200
WORKERS = 8


def _process_batch(
    batch_ids: list[str],
    src_index,
    dest_index,
    dest_namespace: str,
    execute: bool,
) -> tuple[int, float, float, float]:
    from hhgoa_rag.retrieval.local_embedder import embed_passages_batch

    t0 = time.monotonic()
    fetch_resp = src_index.fetch(ids=batch_ids, namespace=SOURCE_NAMESPACE)
    vectors = fetch_resp.vectors if hasattr(fetch_resp, "vectors") else fetch_resp["vectors"]
    fetch_s = time.monotonic() - t0

    records = []
    for vid in batch_ids:
        v = vectors.get(vid)
        if v is None:
            logger.warning("ID %s missing from fetch response — skipping", vid)
            continue
        metadata = dict(v.metadata) if hasattr(v, "metadata") else dict(v["metadata"])
        text = metadata.get(TEXT_FIELD, "")
        if not text:
            logger.warning("ID %s has empty %s — skipping", vid, TEXT_FIELD)
            continue
        records.append((vid, text, metadata))

    if not records:
        return 0, fetch_s, 0.0, 0.0

    t0 = time.monotonic()
    texts = [r[1] for r in records]
    embeddings = embed_passages_batch(texts, batch_size=EMBED_BATCH)
    embed_s = time.monotonic() - t0

    t0 = time.monotonic()
    if execute:
        assert dest_index is not None
        for up_start in range(0, len(records), UPSERT_BATCH):
            chunk = records[up_start : up_start + UPSERT_BATCH]
            chunk_vecs = embeddings[up_start : up_start + UPSERT_BATCH]
            upsert_payload = [
                {"id": vid, "values": vec, "metadata": metadata}
                for (vid, _text, metadata), vec in zip(chunk, chunk_vecs, strict=True)
            ]
            dest_index.upsert(vectors=upsert_payload, namespace=dest_namespace)
    upsert_s = time.monotonic() - t0

    return len(records), fetch_s, embed_s, upsert_s


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest-index", required=True)
    parser.add_argument("--dest-namespace", default=SOURCE_NAMESPACE)
    parser.add_argument("--execute", action="store_true", help="Without this, dry-run only")
    parser.add_argument("--limit", type=int, default=None, help="Cap passages (smoke test)")
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args()

    import os

    from pinecone import Pinecone, ServerlessSpec

    from hhgoa_rag.pinecone_store import PineconeStore
    from hhgoa_rag.retrieval.local_embedder import EMBED_DIM

    api_key = os.environ.get("PINECONE_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("PINECONE_API_KEY not set")

    pc = Pinecone(api_key=api_key)
    src_index = pc.Index(SOURCE_INDEX)
    src_store = PineconeStore(index=src_index, embed_model="multilingual-e5-large")

    logger.info("Enumerating live IDs in %s/%s …", SOURCE_INDEX, SOURCE_NAMESPACE)
    t0 = time.monotonic()
    ids = src_store.list_vector_ids(namespace=SOURCE_NAMESPACE)
    logger.info("Enumerated %d IDs in %.1fs", len(ids), time.monotonic() - t0)

    if args.limit:
        ids = ids[: args.limit]
        logger.info("Limited to %d IDs for this run", len(ids))

    if args.dest_index not in {i.name for i in pc.list_indexes()}:
        if not args.execute:
            logger.info("[dry-run] Would create index %r (dim=%d)", args.dest_index, EMBED_DIM)
        else:
            logger.info(
                "Creating index %r (dim=%d, cosine, aws/us-east-1) …", args.dest_index, EMBED_DIM
            )
            pc.create_index(
                name=args.dest_index,
                dimension=EMBED_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            while not pc.describe_index(args.dest_index).status["ready"]:
                time.sleep(1)

    dest_index = pc.Index(args.dest_index) if args.execute else None

    batches = [ids[i : i + FETCH_BATCH] for i in range(0, len(ids), FETCH_BATCH)]

    total_upserted = 0
    total_fetch_s = 0.0
    total_embed_s = 0.0
    total_upsert_s = 0.0
    lock = threading.Lock()
    start = time.monotonic()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _process_batch, b, src_index, dest_index, args.dest_namespace, args.execute
            ): b
            for b in batches
        }
        for fut in as_completed(futures):
            n, fetch_s, embed_s, upsert_s = fut.result()
            with lock:
                total_upserted += n
                total_fetch_s += fetch_s
                total_embed_s += embed_s
                total_upsert_s += upsert_s
                elapsed = time.monotonic() - start
                logger.info(
                    "%d/%d re-embedded%s (%.1fs elapsed, %.1f passages/s)",
                    total_upserted,
                    len(ids),
                    "" if args.execute else " [dry-run, not upserted]",
                    elapsed,
                    total_upserted / elapsed if elapsed > 0 else 0.0,
                )

    report = {
        "source": f"{SOURCE_INDEX}/{SOURCE_NAMESPACE}",
        "dest": f"{args.dest_index}/{args.dest_namespace}",
        "dest_dim": EMBED_DIM,
        "total_ids": len(ids),
        "total_upserted": total_upserted,
        "executed": args.execute,
        "workers": args.workers,
        "fetch_s_total": round(total_fetch_s, 1),
        "embed_s_total": round(total_embed_s, 1),
        "upsert_s_total": round(total_upsert_s, 1),
        "wall_clock_s": round(time.monotonic() - start, 1),
        "generated_at": datetime.now(UTC).isoformat(),
    }
    logger.info("Done: %s", report)

    import json
    from pathlib import Path

    reports_dir = Path("artifacts/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / f"reindex_e5small_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
