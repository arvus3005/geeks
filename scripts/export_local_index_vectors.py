"""Export embedding vectors from a built local HNSW index to a portable
.npy file, keyed by the same sequential integer `key` field used in
passages.jsonl (0..N-1, no gaps -- see build_full_local_index.py).

Needed because usearch does not support merging two independently-built
HNSW indexes directly. scripts/merge_local_indexes.py calls this (or reads
its output if already run) to rebuild one combined index from multiple
shards without re-computing any embeddings.

Usage:
    uv run python -m scripts.export_local_index_vectors artifacts/full_local_index
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from usearch.index import Index

EMBED_DIM = 384


def export_vectors(shard_dir: Path) -> Path:
    hnsw_path = shard_dir / "hnsw.usearch"
    if not hnsw_path.exists():
        raise FileNotFoundError(f"No hnsw.usearch in {shard_dir}")
    out_path = shard_dir / "embeddings.npy"

    idx = Index(ndim=EMBED_DIM, metric="cos", dtype="f32")
    idx.load(str(hnsw_path))
    n = idx.size

    # idx.vectors is NOT guaranteed to be in key order (usearch stores by
    # internal HNSW graph position) -- verified empirically that idx.get()
    # with an explicit key array IS correctly ordered, so use that even
    # though it's a slightly less direct API.
    keys = np.arange(n)
    vectors = np.array(idx.get(keys), dtype=np.float32)
    if vectors.shape != (n, EMBED_DIM):
        raise RuntimeError(
            f"Expected ({n}, {EMBED_DIM}) vectors, got {vectors.shape} -- "
            "index may have gaps in its key sequence (shouldn't happen with "
            "build_full_local_index.py's sequential key assignment)."
        )
    np.save(out_path, vectors)

    manifest_path = shard_dir / "manifest.json"
    n_passages = None
    if manifest_path.exists():
        n_passages = json.loads(manifest_path.read_text()).get("n_passages")
    if n_passages is not None and n_passages != n:
        print(
            f"WARNING: manifest.json says n_passages={n_passages} but the "
            f"HNSW index has {n} vectors -- shard may be from an in-progress "
            "run. embeddings.npy reflects the HNSW index's current state."
        )
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("shard_dir", type=Path)
    args = ap.parse_args()
    out = export_vectors(args.shard_dir)
    print(f"Exported {out}")


if __name__ == "__main__":
    main()
