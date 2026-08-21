"""Build a self-hosted BM25 + HNSW hybrid retrieval index from the live
corpus, as an alternative to Pinecone (no network hop, no hosting quota).

Branch scope (feat/self-hosted-hybrid-retrieval)
--------------------------------------------------
Pinecone already meets the latency target once deployed near its region
(server-side P50 42.9ms, verified live). This branch is an exploration of
"vast chunking/retrieval" per the task spec, not a fix for a problem —
in-process hybrid search removes the network hop entirely and adds lexical
(BM25) recall that pure dense search can miss (exact keyword/name/number
matches).

Pulls passage text from the live Pinecone index's own metadata (same
approach as scripts/reindex_e5small.py) rather than re-downloading from
HuggingFace. Builds two indexes over the same 57,240 passages:

1. BM25 (bm25s) — tokenized on SentencePiece subword pieces (not naive
   whitespace splitting), since the corpus spans en/hi/bn and we already
   have a correctly-configured multilingual tokenizer loaded for the dense
   path. Reusing it avoids needing a separate Indic word-tokenizer.
2. HNSW (usearch) — e5-small embeddings, same model/tokenization as the
   Pinecone path (local_embedder.py), so results are comparable.

Usage:
    PINECONE_API_KEY=... uv run python -m scripts.build_local_hybrid_index
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SOURCE_INDEX = "msmarco-xi-e5small"
SOURCE_NAMESPACE = "pilot_v1"
TEXT_FIELD = "chunk_text"
FETCH_BATCH = 100
OUTPUT_DIR = Path("artifacts/local_index")


def _sp_tokenize(texts: list[str]) -> list[list[str]]:
    """Tokenize on SentencePiece subword pieces, stringified as BM25 terms.

    Not whitespace splitting — that breaks for Devanagari/Bengali, which
    don't reliably word-segment on spaces the way Latin scripts do.
    Subword pieces are a script-agnostic, already-verified-correct choice
    since local_embedder.py's tokenizer is used unchanged.
    """
    import hhgoa_rag.retrieval.local_embedder as le

    le._lazy_load()
    assert le._sp is not None
    return [le._sp.encode(t, out_type=str) for t in texts]


def main() -> None:
    import os

    from pinecone import Pinecone

    from hhgoa_rag.pinecone_store import PineconeStore

    api_key = os.environ.get("PINECONE_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("PINECONE_API_KEY not set")

    pc = Pinecone(api_key=api_key)
    src_index = pc.Index(SOURCE_INDEX)
    src_store = PineconeStore(index=src_index, embed_model="multilingual-e5-small")

    logger.info("Enumerating live IDs in %s/%s …", SOURCE_INDEX, SOURCE_NAMESPACE)
    t0 = time.monotonic()
    ids = src_store.list_vector_ids(namespace=SOURCE_NAMESPACE)
    logger.info("Enumerated %d IDs in %.1fs", len(ids), time.monotonic() - t0)

    passage_ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict] = []

    t0 = time.monotonic()
    for batch_start in range(0, len(ids), FETCH_BATCH):
        batch_ids = ids[batch_start : batch_start + FETCH_BATCH]
        resp = src_index.fetch(ids=batch_ids, namespace=SOURCE_NAMESPACE)
        vectors = resp.vectors if hasattr(resp, "vectors") else resp["vectors"]
        for vid in batch_ids:
            v = vectors.get(vid)
            if v is None:
                continue
            metadata = dict(v.metadata) if hasattr(v, "metadata") else dict(v["metadata"])
            text = metadata.get(TEXT_FIELD, "")
            if not text:
                continue
            passage_ids.append(vid)
            texts.append(text)
            metadatas.append(metadata)
        if (batch_start // FETCH_BATCH) % 20 == 0:
            logger.info("Fetched %d/%d …", len(passage_ids), len(ids))
    logger.info("Fetched %d passages in %.1fs", len(passage_ids), time.monotonic() - t0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── BM25 ──────────────────────────────────────────────────────────────
    import bm25s

    logger.info("Tokenizing corpus for BM25 (SentencePiece pieces) …")
    t0 = time.monotonic()
    corpus_tokens = _sp_tokenize(texts)
    logger.info("Tokenized in %.1fs", time.monotonic() - t0)

    logger.info("Building BM25 index …")
    t0 = time.monotonic()
    bm25 = bm25s.BM25()
    bm25.index(corpus_tokens)
    bm25.save(str(OUTPUT_DIR / "bm25"))
    logger.info("BM25 index built + saved in %.1fs", time.monotonic() - t0)

    # ── HNSW (usearch) ───────────────────────────────────────────────────
    from usearch.index import Index

    from hhgoa_rag.retrieval.local_embedder import EMBED_DIM, embed_passages_batch

    logger.info("Embedding %d passages for HNSW (e5-small) …", len(texts))
    t0 = time.monotonic()
    embeddings = embed_passages_batch(texts, batch_size=64)
    logger.info("Embedded in %.1fs", time.monotonic() - t0)

    logger.info("Building HNSW index …")
    t0 = time.monotonic()
    hnsw = Index(ndim=EMBED_DIM, metric="cos", dtype="f32")
    keys = list(range(len(passage_ids)))
    import numpy as np

    hnsw.add(np.array(keys), np.array(embeddings, dtype=np.float32))
    hnsw.save(str(OUTPUT_DIR / "hnsw.usearch"))
    logger.info("HNSW index built + saved in %.1fs", time.monotonic() - t0)

    # ── Sidecar: id/text/metadata lookup by integer key ─────────────────
    with (OUTPUT_DIR / "passages.jsonl").open("w") as f:
        for i, (pid, text, meta) in enumerate(zip(passage_ids, texts, metadatas, strict=True)):
            f.write(json.dumps({"key": i, "id": pid, "text": text, "metadata": meta}) + "\n")

    manifest = {
        "source": f"{SOURCE_INDEX}/{SOURCE_NAMESPACE}",
        "n_passages": len(passage_ids),
        "embed_dim": EMBED_DIM,
        "bm25_backend": "bm25s",
        "hnsw_backend": "usearch",
        "tokenization": "sentencepiece_pieces",
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("Done: %s", manifest)


if __name__ == "__main__":
    main()
