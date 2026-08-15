"""Smoke ingestion — deterministic fixtures covering all 15 language codes.

Uses FakeEmbedder for dense vectors and BM25SparseEncoder for sparse vectors,
matching exactly the encoders used at query time.
"""

import json
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from ..ingestion.normalizer import content_hash, normalize_text
from ..ingestion.passage_ids import make_point_id
from ..qdrant_lifecycle import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    create_collection,
    validate_collection,
)
from ..retrieval.embedder import FakeEmbedder
from ..retrieval.sparse_encoder import BM25SparseEncoder

SMOKE_FIXTURES_PATH = (
    Path(__file__).parent.parent.parent.parent / "tests" / "fixtures" / "smoke_passages.json"
)

DATASET_REVISION = "smoke-fixture-v1"
CHUNK_STRATEGY_VERSION = "passage_native_v1"


def run_smoke_ingest(
    qdrant_url: str = "http://localhost:6333",
    collection: str = "msmarco_xi_passages_smoke_v001",
    fixtures_path: Path = SMOKE_FIXTURES_PATH,
) -> dict:
    """Ingest smoke fixtures into Qdrant. Idempotent (same point IDs)."""
    client = QdrantClient(url=qdrant_url, timeout=30)
    dense_embedder = FakeEmbedder()
    sparse_encoder = BM25SparseEncoder()

    existing = {c.name for c in client.get_collections().collections}
    if collection not in existing:
        create_collection(client, collection, force=False)

    with open(fixtures_path) as f:
        passages = json.load(f)

    texts = [normalize_text(p["text"]) for p in passages]
    sparse_vecs = sparse_encoder.encode_passages_batch(texts)

    points = []
    for p, norm, sparse_vec in zip(passages, texts, sparse_vecs):
        lang = p["language"]
        chash = content_hash(p["text"])
        point_id = make_point_id(
            dataset_revision=DATASET_REVISION,
            language=lang,
            content_hash=chash,
            chunk_strategy_version=CHUNK_STRATEGY_VERSION,
            chunk_ordinal=0,
        )
        dense_vec = dense_embedder.embed_query(f"passage: {norm}")
        points.append(
            PointStruct(
                id=point_id,
                vector={
                    DENSE_VECTOR_NAME: dense_vec.tolist(),
                    SPARSE_VECTOR_NAME: sparse_vec,
                },
                payload={
                    "text": norm,
                    "language": lang,
                    "content_hash": chash,
                    "chunk_strategy": "passage_native",
                    "chunk_strategy_version": CHUNK_STRATEGY_VERSION,
                    "chunk_ordinal": 0,
                    "source_split": "smoke",
                    "index_manifest_id": "smoke-v001",
                },
            )
        )

    client.upsert(collection_name=collection, points=points, wait=True)
    result = validate_collection(client, collection)
    return {
        "success": result["valid"],
        "points_ingested": len(points),
        "validation": result,
    }
