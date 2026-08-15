"""Smoke ingestion — uses local deterministic fixtures covering all 15 language codes."""

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
from ..retrieval.hybrid import text_to_sparse

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
    embedder = FakeEmbedder()

    # Create collection if needed
    existing = {c.name for c in client.get_collections().collections}
    if collection not in existing:
        create_collection(client, collection, force=False)

    with open(fixtures_path) as f:
        passages = json.load(f)

    points = []
    for p in passages:
        lang = p["language"]
        text = p["text"]
        norm = normalize_text(text)
        chash = content_hash(text)
        point_id = make_point_id(
            dataset_revision=DATASET_REVISION,
            language=lang,
            content_hash=chash,
            chunk_strategy_version=CHUNK_STRATEGY_VERSION,
            chunk_ordinal=0,
        )
        dense_vec = embedder.embed_query(f"passage: {norm}")
        sparse_vec = text_to_sparse(norm)
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

    # Batch upsert
    client.upsert(collection_name=collection, points=points, wait=True)
    result = validate_collection(client, collection)
    return {
        "success": result["valid"],
        "points_ingested": len(points),
        "validation": result,
    }
