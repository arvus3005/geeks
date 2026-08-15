"""Smoke ingestion — deterministic fixtures covering all 15 language codes.

Uploads 50–100 representative passages to the Pinecone smoke namespace.
Pinecone handles server-side embedding; no local model is loaded.
Records are idempotent — re-ingesting the same IDs does not grow the count.

SMOKE ONLY: never targets the full namespace.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..ingestion.normalizer import content_hash, normalize_text
from ..ingestion.passage_ids import make_point_id
from ..pinecone_store import SMOKE_NAMESPACE, TEXT_RECORD_FIELD, PineconeStore

SMOKE_FIXTURES_PATH = (
    Path(__file__).parent.parent.parent.parent / "tests" / "fixtures" / "smoke_passages.json"
)

DATASET_REVISION = "smoke-fixture-v1"
CHUNK_STRATEGY_VERSION = "passage_native_v1"


def _build_record(p: dict) -> dict:
    lang = p["language"]
    norm = normalize_text(p["text"])
    chash = content_hash(p["text"])
    point_id = make_point_id(
        dataset_revision=DATASET_REVISION,
        language=lang,
        content_hash=chash,
        chunk_strategy_version=CHUNK_STRATEGY_VERSION,
        chunk_ordinal=0,
    )
    return {
        "id": point_id,
        TEXT_RECORD_FIELD: norm,
        "language": lang,
        "content_hash": chash,
        "chunk_strategy": "passage_native",
        "chunk_strategy_version": CHUNK_STRATEGY_VERSION,
        "chunk_ordinal": 0,
        "source_split": "smoke",
        "index_manifest_id": "smoke-v001",
    }


def run_smoke_ingest(
    store: PineconeStore,
    namespace: str = SMOKE_NAMESPACE,
    fixtures_path: Path = SMOKE_FIXTURES_PATH,
) -> dict:
    """Ingest smoke fixtures into Pinecone. Idempotent (same point IDs)."""
    if namespace != SMOKE_NAMESPACE:
        raise ValueError(f"Smoke ingest only targets namespace '{SMOKE_NAMESPACE}', got '{namespace}'")

    with open(fixtures_path) as f:
        passages = json.load(f)

    records = [_build_record(p) for p in passages]
    submitted = store.upsert_records(records, namespace=namespace, context="smoke")
    return {
        "success": submitted == len(records),
        "records_submitted": submitted,
        "namespace": namespace,
        "fixtures_count": len(records),
    }
