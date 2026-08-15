"""Safe, versioned Qdrant collection lifecycle management."""

from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    PayloadSchemaType,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

DENSE_DIM = 384
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
SERVING_ALIAS = "msmarco_xi_passages_current"
_SMOKE_PREFIX = "msmarco_xi_passages_smoke"
_PILOT_PREFIX = "msmarco_xi_passages_pilot"


def _is_destructive_safe(collection_name: str) -> bool:
    """Only allow destructive ops on smoke/test collections, never production/alias."""
    return (
        collection_name.startswith(_SMOKE_PREFIX)
        or collection_name.startswith(_PILOT_PREFIX)
        or collection_name.startswith("test_")
    )


def create_collection(
    client: QdrantClient,
    collection_name: str,
    force: bool = False,
) -> None:
    """Create a new versioned Qdrant collection with dense+sparse vectors and payload indexes.

    Never calls recreate_collection. Fails if physical name exists with mismatched schema
    unless force=True and the name matches a smoke/pilot pattern.
    """
    if force and not _is_destructive_safe(collection_name):
        raise ValueError(f"--force refused for non-smoke/pilot collection: {collection_name}")

    existing = {c.name for c in client.get_collections().collections}

    if collection_name in existing:
        if force and _is_destructive_safe(collection_name):
            client.delete_collection(collection_name)
        else:
            raise RuntimeError(
                f"Collection '{collection_name}' already exists. "
                "Use --force with a smoke/pilot collection name to recreate."
            )

    client.create_collection(
        collection_name=collection_name,
        vectors_config={DENSE_VECTOR_NAME: VectorParams(size=DENSE_DIM, distance=Distance.COSINE)},
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: SparseVectorParams(
                index=SparseIndexParams(on_disk=False),
            )
        },
        hnsw_config=HnswConfigDiff(m=16, ef_construct=200),
    )

    # Create payload indexes before ingestion
    client.create_payload_index(
        collection_name=collection_name,
        field_name="language",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=collection_name,
        field_name="chunk_strategy",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=collection_name,
        field_name="content_hash",
        field_schema=PayloadSchemaType.KEYWORD,
    )


def validate_collection(client: QdrantClient, collection_name: str) -> dict:
    """Validate collection health, vectors, indexes."""
    info = client.get_collection(collection_name)
    issues = []

    vectors_cfg = info.config.params.vectors
    if not isinstance(vectors_cfg, dict) or DENSE_VECTOR_NAME not in vectors_cfg:
        issues.append(f"Missing '{DENSE_VECTOR_NAME}' vector config")

    sparse_cfg = info.config.params.sparse_vectors
    if sparse_cfg is None or SPARSE_VECTOR_NAME not in sparse_cfg:
        issues.append(f"Missing '{SPARSE_VECTOR_NAME}' sparse vector config")

    count = client.count(collection_name).count

    return {
        "collection": collection_name,
        "status": info.status.value if hasattr(info.status, "value") else str(info.status),
        "points": count,
        "issues": issues,
        "valid": len(issues) == 0,
    }


def switch_alias(
    client: QdrantClient,
    target_collection: str,
    alias: str = SERVING_ALIAS,
    smoke_ok: bool = False,
) -> None:
    """Atomically switch the serving alias to target_collection.

    Refuses pilot/smoke collections unless smoke_ok=True (for tests only).
    """
    is_smoke = _is_destructive_safe(target_collection)
    if is_smoke and not smoke_ok:
        raise ValueError(
            f"Collection '{target_collection}' is smoke/pilot and cannot be activated "
            "as the production serving alias without smoke_ok=True."
        )

    # Validate first
    result = validate_collection(client, target_collection)
    if not result["valid"]:
        raise RuntimeError(f"Validation failed before alias switch: {result['issues']}")

    from qdrant_client.models import (
        CreateAlias,
        CreateAliasOperation,
        DeleteAlias,
        DeleteAliasOperation,
    )

    try:
        client.update_collection_aliases(
            change_aliases_operations=[
                DeleteAliasOperation(delete_alias=DeleteAlias(alias_name=alias)),
                CreateAliasOperation(
                    create_alias=CreateAlias(collection_name=target_collection, alias_name=alias)
                ),
            ]
        )
    except Exception:
        # If delete fails (alias didn't exist), just create
        client.update_collection_aliases(
            change_aliases_operations=[
                CreateAliasOperation(
                    create_alias=CreateAlias(collection_name=target_collection, alias_name=alias)
                ),
            ]
        )
