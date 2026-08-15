"""Create the Qdrant collection with dense + sparse vectors."""
from qdrant_client.models import (
    VectorParams,
    Distance,
    SparseVectorParams,
    SparseIndexParams,
    HnswConfigDiff,
)

COLLECTION_PHYSICAL = "msmarco_xi_passages_v001"
COLLECTION_ALIAS = "msmarco_xi_passages_current"
DENSE_DIM = 384


def build_collection_config() -> dict:
    return {
        "vectors_config": {"dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE)},
        "sparse_vectors_config": {
            "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
        },
        "hnsw_config": HnswConfigDiff(m=16, ef_construct=200),
    }


def create_collection(url: str = "http://localhost:6333", api_key: str | None = None):
    from qdrant_client import QdrantClient

    client = QdrantClient(url=url, api_key=api_key)
    config = build_collection_config()
    client.recreate_collection(
        collection_name=COLLECTION_PHYSICAL,
        vectors_config=config["vectors_config"],
        sparse_vectors_config=config["sparse_vectors_config"],
        hnsw_config=config["hnsw_config"],
    )
    print(f"Created collection: {COLLECTION_PHYSICAL}")


if __name__ == "__main__":
    import sys

    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:6333"
    create_collection(url)
