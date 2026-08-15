from qdrant_client import QdrantClient
from qdrant_client.models import (
    SparseVector,
    Prefetch,
    FusionQuery,
    Fusion,
)
import numpy as np
import re
from collections import Counter


def text_to_sparse(text: str) -> SparseVector:
    """BM25-style sparse vector from token frequencies."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    counts = Counter(tokens)
    total = sum(counts.values())
    indices = [hash(t) % 100_000 for t in counts]
    values = [c / total for c in counts.values()]
    return SparseVector(indices=indices, values=values)


class HybridRetriever:
    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        dense_k: int = 32,
        sparse_k: int = 32,
        fused_k: int = 20,
    ):
        self.client = client
        self.collection = collection
        self.dense_k = dense_k
        self.sparse_k = sparse_k
        self.fused_k = fused_k

    def retrieve(
        self,
        query_vector: np.ndarray,
        query_text: str,
        language_filter: list[str] | None = None,
    ) -> list[dict]:
        sparse_vec = text_to_sparse(query_text)

        from qdrant_client.models import Filter, FieldCondition, MatchAny

        flt = None
        if language_filter:
            flt = Filter(
                must=[FieldCondition(key="language", match=MatchAny(any=language_filter))]
            )

        results = self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                Prefetch(
                    query=query_vector.tolist(),
                    using="dense",
                    limit=self.dense_k,
                    filter=flt,
                ),
                Prefetch(
                    query=sparse_vec,
                    using="sparse",
                    limit=self.sparse_k,
                    filter=flt,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=self.fused_k,
            with_payload=True,
        )

        return [
            {
                "id": str(p.id),
                "score": p.score,
                "payload": p.payload or {},
            }
            for p in results.points
        ]
