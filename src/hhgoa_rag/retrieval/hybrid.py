import hashlib
import re
from collections import Counter

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Fusion, FusionQuery, Prefetch, SparseVector


def _stable_token_id(token: str) -> int:
    """Stable cross-process token ID using SHA-256. Range: [0, 2^20)."""
    return int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:5], 16)


def text_to_sparse(text: str) -> SparseVector:
    """Lexical sparse vector with stable cross-process token IDs.

    Note: this is TF-normalized term frequency, NOT full BM25 (no IDF/doc-length normalization).
    For production BM25, use FastEmbed BM25 encoder.
    """
    tokens = re.findall(r"\b\w+\b", text.lower())
    if not tokens:
        return SparseVector(indices=[0], values=[0.0])
    counts = Counter(tokens)
    total = sum(counts.values())
    indices = [_stable_token_id(t) for t in counts]
    values = [c / total for c in counts.values()]
    # Sort by index (required by Qdrant)
    pairs = sorted(zip(indices, values))
    indices_sorted = [p[0] for p in pairs]
    values_sorted = [p[1] for p in pairs]
    return SparseVector(indices=indices_sorted, values=values_sorted)


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

        from qdrant_client.models import FieldCondition, Filter, MatchAny

        flt = None
        if language_filter:
            flt = Filter(must=[FieldCondition(key="language", match=MatchAny(any=language_filter))])

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
