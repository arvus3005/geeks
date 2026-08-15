"""Hybrid dense+sparse retrieval using Qdrant RRF fusion.

Sparse encoding uses FastEmbed BM25 (Qdrant/bm25) at both ingestion and query time,
giving stable cross-process token IDs and proper BM25 scoring with IDF weighting.
"""

from __future__ import annotations

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, Fusion, FusionQuery, MatchAny, Prefetch

from .sparse_encoder import BM25SparseEncoder, get_bm25_encoder


class HybridRetriever:
    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        dense_k: int = 32,
        sparse_k: int = 32,
        fused_k: int = 20,
        sparse_encoder: BM25SparseEncoder | None = None,
    ) -> None:
        self.client = client
        self.collection = collection
        self.dense_k = dense_k
        self.sparse_k = sparse_k
        self.fused_k = fused_k
        # Allow injection for tests; default to process-level singleton.
        self._sparse_encoder = sparse_encoder or get_bm25_encoder()

    def retrieve(
        self,
        query_vector: np.ndarray,
        query_text: str,
        language_filter: list[str] | None = None,
    ) -> list[dict]:
        sparse_vec = self._sparse_encoder.encode_query(query_text)

        flt: Filter | None = None
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
