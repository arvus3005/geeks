"""Stable, deterministic sparse encoding using FastEmbed BM25 (Qdrant/bm25).

Uses distinct passage_embed and query_embed methods as required by the model.
Token IDs are stable across processes — no built-in hash() used.
"""

from __future__ import annotations

from typing import Any

from qdrant_client.models import SparseVector


class BM25SparseEncoder:
    """FastEmbed BM25 sparse encoder — one shared instance per process."""

    MODEL_NAME = "Qdrant/bm25"

    def __init__(self) -> None:
        self._model: Any = None  # SparseTextEmbedding, loaded lazily

    def _load(self) -> None:
        if self._model is None:
            from fastembed import SparseTextEmbedding

            self._model = SparseTextEmbedding(model_name=self.MODEL_NAME)

    def encode_passage(self, text: str) -> SparseVector:
        """Encode a passage for ingestion (BM25 document-side weights)."""
        self._load()
        assert self._model is not None
        results = list(self._model.passage_embed([text]))
        emb = results[0]
        indices = emb.indices.tolist()
        values = emb.values.tolist()
        pairs = sorted(zip(indices, values))
        return SparseVector(
            indices=[p[0] for p in pairs],
            values=[p[1] for p in pairs],
        )

    def encode_passages_batch(self, texts: list[str]) -> list[SparseVector]:
        """Batch encode passages. More efficient than calling encode_passage repeatedly."""
        self._load()
        assert self._model is not None
        results = list(self._model.passage_embed(texts))
        out = []
        for emb in results:
            pairs = sorted(zip(emb.indices.tolist(), emb.values.tolist()))
            out.append(
                SparseVector(
                    indices=[p[0] for p in pairs],
                    values=[p[1] for p in pairs],
                )
            )
        return out

    def encode_query(self, text: str) -> SparseVector:
        """Encode a query (BM25 query-side weights — different from passage)."""
        self._load()
        assert self._model is not None
        results = list(self._model.query_embed(text))
        emb = results[0]
        pairs = sorted(zip(emb.indices.tolist(), emb.values.tolist()))
        return SparseVector(
            indices=[p[0] for p in pairs],
            values=[p[1] for p in pairs],
        )

    @property
    def model_name(self) -> str:
        return self.MODEL_NAME


# Process-level singleton — loaded once, reused across calls.
_bm25_encoder: BM25SparseEncoder | None = None


def get_bm25_encoder() -> BM25SparseEncoder:
    """Return the process-level BM25 encoder, creating it on first call."""
    global _bm25_encoder
    if _bm25_encoder is None:
        _bm25_encoder = BM25SparseEncoder()
    return _bm25_encoder
