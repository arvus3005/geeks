"""Reranking module — placeholder for cross-encoder reranker integration."""
from abc import ABC, abstractmethod


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, passages: list[dict], top_k: int = 10) -> list[dict]:
        ...


class IdentityReranker(BaseReranker):
    """Pass-through reranker (preserves retrieval order). Used in smoke mode."""

    def rerank(self, query: str, passages: list[dict], top_k: int = 10) -> list[dict]:
        return passages[:top_k]


def get_reranker(mode: str = "identity") -> BaseReranker:
    if mode == "identity":
        return IdentityReranker()
    raise ValueError(f"Unknown reranker mode: {mode}")
