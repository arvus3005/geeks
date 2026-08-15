"""Local embedder utilities — test/development only.

Production retrieval uses Pinecone integrated server-side embedding.
FakeEmbedder is kept for unit/behavioural tests that need deterministic vectors.
"""

import hashlib
from abc import ABC, abstractmethod

import numpy as np


class BaseEmbedder(ABC):
    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray: ...

    @abstractmethod
    def embed_passages(self, texts: list[str]) -> np.ndarray: ...

    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @property
    @abstractmethod
    def model_id(self) -> str: ...


class FakeEmbedder(BaseEmbedder):
    """Deterministic fake embedder — stable across processes (no built-in hash())."""

    _DIM = 384

    @staticmethod
    def _seed(text: str) -> int:
        return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)

    def embed_query(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(self._seed(text))
        v = rng.standard_normal(self._DIM).astype(np.float32)
        return v / np.linalg.norm(v)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.embed_query(t) for t in texts])

    @property
    def dimension(self) -> int:
        return self._DIM

    @property
    def model_id(self) -> str:
        return "fake-embedder-v0"
