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


class E5MultilingualEmbedder(BaseEmbedder):
    """Production embedder using multilingual-e5-small."""

    def __init__(self, model_id: str = "intfloat/multilingual-e5-small"):
        self._model_id = model_id
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_id)

    def embed_query(self, text: str) -> np.ndarray:
        self._load()
        prefixed = f"query: {text}"
        emb = self._model.encode([prefixed], normalize_embeddings=True)[0]
        return emb.astype(np.float32)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        self._load()
        prefixed = [f"passage: {t}" for t in texts]
        return self._model.encode(prefixed, normalize_embeddings=True).astype(np.float32)

    @property
    def dimension(self) -> int:
        return 384

    @property
    def model_id(self) -> str:
        return self._model_id
