from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Corpus mode
    corpus_mode: str = "smoke"  # "smoke" or "full"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection_alias: str = "msmarco_xi_passages_current"
    qdrant_collection_physical: str = "msmarco_xi_passages_v001"

    # Embedding
    embedding_model_id: str = "intfloat/multilingual-e5-small"
    embedding_dim: int = 384
    embedding_normalize: bool = True

    # Retrieval
    dense_prefetch_k: int = 32
    sparse_prefetch_k: int = 32
    fused_k: int = 20
    context_n: int = 3

    # STT
    sarvam_api_key: str | None = None
    sarvam_model: str = "saaras:v2"
    whisper_enabled: bool = False  # dev fallback only

    # Timeouts (ms)
    qdrant_connect_timeout_ms: int = 5000
    qdrant_read_timeout_ms: int = 10000
    embed_timeout_ms: int = 30000

    # Guardrails
    max_query_chars: int = 512
    min_retrieval_score: float = 0.45

    # Manifests
    index_manifest_id: str = "smoke-v001"
    model_manifest_id: str = "multilingual-e5-small-full-precision-v001"


def get_settings() -> Settings:
    return Settings()
