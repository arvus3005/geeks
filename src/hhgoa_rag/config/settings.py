from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Corpus mode
    corpus_mode: str = "smoke"  # "smoke" | "pilot" | "full"

    # Pinecone
    pinecone_api_key: str | None = None
    pinecone_index: str = "msmarco-xi"
    pinecone_namespace: str = "smoke"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_embed_model: str = "multilingual-e5-large"

    # Retrieval
    retrieval_top_k: int = 20
    context_n: int = 3

    # STT
    sarvam_api_key: str | None = None
    sarvam_model: str = "saaras:v2"
    whisper_enabled: bool = False  # dev fallback only

    # Timeouts (ms)
    pinecone_search_timeout_ms: int = 10000
    pinecone_upsert_timeout_ms: int = 30000

    # Guardrails
    max_query_chars: int = 512
    min_retrieval_score: float = 0.45

    # Manifests
    index_manifest_id: str = "smoke-v001"
    model_manifest_id: str = "pinecone-multilingual-e5-large-v001"


def get_settings() -> Settings:
    return Settings()
