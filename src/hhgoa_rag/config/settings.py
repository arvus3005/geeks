from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Corpus mode — "full" here means the self-hosted local index, which as
    # of 2026-08-22 covers 6 of 14 MSMARCO-XI language configs (54.25M
    # passages) -- honestly "partial full corpus", not smoke/pilot and not
    # the complete 14-language target either. See README's indexing-status
    # section for the exact language list.
    corpus_mode: str = "full"  # "smoke" | "pilot" | "full"

    # Local self-hosted hybrid index (BM25 + HNSW, sharded) — the ONLY
    # retrieval backend, since 2026-08-22 (team decision). See
    # src/hhgoa_rag/retrieval/sharded_local_hybrid_store.py. No managed
    # vector DB is used anywhere in the serving path.
    local_index_root: str = "artifacts/full_local_index"

    # Retrieval
    retrieval_top_k: int = 20
    context_n: int = 3

    # STT
    sarvam_api_key: str | None = None
    sarvam_model: str = "saaras:v3"
    whisper_enabled: bool = False  # dev fallback only

    # Guardrails
    max_query_chars: int = 512
    min_retrieval_score: float = 0.45

    # Manifests
    index_manifest_id: str = "local-hybrid-sharded-6lang-v001"
    model_manifest_id: str = "e5-small-int8-onnx-query-v001"


def get_settings() -> Settings:
    return Settings()
