from pydantic import BaseModel
from typing import Literal


class CorpusManifest(BaseModel):
    manifest_id: str
    corpus_mode: Literal["smoke", "full"]
    embedding_model_id: str
    embedding_dim: int
    collection_physical: str
    collection_alias: str
    total_source_rows: int
    total_passages: int
    unique_passages: int
    total_chunks: int
    rejected_records: int
    languages_covered: list[str]
    chunk_strategy: str
    chunk_strategy_version: str
    created_at: str
    git_commit: str | None
    warning: str | None = None
