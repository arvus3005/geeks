# Pinecone Record Schema — MSMARCO-XI (Authoritative)

> This schema is enforced by `src/hhgoa_rag/ingestion/schema.py`.
> All ingestion paths use `build_record()` and `validate_record()`.
> No Pinecone index or records exist yet — schema is for the upcoming canary.


## Index configuration
| Property       | Value                      |
|---------------|----------------------------|
| Plan           | Pinecone Starter           |
| Cloud          | AWS                        |
| Region         | us-east-1                  |
| Index type     | Serverless (integrated embedding) |
| Embed model    | `multilingual-e5-large`    |
| Text field     | `chunk_text`               |
| Field map      | `{"text": "chunk_text"}`   |

## Record fields (safe production metadata)
| Field                    | Type   | Description                                      |
|--------------------------|--------|--------------------------------------------------|
| `id`                     | str    | Deterministic UUIDv5 (dataset rev + lang + hash + strategy + ordinal) |
| `chunk_text`             | str    | The embedded passage chunk                       |
| `language`               | str    | Passage language code (e.g. "hi", "en")          |
| `config_language`        | str    | MSMARCO-XI config identifier                     |
| `dataset_revision`       | str    | Pinned HuggingFace dataset commit hash           |
| `split`                  | str    | "train" or "validation"                          |
| `physical_shard`         | str    | Physical source shard identifier                 |
| `local_source_row`       | int    | Row index within the physical shard              |
| `passage_position`       | int    | Position of passage in original record           |
| `parent_passage_id`      | str    | Content hash of source passage                   |
| `content_hash`           | str    | SHA-256 of normalised passage text               |
| `chunk_strategy`         | str    | Chunker name (e.g. "passage_native")             |
| `chunk_strategy_version` | str    | Chunker version string (e.g. "v1")               |
| `chunk_ordinal`          | int    | 0-based position of chunk within passage         |
| `chunk_total`            | int    | Total chunks emitted from this passage           |
| `token_length`           | int    | Exact token count from `intfloat/multilingual-e5-large` tokenizer (includes prefix + special tokens) |
| `tokenizer_fingerprint`  | str    | Tokenizer revision/fingerprint used              |
| `manifest_id`            | str    | Ingestion manifest identifier                    |

## Forbidden fields — MUST NEVER enter Pinecone records
The following fields come from MSMARCO-XI evaluation annotations and are strictly forbidden
from production records, dense vector inputs, and reranking features:

- `query`
- `Answer`
- `Eng_Query`
- `Eng_Answer`
- `query_type`
- `is_selected`
- Any relevance or evaluation label

These fields may only appear in isolated offline evaluation fixtures with separate storage.
Contract tests enforce this boundary.

## English deduplication
- One Pinecone record per unique English chunk.
- Source occurrence mappings (which passage contributed each unique chunk) are stored locally
  in SQLite or Parquet — NOT in Pinecone records.
- The `content_hash` field serves as the canonical provenance link.
- Pinecone records store occurrence count only (not a full provenance array).

## Starter plan limits (operational ceilings)
| Limit              | Hard cap  | Operational ceiling |
|--------------------|-----------|---------------------|
| Storage            | 2 GB      | 1.5 GB              |
| Embedding tokens   | 5M/month  | 4M (ingestion)      |
| Records            | —         | 10,000              |
| Rerank requests    | 500/month | Reserved for judging|
