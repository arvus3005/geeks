# Qdrant Collection Schema

## Collection Naming Convention

| Pattern | Purpose | Destructive force allowed? |
|---------|---------|--------------------------|
| `msmarco_xi_passages_smoke_v*` | CI/test smoke runs | YES |
| `msmarco_xi_passages_pilot_v*` | Pilot subset | YES |
| `msmarco_xi_passages_v*` | Production | NO |
| `msmarco_xi_passages_current` | Serving alias | NO |

## Vector Configuration

### Dense Vector (`dense`)
- **Dimension**: 384
- **Distance**: Cosine
- **Model**: `intfloat/multilingual-e5-small`
- **Input prefix**: `"passage: {text}"` for indexing, `"query: {text}"` for queries
- **HNSW**: m=16, ef_construct=200

### Sparse Vector (`sparse`)
- **Encoding**: SHA-256-based stable token IDs (TF-normalized)
- **Production target**: FastEmbed BM25
- **IDF modifier**: applied at collection level

## Point Payload Schema

```json
{
  "text": "NFC-normalized passage text",
  "language": "en|as|bn|gu|hi|kn|ml|mr|ne|or|pa|sa|ta|te|ur",
  "content_hash": "sha256hex of normalized text",
  "chunk_strategy": "passage_native|fixed_token_overlap",
  "chunk_strategy_version": "passage_native_v1",
  "chunk_ordinal": 0,
  "source_split": "train|validation|smoke",
  "index_manifest_id": "smoke-v001|pilot-v001|..."
}
```

## Point ID

UUIDv5 computed from:
```
f"{dataset_revision}|{language}|{content_hash}|{chunk_strategy_version}|{chunk_ordinal}"
```

This is **stable across processes and runs** — same passage always gets same ID.

## Payload Indexes

| Field | Type |
|-------|------|
| `language` | KEYWORD |
| `chunk_strategy` | KEYWORD |
| `content_hash` | KEYWORD |

## Alias Management

The serving alias `msmarco_xi_passages_current` points to the active production collection.
Alias switches are atomic (delete + create in one operation) and require validation to pass first.
