# Pinecone Index Schema — MSMARCO-XI

**Source of truth:** `src/hhgoa_rag/pinecone_contract.py`

All lifecycle, ingestion, validation, and reporting code imports constants from
`pinecone_contract.py`. No hand-written contract dicts are permitted elsewhere.

---

## Index contract

| Property | Value |
|----------|-------|
| Contract version | `1` |
| Index name | `msmarco-xi` |
| Namespace | `pilot_v1` |
| Cloud | `aws` |
| Region | `us-east-1` |
| Model | `multilingual-e5-large` |
| Dimension | 1024 |
| Metric | `cosine` |
| Text field | `chunk_text` |
| Field map | `{"text": "chunk_text"}` |
| Write parameters | `{"input_type": "passage", "truncate": "NONE"}` |
| Read parameters | `{"input_type": "query", "truncate": "NONE"}` |
| Max input tokens | 507 |
| Max batch size | 96 records/request |

Contract SHA-256 fingerprint (computed from `pinecone_contract.canonical_contract_json()`):
`a76947f5d5f5afb41a693501e927394705c607ff4f59b160c225ad6c2be9ddaa`

---

## Record schema

Every record MUST be constructed via `build_record()` and validated via
`validate_record()` in `src/hhgoa_rag/ingestion/schema.py`.

### Required fields (exact set — no additions permitted)

| Field | Type | Constraints |
|-------|------|-------------|
| `id` | str | UUIDv5 deterministic; ≤ 512 chars |
| `chunk_text` | str | Non-empty; the embedded text |
| `language` | str | One of `en`, `hi`, `bn` |
| `config_language` | str | MSMARCO-XI config identifier |
| `dataset_revision` | str | 40-char hex SHA |
| `split` | str | `train` or `validation` |
| `physical_shard` | str | Real parquet file path (e.g. `train/hintrain.parquet`) |
| `local_source_row` | int | Row index within the physical shard |
| `passage_position` | int | Position of passage in original record |
| `parent_passage_id` | str | Content hash of source passage |
| `content_hash` | str | SHA-256 of normalised passage text |
| `chunk_strategy` | str | Chunker name (e.g. `sentence_aware`) |
| `chunk_strategy_version` | str | Chunker version string (e.g. `v1`) |
| `chunk_ordinal` | int | 0-based; `0 <= ordinal < chunk_total` |
| `chunk_total` | int | Total chunks from this passage; ≥ 1 |
| `token_length` | int | Exact token count; 1 ≤ length ≤ 507 |
| `tokenizer_fingerprint` | str | Tokenizer revision fingerprint |
| `manifest_id` | str | Ingestion manifest identifier |

### Forbidden fields (recursively checked, never permitted)

```
query, Answer, Eng_Query, Eng_Answer, query_type, is_selected
```

These are offline evaluation annotations from the MSMARCO-XI HuggingFace dataset.
They must NEVER appear in Pinecone records, dense embeddings, sparse inputs,
or reranking features.

### Removed/unofficial fields

- `physical_source` — was an unofficial extra field appended after schema validation; removed. The real path is now in `physical_shard`.
- `dataset_repo` — corpus-level provenance; lives in the manifest only, not per-record.

---

## Manifest schema (version 3)

Manifests produced by `prepare_canary.py` now include:

| Field | Description |
|-------|-------------|
| `manifest_schema_version` | `"3"` |
| `contract_version` | Contract version (`"1"`) |
| `contract_fingerprint` | SHA-256 of canonical contract JSON |
| `index_contract` | Full canonical contract dict |
| `index_name` | `"msmarco-xi"` |
| `index_namespace` | `"pilot_v1"` |

`ingest_prepared.py` validates all of these before any Pinecone client construction.

---

## Validation gates

`validate_index()` in `pinecone_lifecycle.py` checks ALL canonical fields:
- embed model
- field_map
- metric
- write_parameters (input_type, truncate)
- read_parameters (input_type, truncate)
- dimension
- cloud
- region

Any field that cannot be verified produces an explicit "unverifiable contract field" error.
There are no silent passes.
