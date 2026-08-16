# MSMARCO-XI Dataset Contract & Leakage Boundary Rules

**Dataset Repository:** `ai4bharat/MSMARCO-XI`  
**Pinned Dataset Revision:** `bf5cdc1f26e581e519018e434db14edd1b77602b`  
**Tokenizer Repository:** `intfloat/multilingual-e5-large`  
**Pinned Tokenizer Revision:** `3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3`  

---

## 1. Scope and Configuration

The `ai4bharat/MSMARCO-XI` dataset provides machine-translated multilingual passages based on MS MARCO for 14 Indic languages plus English:

- **14 Indic configurations:** `as`, `bn`, `gu`, `hi`, `kn`, `ml`, `mr`, `ne`, `or`, `pa`, `sa`, `ta`, `te`, `ur`
- **Target Canary configurations:** `hi` (`train/hintrain.parquet`) and `bn` (`train/bentrain.parquet`), which yield:
  - 100 English passages (globally deduplicated)
  - 100 Hindi passages
  - 100 Bengali passages
  - **300 total canary records**

---

## 2. Leakage Boundary — Forbidden Fields

The following fields from the raw HuggingFace Parquet dataset are **evaluation-only annotations** and MUST NEVER appear in indexed records, vector embeddings, payload metadata, or live retrieval contexts:

```
query
Answer
Eng_Query
Eng_Answer
query_type
is_selected
```

### Enforcement
- **`validate_record()` in `src/hhgoa_rag/ingestion/schema.py`:** Recursively inspects every record dictionary and raises `SchemaViolationError` if any forbidden key is detected.
- **`scripts/prepare_canary.py`:** Filters out forbidden keys before passage normalization and records a `forbidden_field_audit: "PASS"` in the manifest.
- **`scripts/index_canary.py`:** Runs `validate_record()` on every record before constructing any Pinecone client.

---

## 3. Provenance & Record Structure

Every indexed record must be constructed via `build_record()` in `src/hhgoa_rag/ingestion/schema.py` with:
- `id`: Deterministic UUIDv5 (`dataset_revision|language|content_hash|chunk_strategy_version|chunk_ordinal`)
- `chunk_text`: Normalized passage text (max 507 tokens)
- `language`: Passage language code (`en`, `hi`, `bn`, etc.)
- `config_language`: Source MSMARCO-XI configuration
- `physical_shard`: Repo-relative Parquet path (e.g., `train/hintrain.parquet`)
- `local_source_row`: 0-indexed row in the Parquet shard
- `parent_passage_id`: Content hash of the unchunked source passage
- `content_hash`: SHA-256 of the NFC-normalized chunk text
- `token_length`: Exact token count under `multilingual-e5-large` (1 ≤ length ≤ 507)
- `tokenizer_fingerprint`: Pinned tokenizer fingerprint
- `manifest_id`: ID of the generating manifest

---

## 4. Deduplication Rules

- **English passages:** English passages appear across all 14 language Parquet shards. They are globally deduplicated across shards using the SHA-256 hash of NFC-normalized text.
- **Translated passages:** Deduplicated within their language configuration.
