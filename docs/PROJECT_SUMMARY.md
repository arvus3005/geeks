# MSMARCO-XI → Pinecone: Project Summary

**Date:** 2026-08-16  
**Status: READY FOR GEMINI INDEXING**  
No live Pinecone / Sarvam / ElevenLabs / WhisperFlow calls were made at any point.

---

## What this project does

Offline preparation and validation pipeline for indexing the MSMARCO-XI multilingual dataset
into a Pinecone Starter integrated-embedding index, ready for multilingual voice-based RAG retrieval.

---

## Fixed configuration (immutable)

| Property | Value |
|----------|-------|
| Pinecone plan | Starter |
| Cloud / Region | AWS / us-east-1 |
| Index name | `msmarco-xi` |
| Namespace | `pilot_v1` |
| Index type | Integrated dense embedding |
| Embedding model | `multilingual-e5-large` |
| Dimension | 1024 |
| Metric | cosine |
| Text field | `chunk_text` |
| Field map | `{"text": "chunk_text"}` |
| Write input type | `passage` |
| Read input type | `query` |
| Truncation | `NONE` (enforced offline) |
| Max token sequence | 507 |
| Max batch size | 96 records/request |
| Dataset | `ai4bharat/MSMARCO-XI` @ `bf5cdc1f26e581e519018e434db14edd1b77602b` |
| Tokenizer | `intfloat/multilingual-e5-large` @ `3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3` |
| Canary: split / seed / size | train / 42 / 300 (100 en + 100 hi + 100 bn) |
| Hindi source | `train/hintrain.parquet` |
| Bengali source | `train/bentrain.parquet` |
| STT model (config only) | `saaras:v3` (not implemented) |

---

## Dataset contract

**Source:** `ai4bharat/MSMARCO-XI` — 14 Indic language configs (`as`, `bn`, `gu`, `hi`, `kn`, `ml`, `mr`, `ne`, `or`, `pa`, `sa`, `ta`, `te`, `ur`) plus English.

**Passage types:**
- English passages — globally deduplicated across all 14 configs
- Translated passages — language-scoped dedup

**Leakage boundary — these fields MUST NEVER enter Pinecone records, embeddings or reranking:**

```
query, Answer, Eng_Query, Eng_Answer, query_type, is_selected
```

These are offline evaluation fields only. `validate_record()` enforces this recursively.

---

## Pinecone record schema

Every record is built via `build_record()` in `src/hhgoa_rag/ingestion/schema.py`.

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | Deterministic UUIDv5 (dataset rev + lang + hash + strategy + ordinal) |
| `chunk_text` | str | The embedded passage chunk |
| `language` | str | Passage language code (`en`, `hi`, `bn`, …) |
| `config_language` | str | MSMARCO-XI config identifier |
| `dataset_revision` | str | Pinned 40-char HuggingFace commit SHA |
| `split` | str | `train` or `validation` |
| `physical_source` | str | Physical source shard identifier |
| `local_source_row` | int | Row index within the physical shard |
| `passage_position` | int | Position of passage in original record |
| `parent_passage_id` | str | Content hash of source passage |
| `content_hash` | str | SHA-256 of normalised passage text |
| `chunk_strategy` | str | Chunker name (e.g. `sentence_aware`) |
| `chunk_strategy_version` | str | Chunker version string (e.g. `v1`) |
| `chunk_ordinal` | int | 0-based position of chunk within passage |
| `chunk_total` | int | Total chunks emitted from this passage |
| `token_length` | int | Exact token count under `multilingual-e5-large` |
| `tokenizer_fingerprint` | str | Tokenizer revision fingerprint |
| `manifest_id` | str | Ingestion manifest identifier |

**Schema validation enforces:**
- Allowed language codes
- 40-char hex dataset/tokenizer revisions
- `0 <= chunk_ordinal < chunk_total`
- ID length ≤ 512 chars
- Metadata ≤ 40 KB per record
- Positive token length, ≤ 507
- Recursive rejection of all forbidden/leakage fields
- No unknown fields

---

## What was built (all tasks)

### Task 1 — Safety infrastructure
- `StarterFullModeError`: full-corpus ingestion permanently blocked on Starter plan before any API call — no flag or env var can bypass
- Single-worker enforcement for canary/pilot modes
- `BudgetGuard`: checked before every write, committed only after Pinecone acknowledges
- `IngestCheckpoint`: includes `num_workers`; old checkpoints without it fail closed
- Dedup hashes committed to SQLite only after Pinecone ack
- Batch size capped at 96 records, validated at construction time

### Task 2 — Schema and tokenizer
- `src/hhgoa_rag/ingestion/schema.py`: authoritative `build_record()` and `validate_record()`
- `src/hhgoa_rag/ingestion/tokenizer.py`: real `intfloat/multilingual-e5-large` tokenizer via HuggingFace transformers (no torch); `MODEL_INPUT_LIMIT = 507`
- Forbidden-field recursive checker
- English deduplication via SHA-256 of NFC-normalised text

### Task 3 — Canary preparation pipeline
- `scripts/prepare_canary.py`: zero Pinecone imports; routes through `get_chunker()` registry; default strategy `sentence_aware`
- `_split_text_to_fit()`: sentence→word→char cascade, retains all sub-chunks, no text duplication or silent loss, rejects unsplittable content and backfills
- Materialises all candidate chunks first, then deterministically selects exactly 100 per language
- Every record built via `build_record()`; real `chunk_ordinal`/`chunk_total`
- `manifest_id` depends on all corpus-affecting inputs (dataset/tokenizer repos+revisions+fingerprint, model, 507 limit, split, sources, seed, quotas, normalization/schema versions, chunk strategy/version/params, text-field/field-map)
- `_download_and_read_parquet()` (formerly `_stream_parquet`): honest naming; downloads full shard before reading; includes disk-space preflight
- Portable JSONL path relative to manifest location
- Conservative storage estimate (not `total_records * 1500`)

### Task 4 — Ingest CLI (fail-closed)
- `scripts/ingest_prepared.py`: `--dry-run` and `--execute` mutually exclusive (exit 2)
- `--execute` requires `CONFIRM_PINECONE_WRITE=1` (exit 2 otherwise)
- `--dry-run` never goes live even with credentials + confirmation present
- No Pinecone import on any dry-run path
- Index-contract validation (`model`, `dimension`, `metric`, `field_map`, `write_parameters`, `read_parameters`)
- Namespace locked to `pilot_v1` unless explicitly overridden
- Manifest/index contract mismatch rejected

### Task 5 — CI isolation
- Offline CI job no longer receives `PINECONE_API_KEY` or `SARVAM_API_KEY`
- Writable HF cache configured in CI
- Separate opt-in `integration` job for live tests
- Tokenizer tests do not silently skip

### Task 6 — STT config update
- `saaras:v2` → `saaras:v3` in `settings.py` and `sarvam.py` (config/adapter only; STT not implemented)

### Task 7 — Tests
- `tests/unit/test_preindex_hardening.py`: 14 new tests
- Updated fixtures across `test_schema.py`, `test_adversarial.py`, `test_tokenizer.py`, `test_config.py`, `test_canary_preparation.py` for 507 limit, 40-char revisions, new signatures, robust missing-credential tests
- Coverage includes: chunker routing, split integrity, 507/508-token boundary, schema validation, flag safety, index contract, forbidden-field rejection, manifest identity, dry-run isolation

### Task 8 — Secret scanning
- `scripts/scan_secrets.py`: prints filenames only; only finding is `.env` (git-ignored)

---

## Verification gates (2026-08-16, real results)

| Gate | Result |
|------|--------|
| `ruff format --check src tests bench scripts` | 99 files clean |
| `ruff check src tests bench scripts` | All passed |
| `mypy src` | 0 issues in 47 source files |
| `pytest tests/unit tests/behavioural tests/contract` (no credentials) | **353 passed, 0 failed, 0 skipped** |
| `scripts/scan_secrets.py` | No committed secrets |

---

## Generated canary artifact

Artifacts are git-ignored. Regenerate after a fresh clone with:

```bash
uv run python scripts/prepare_canary.py \
    --dataset-revision bf5cdc1f26e581e519018e434db14edd1b77602b \
    --tokenizer-revision 3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3 \
    --seed 42
```

| Property | Value |
|----------|-------|
| Manifest ID | `canary-42-02c06c8a0809` |
| Records | 300 (en=100, hi=100, bn=100) |
| JSONL SHA-256 | `a580705f0ccb8bd4c0c66e19d94c79c9cb9401fee9752464bef2eb2452aafe17` |
| JSONL bytes | 414,371 |
| Total tokens | 28,366 |
| Max token length | 211 (≤ 507 ✓) |
| Duplicate IDs | 0 |
| Forbidden fields | None |
| Planned requests | 4 × ≤96 records |
| Projected indexed bytes | 2,136,122 |
| `ready_for_write` | **true** — empty `readiness_failures` |
| Determinism | Two runs produced byte-identical JSONL and same manifest ID |

---

## Dry-run evidence

| Scenario | Result |
|----------|--------|
| `--dry-run` from repo root with live-looking credentials | Offline; printed plan; no Pinecone client constructed |
| `--dry-run` from `/tmp` (different CWD) | Resolved data via manifest-relative path; 4 batches; namespace `pilot_v1` |
| `--dry-run --execute` together | Exit 2 (mutually exclusive) |
| `--execute` without `CONFIRM_PINECONE_WRITE=1` | Exit 2 |

---

## Confirmation of no live calls

No Pinecone index was created, listed, queried, upserted or deleted.  
No Sarvam / ElevenLabs / WhisperFlow calls were made.  
`.env` was not modified.  
Nothing was committed or pushed.  
Remote Pinecone state remains unverified by design.

---

## Remaining blockers

None for offline pre-index gates. Creating the `msmarco-xi` index and upserting
into `pilot_v1` are live actions out of scope for this pass — Gemini's responsibility.

---

## First command Gemini should run to begin live indexing

> Regenerate the artifact first if working from a fresh clone (command above).  
> The `msmarco-xi` integrated-embedding index must be created first with the contract above.

```bash
PINECONE_API_KEY="$PINECONE_API_KEY" CONFIRM_PINECONE_WRITE=1 \
  uv run python scripts/ingest_prepared.py \
    --manifest artifacts/prepared/canary-42-02c06c8a0809_manifest.json \
    --execute \
    --namespace pilot_v1
```
