# MSMARCO-XI → Pinecone: Project Summary

**Date:** 2026-08-16
**Status: CODE READY FOR GEMINI CANARY EXECUTION**
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
| `physical_shard` | str | Real physical source parquet path (e.g. `train/hintrain.parquet`) |
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
- `scripts/index_canary.py` is the ONLY live ingestion path. It performs the
  empty-namespace preflight, exact resume ownership, atomic per-batch
  checkpointing, and post-write reconciliation that requires exact ID-set equality.
- `scripts/ingest_prepared.py` is an offline validator/dry-run tool ONLY; its
  `--execute` path is DISABLED and exits non-zero before any Pinecone import,
  even with `CONFIRM_PINECONE_WRITE=1` and `PINECONE_API_KEY` present.
- `--execute` (canary indexer) requires `CONFIRM_PINECONE_WRITE=1` (exit 2 otherwise)
- Dry-run never goes live even with credentials + confirmation present
- No Pinecone import on any dry-run path
- Index-contract validation (`model`, `dimension`, `metric`, `field_map`, `write_parameters`, `read_parameters`)
- Namespace locked to `pilot_v1`

### Task 5 — CI isolation
- Offline CI job never receives `PINECONE_API_KEY`, `SARVAM_API_KEY`, or `ELEVENLABS_API_KEY`
- Writable HF cache configured in CI
- Live integration tests moved to `workflow_dispatch` only (manual trigger with `YES_RUN_LIVE` confirmation)
- Separate `live-integration` job; push/PR CI is unconditionally credential-free
- `scan_secrets.py` runs in normal offline CI before tests
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
| `ruff format --check .` | 106 files clean |
| `ruff check .` | All passed |
| `mypy src scripts` | 0 issues in 64 source files |
| `pytest tests/unit tests/contract tests/behavioural` (no credentials) | **557 passed, 0 failed, 0 skipped** |
| `scripts/scan_secrets.py .` | No committed secrets |

---

## Generated canary artifact

Artifacts are git-ignored. Regenerate after a fresh clone with:

```bash
uv run python scripts/prepare_canary.py \
    --dataset-revision bf5cdc1f26e581e519018e434db14edd1b77602b \
    --tokenizer-revision 3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3 \
    --seed 42 \
    --chunk-strategy sentence_aware
```

| Property | Value |
|----------|-------|
| Manifest ID | `canary-42-ee540c17772a` |
| Records | 300 (en=100, hi=100, bn=100) |
| JSONL SHA-256 | `ca912d133c3033eca71cc86045923e0165f5b43baba6f1951a1741ff0a3a9217` |
| JSONL bytes | 395,171 |
| Total tokens | 28,366 |
| Max token length | 211 (≤ 507 ✓) |
| Duplicate IDs | 0 |
| Forbidden fields | None |
| Planned requests | 4 × ≤96 records |
| `ready_for_write` | **true** — empty `readiness_failures` |
| Determinism | Two runs produced byte-identical JSONL and same manifest ID |

---

## Dry-run evidence

| Scenario | Result |
|----------|--------|
| `--dry-run` from repo root with live-looking credentials | Offline; printed plan; no Pinecone client constructed |
| `--dry-run` from `/tmp` (different CWD) | Resolved data via manifest-relative path; 4 batches; namespace `pilot_v1` |
| `ingest_prepared.py --execute` (even with confirm + key) | Exit 2 (legacy live path DISABLED; redirects to index_canary.py) |
| `index_canary.py --execute` without `CONFIRM_PINECONE_WRITE=1` | Exit 2 |
| Two `prepare_canary.py` runs into different dirs | Byte-identical JSONL and byte-identical manifest (same SHA-256) |

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

## Approved canary command for Gemini live execution

> Regenerate the artifact first if working from a fresh clone (command above).  
> The `msmarco-xi` integrated-embedding index must be created first with the contract above.

```bash
export PINECONE_API_KEY=<your-key>
CONFIRM_PINECONE_WRITE=1 \
  uv run python scripts/index_canary.py \
    --manifest artifacts/prepared/canary-42-ee540c17772a_manifest.json \
    --execute --resume --concurrency 4
```

*(Note: `scripts/ingest_prepared.py` is an offline validator/dry-run tool only; its `--execute` live path is DISABLED. `scripts/index_canary.py` is the sole approved live path, with empty-namespace preflight, atomic per-batch checkpointing, and post-write reconciliation requiring exact ID-set equality.)*
