# Indexing Readiness Report

**Final verdict: READY TO START INDEXING**

**Generated:** 2026-08-16  
**Commit:** `010dd578d554223ec1dc91eadcd0b8bf800eb653`

---

## Git Status

```
On branch main
Clean working tree — all changes committed.
```

**Files changed in this pass:**

| File | Change |
|------|--------|
| `scripts/prepare_canary.py` | Rewrote: `--chunk-strategy` option, tokenizer SHA resolution, backfill, hf_hub_download + pyarrow for reliable nested-struct reading, dedup counts, batch projection fields |
| `scripts/ingest_prepared.py` | Rewrote: `--execute` flag, `CONFIRM_PINECONE_WRITE=1`, schema validation, 100/100/100 enforcement, reject individually oversized records, bounded retry |
| `scripts/ingest_all.py` | Block pilot/canary raw-HF writes → redirect to `ingest_prepared.py` |
| `scripts/ingest_shard.py` | Block pilot raw-HF writes → redirect to `ingest_prepared.py` |
| `scripts/resume_ingest.py` | Block pilot/canary checkpoint resumption → redirect to `ingest_prepared.py` |
| `tests/contract/test_ingestion_safety.py` | Updated 5 tests to reflect pilot-blocking behavior |
| `tests/unit/test_adversarial.py` | Added `actual_per_language_records` to minimal manifest helper |
| `tests/unit/test_canary_preparation.py` | **New**: 25 focused tests covering all 16 spec scenarios |

**Dependencies changed:** None.

---

## Quality Gates

### Tests
```
339 passed, 10 skipped
```

**Skip reasons (all expected):** Integration tests for real Pinecone smoke (need `PINECONE_API_KEY` and `PINECONE_SMOKE_TEST=1`).

### Ruff
```
All checks passed!
```

### Ruff Format
```
98 files already formatted
```

### Mypy
```
Success: no issues found in 47 source files
```

---

## Real Prepared Canary — Manifest Data

**Manifest ID:** `canary-42-d736ec01`  
**Manifest path:** `artifacts/prepared/canary-42-d736ec01_manifest.json`  
**JSONL path:** `artifacts/prepared/canary-42-d736ec01_records.jsonl`

| Field | Value |
|-------|-------|
| `ready_for_write` | **true** |
| Dataset repository | `ai4bharat/MSMARCO-XI` |
| Dataset revision SHA | `bf5cdc1f26e581e519018e434db14edd1b77602b` |
| `dataset_revision_pinned` | true |
| Hindi physical source | `train/hintrain.parquet` |
| Bengali physical source | `train/bentrain.parquet` |
| Source identity check | PASS (different files) |
| Tokenizer repository | `intfloat/multilingual-e5-large` |
| Tokenizer revision SHA | `3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3` |
| `tokenizer_revision_pinned` | true |
| Tokenizer fingerprint | `2616986da866a9dc` |
| Model input limit | 512 tokens |
| Chunking strategy | `sentence_aware` |
| Chunk strategy version | `v1` |
| Seed | 42 |

### Per-Language Record Totals

| Language | Records | Tokens |
|----------|---------|--------|
| English (`en`) | **100** | 8,253 |
| Hindi (`hi`) | **100** | 10,495 |
| Bengali (`bn`) | **100** | 10,141 |
| **Total** | **300** | **28,889** |

### Deduplication Counts

| Type | Count |
|------|-------|
| English cross-file dedup | 50,264 |
| Hindi within-file dedup | 374 |
| Bengali within-file dedup | 354 |
| Passages split (oversized) | 0 |
| Passages rejected | 0 |

### File Checksums

| File | SHA-256 |
|------|---------|
| JSONL | `8f6d76470d36f671018177ccef9888bbfc52b97dd410f3b32beeb37c33f32704` |
| Logical data checksum | `05f44c15dbe4e4295a04d0e6fcaa5e0539d1f1b035e87b6cb6e4c86b33ba5274` |
| Manifest | `fd782315242c0fdde23e4e116b653f8840fef5d2b10deaf2949d3a74a081a341` |

### Batch Projections

| Metric | Value |
|--------|-------|
| Planned requests | 4 |
| Max records per request | 96 |
| Max serialized bytes per request | 1,800,000 |
| Projected indexed bytes | 450,000 |
| Actual max batch bytes | 153,337 |

### Starter Budget Projections

| Check | Status |
|-------|--------|
| Records ≤ 10,000 | ✓ (300) |
| Tokens ≤ 4,000,000 | ✓ (28,889) |
| Storage ≤ 1.5 GB | ✓ (450 KB) |
| **Overall** | **PASS** |

### Forbidden Field Audit
```
PASS — no forbidden fields in any prepared record
```

### `readiness_failures`
```json
[]
```

---

## Dry-Run Validation

```
uv run python scripts/ingest_prepared.py \
  --manifest artifacts/prepared/canary-42-d736ec01_manifest.json \
  --dry-run
```

**Output:**
```
Manifest loaded: id=canary-42-d736ec01 revision=bf5cdc1f26e581e519018e434db14edd1b77602b records=300
Data file verified: artifacts/prepared/canary-42-d736ec01_records.jsonl
Loaded and validated 300 records from data file
Built 4 batches from 300 records

INGESTION PLAN (dry-run — no Pinecone calls)
  Manifest ID       : canary-42-d736ec01
  Dataset revision  : bf5cdc1f26e581e519018e434db14edd1b77602b
  Total records     : 300
  Total tokens      : 28,889
  Batches           : 4
  Max batch records : 96
  Max batch bytes   : 1,800,000
  Per-language counts: bn: 100, en: 100, hi: 100
  Batch 001: 96 records, 99,853 bytes
  Batch 002: 96 records, 153,337 bytes
  Batch 003: 96 records, 145,094 bytes
  Batch 004: 12 records, 17,691 bytes
  Max actual batch bytes : 153,337
  Ready for write   : True

DRY RUN complete — no records were written.
```

**No Pinecone client was constructed.**

---

## Proof: Live Pilot Ingestion Consumes Only Prepared Records

`ingest_prepared.py` enforces:
1. `--manifest` required (points to a `prepare_canary.py`-generated file)
2. Validates manifest checksum before reading any records
3. Validates JSONL checksum against manifest
4. Validates every record with `validate_record()` from authoritative schema
5. Enforces exactly 100 English, 100 Hindi, 100 Bengali records
6. Never calls HuggingFace APIs
7. Never streams dataset independently

Legacy scripts are blocked:
- `ingest_all.py --mode pilot` → exits 1, redirects to `ingest_prepared.py`
- `ingest_shard.py --mode pilot` → exits 1, redirects to `ingest_prepared.py`
- `resume_ingest.py` with pilot checkpoint → exits 1, redirects to `ingest_prepared.py`

---

## Proof: Dry-Run Creates No Pinecone Client

`ingest_prepared.py` dry-run path (active when `--execute` is missing OR `CONFIRM_PINECONE_WRITE≠1`):
- `_is_live_mode(args)` returns `False`
- Exits via `sys.exit(0)` before the `from pinecone import Pinecone` line
- `PINECONE_API_KEY` is never read
- Verified by tests `test_dry_run_never_touches_pinecone` and `test_live_mode_requires_execute_and_env`

---

## Unresolved Blockers

**None for starting the bounded pilot.**

Full-corpus ingestion remains permanently blocked on Pinecone Starter (enforced in code before any API call).

---

## Confirmation: No Live Pinecone Operations During This Task

- No Pinecone index was created ✓
- No Pinecone records were written ✓
- No Pinecone searches or reranking occurred ✓
- No paid provider quota was consumed ✓
- No credentials were printed, stored, or committed ✓

---

## Exact Live Commands (Using Actual Generated Manifest)

### 1. Export a newly rotated API key

```bash
export PINECONE_API_KEY="<your-rotated-api-key>"
```

### 2. Create the integrated `multilingual-e5-large` index

```bash
CONFIRM_PINECONE_CREATE=1 uv run python scripts/create_pinecone_index.py \
  --pinecone-index msmarco-xi \
  --embed-model multilingual-e5-large \
  --execute
```

### 3. Validate the index

```bash
uv run python scripts/validate_pinecone_config.py \
  --pinecone-index msmarco-xi \
  --embed-model multilingual-e5-large
```

### 4. Run bounded smoke integration tests

```bash
PINECONE_API_KEY=$PINECONE_API_KEY PINECONE_SMOKE_TEST=1 \
  uv run pytest tests/integration/test_pinecone_smoke.py -v
```

### 5. Ingest the prepared 300-record pilot

```bash
PINECONE_API_KEY=$PINECONE_API_KEY CONFIRM_PINECONE_WRITE=1 \
  uv run python scripts/ingest_prepared.py \
  --manifest artifacts/prepared/canary-42-d736ec01_manifest.json \
  --namespace pilot_canary_v1 \
  --execute
```

> **Dry-run first (no credentials needed):**
> ```bash
> uv run python scripts/ingest_prepared.py \
>   --manifest artifacts/prepared/canary-42-d736ec01_manifest.json \
>   --dry-run
> ```
