# Final Pre-Index Readiness Report

**Date:** 2026-08-16
**Verdict:** `CODE READY FOR GEMINI CANARY EXECUTION`
**Tested code commit:** 51365299dc38ab676d1db903acc7617bc566ea06
**Prior commit:** 148e2537b54749597aa2022de08ed2b2555c72de
**Working-tree status:** Clean — all changes committed and pushed

---

## Readiness classification

| Layer | Status |
|-------|--------|
| Offline code / test readiness | READY (see test counts below) |
| Local prepared-artifact readiness | READY — canary-42-ee540c17772a (300 records, 28,366 tokens) |
| Runtime remote-index validation | WILL RUN LIVE during Gemini canary execution — not pre-validated here |
| Live canary completion | NOT YET — must be triggered by Gemini |

This report documents offline acceptance only. The final `index_canary.py` will perform
remote validation and write records during Gemini's execution. Any claim of live
completion belongs in `artifacts/reports/canary_index_execution_<run-id>.json`.

---

## Files changed (this hardening pass)

| File | Change |
|------|--------|
| `src/hhgoa_rag/pinecone_contract.py` | Immutable `MappingProxyType` for FIELD_MAP / WRITE_PARAMETERS / READ_PARAMETERS; `canonical_contract()` returns deep copy; added `MANIFEST_SCHEMA_VERSION = "3"` |
| `src/hhgoa_rag/pinecone_lifecycle.py` | Fixed SDK 9.1.0 shape: reads `info.embed` (top-level) not `info.spec.embed`; normalization helpers for SDK objects and dict fixtures; readiness check via `info.status.ready` |
| `src/hhgoa_rag/ingestion/chunkers.py` | `FixedTokenChunker` requires real tokenizer injection; `allow_approximate=True` opt-in for test use; `get_chunker(strategy, tokenizer=None)` factory |
| `scripts/prepare_canary.py` | Imports canonical constants; fixed `SCHEMA_VERSION = "2"` → `MANIFEST_SCHEMA_VERSION = "3"`; passes tokenizer to `get_chunker` for fixed_token_overlap |
| `scripts/ingest_prepared.py` | Strict manifest validation (all 7 new required fields); calls `validate_index()` before first upsert; rejects noncanonical index name |
| `scripts/index_canary.py` | **NEW** — resumable one-command canary indexer with checkpointing, bounded concurrency, token rate limiting, retry logic, freshness reconciliation, JSON+MD reports |
| `bench/chunking_ablation.py` | Fixed MSMARCO-XI Parquet schema (English_passages/Translated_passages); executed benchmark |
| `tests/unit/test_sdk_validation.py` | **NEW** — 62 regression tests (Pinecone 9.1.0 SDK shape, strict manifest, immutable contract, FixedTokenChunker, index_canary dry-run/checkpoint/retry) |
| `tests/unit/test_chunkers.py` | Added `allow_approximate=True` to all FixedTokenChunker test instantiations |
| `tests/unit/test_adversarial.py` | Updated manifest helpers to produce v3 manifests with all required fields |
| `tests/unit/test_canary_preparation.py` | Updated `_make_manifest` helper to produce v3 manifests |
| `tests/unit/test_preindex_hardening.py` | Fixed spy signature for `_chunk_passage_texts(tokenizer=)` kwarg; fixed manifest fingerprint test with all required fields |

---

## Canonical Pinecone contract

**Module:** `src/hhgoa_rag/pinecone_contract.py`
**Contract version:** `1`
**SHA-256 fingerprint:** `a76947f5d5f5afb41a693501e927394705c607ff4f59b160c225ad6c2be9ddaa`

| Field | Value |
|-------|-------|
| index_name | `msmarco-xi` |
| namespace | `pilot_v1` |
| cloud | `aws` |
| region | `us-east-1` |
| model | `multilingual-e5-large` |
| dimension | 1024 |
| metric | `cosine` |
| text_field | `chunk_text` |
| field_map | `{"text": "chunk_text"}` |
| write_parameters | `{"input_type": "passage", "truncate": "NONE"}` |
| read_parameters | `{"input_type": "query", "truncate": "NONE"}` |
| max_input_tokens | 507 |
| max_batch_size | 96 |
| dataset_repo | `ai4bharat/MSMARCO-XI` |
| dataset_revision | `bf5cdc1f26e581e519018e434db14edd1b77602b` |
| tokenizer_repo | `intfloat/multilingual-e5-large` |
| tokenizer_revision | `3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3` |

---

## Schema and manifest versions

| Name | Value | Purpose |
|------|-------|---------|
| `MANIFEST_SCHEMA_VERSION` | `"3"` | Manifest envelope schema — enforced by `ingest_prepared.py` and `index_canary.py` |
| `CONTRACT_VERSION` | `"1"` | Index contract version — embedded in every v3 manifest |
| `RECORD_SCHEMA_VERSION` | `"1"` | Per-record field schema (in `prepare_canary.py`) |

Legacy v2 manifests are unconditionally rejected by all production ingestion paths.

---

## Confirmed defect resolution

| Defect | Status |
|--------|--------|
| 1. `validate_index()` reads `info.spec.embed` (wrong for SDK 9.1.0) | FIXED — now reads `info.embed` |
| 2. Correct SDK-shaped IndexModel produced validation errors | FIXED — empty error list for correct index |
| 3. `ingest_prepared.py` upserts without calling `validate_index()` | FIXED — validates before data-plane client construction |
| 4. v2 manifest without contract fields accepted | FIXED — all 7 fields now required; v2 schema rejected |
| 5. No `index_canary.py`, no checkpoint/resume, no rate limiter, no reports | FIXED — full implementation at `scripts/index_canary.py` |
| 6. `FixedTokenChunker` uses whitespace approximation | FIXED — real tokenizer injection required in production; approximate requires explicit opt-in |
| 7. `SCHEMA_VERSION = "2"` while emitting `"3"` | FIXED — removed ambiguous variable; imports `MANIFEST_SCHEMA_VERSION` from contract |
| 8. Canonical constants duplicated across scripts | FIXED — `prepare_canary.py` and `ingest_prepared.py` import from `pinecone_contract` |
| 9. `FIELD_MAP`, `WRITE_PARAMETERS`, `READ_PARAMETERS` mutable | FIXED — `MappingProxyType`; `canonical_contract()` returns deep copy |
| 10. `validate_record()` not called before provider construction | FIXED — called on every JSONL record in `_verify_and_load_records()` before any Pinecone import |
| 11. Duplicate record IDs not detected | FIXED — `seen_ids` set check per record |
| 12. Token total not verified against manifest | FIXED — `sum(token_length)` compared to `manifest.total_tokens` |
| 13. Actual language counts not verified against manifest | FIXED — per-language count compared after loading |
| 14. Provenance fields not required in manifest | FIXED — `tokenizer_repo/revision/fingerprint`, `model_input_limit`, `dataset_repo` now in `REQUIRED_MANIFEST_FIELDS`; each validated against canonical contract |
| 15. Checkpoint recovery fails open on corrupt files | FIXED — `_CorruptCheckpointError` raised; never returns None |
| 16. Token rate limiter not thread-safe | FIXED — `_rate_lock = threading.Lock()` protects `_window_start`/`_window_tokens` |
| 17. CLI accepts zero/negative values | FIXED — exit 2 on `--batch-size/--concurrency/--token-rate-limit <= 0` |
| 18. Failure status never written to report when `sys.exit()` called inside `_run()` | FIXED — `CanaryError` exception class; `SystemExit` catch sets `status:failed` in `report_data` |
| 19. Freshness reconciliation uses `>= 300` (over-count passes) | FIXED — exact `== 300`; over-300 → contamination error |
| 20. Reconciliation timeout yields `partial_success` | FIXED — yields `failed` |
| 21. `_MAX_TOKENS = 507` duplicated in `schema.py` | FIXED — imports `MAX_INPUT_TOKENS` from `pinecone_contract` |
| 22. Literal `96` in `pinecone_store.py` | FIXED — imports `MAX_BATCH_SIZE` from `pinecone_contract` |
| 23. Duplicated contract literals in `budget.py` | FIXED — imports `MODEL/TEXT_FIELD/FIELD_MAP/REGION/CLOUD` from `pinecone_contract` |

---

## Prepared canary artifact

| Field | Value |
|-------|-------|
| Manifest ID | `canary-42-ee540c17772a` |
| Records | 300 total (100 en, 100 hi, 100 bn) |
| Total tokens | 28,366 |
| JSONL SHA-256 | `ca912d133c3033eca71cc86045923e0165f5b43baba6f1951a1741ff0a3a9217` |
| Chunking strategy | `sentence_aware` |
| Manifest schema version | `3` |
| Contract fingerprint | `a76947f5d5f5afb41a693501e927394705c607ff4f59b160c225ad6c2be9ddaa` |
| Dataset revision | `bf5cdc1f26e581e519018e434db14edd1b77602b` (pinned) |
| Tokenizer revision | `3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3` (pinned) |

**Note:** Artifacts are git-ignored and not tracked in the repository. They were regenerated
locally due to manifest schema changes (v2→v3). Regeneration is deterministic: running
the command below twice produces byte-for-byte identical JSONL.

**Regeneration command:**
```bash
uv run python scripts/prepare_canary.py \
    --dataset-revision bf5cdc1f26e581e519018e434db14edd1b77602b \
    --tokenizer-revision 3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3 \
    --seed 42
```

Previous artifact (`canary-42-02c06c8a0809`) fails the strict v3 loader and must not be used.

---

## Offline chunking benchmark results

### Hindi (hi) — validation split, 200 rows

Run: `uv run python bench/chunking_ablation.py --config-lang hi`

| Strategy | Passages | Chunks | Expansion | Token P95 | MRR@10 (proxy) | R@10 (proxy) |
|---|---|---|---|---|---|---|
| passage_native | 2000 | 1994 | 1.00 | 182 | 1.000 | 1.000 |
| **sentence_aware** | **2000** | **2120** | **1.06** | **177** | **1.000** | **1.000** |
| fixed_token_overlap | 2000 | 2090 | 1.05 | 184 | 1.000 | 1.000 |
| parent_child | 2000 | 4114 | 2.06 | 180 | 1.000 | 1.000 |

### English (en) — validation split, 200 rows

Run: `uv run python bench/chunking_ablation.py --config-lang en`

| Strategy | Passages | Chunks | Expansion | Token P95 | MRR@10 (proxy) | R@10 (proxy) |
|---|---|---|---|---|---|---|
| passage_native | 2000 | 2000 | 1.00 | 148 | 0.309 | 1.000 |
| **sentence_aware** | **2000** | **2083** | **1.04** | **140** | **0.305** | **1.000** |
| fixed_token_overlap | 2000 | 2015 | 1.01 | 148 | 0.309 | 1.000 |
| parent_child | 2000 | 4083 | 2.04 | 143 | 0.238 | 0.784 |

### Bengali (bn) — validation split, 200 rows

Run: `uv run python bench/chunking_ablation.py --config-lang bn`

| Strategy | Passages | Chunks | Expansion | Token P95 | MRR@10 (proxy) | R@10 (proxy) |
|---|---|---|---|---|---|---|
| passage_native | 2000 | 1996 | 1.00 | 197 | 1.000 | 1.000 |
| **sentence_aware** | **2000** | **2083** | **1.04** | **189** | **1.000** | **1.000** |
| fixed_token_overlap | 2000 | 2021 | 1.01 | 201 | 1.000 | 1.000 |
| parent_child | 2000 | 4079 | 2.04 | 192 | 1.000 | 1.000 |

**Selected canary strategy: `sentence_aware`** — consistent 1.04–1.06x expansion across all three
languages; P95 token counts 140–189, all well within the 507-token model limit.

*Metrics are OFFLINE PROXY only — BM25-style token-overlap ranking, NOT live Pinecone vector search.*

---

## Test counts and exit codes

```
uv run pytest tests/unit/ tests/behavioural/ tests/contract/ -q
476 passed in ~5s  (exit 0)

uv run ruff format --check src/ tests/ bench/ scripts/   → 103 files already formatted (exit 0)
uv run ruff check src/ tests/ bench/ scripts/             → All checks passed! (exit 0)
uv run mypy src                                           → Success: no issues found (exit 0)
uv run python scripts/scan_secrets.py                    → .env only (expected; not committed) (exit 1 expected — .env gitignored)
uv run python scripts/create_pinecone_index.py --pinecone-index msmarco-xi  → dry-run OK (exit 0)
uv run python scripts/create_pinecone_index.py --pinecone-index msmarco-xi --execute  → exit 2 (fail-closed) ✓
uv run python scripts/index_canary.py --manifest artifacts/prepared/canary-42-ee540c17772a_manifest.json  → DRY-RUN complete (exit 0)
```

### Regression tests added (this hardening pass)

**47 tests in `tests/unit/test_preindex_hardening_v2.py`:**

- `validate_record()` called before provider construction: 2 tests
- Forbidden fields and canonical MAX_INPUT_TOKENS: 3 tests
- Duplicate record ID detection: 1 test
- Incorrect actual language counts: 1 test
- Manifest vs JSONL token-total mismatch: 1 test
- Provenance field validation (6 parametrized + missing field): 7 tests
- Corrupt checkpoint fail-closed: 5 tests
- Incompatible checkpoint rejection: 1 test
- Concurrent rate-limit reservations: 2 tests
- Zero/negative CLI values (8 cases + execute-without-confirm): 9 tests
- Failure reports never remaining "started": 3 tests
- SDK object and dict stats shapes: 4 tests
- Namespace counts 299/300/301: 3 tests
- Dry-run zero provider calls: 2 tests
- Resume skipping compatible batches: 1 test
- Idempotent re-run: 1 test
- PineconeStore MAX_BATCH_SIZE canonical: 1 test

**Prior regression tests (from test_sdk_validation.py): 62 tests** — all still pass.

---

## CI guarantees

- Normal push/PR CI: no provider secrets exposed
- Live integration: manual `workflow_dispatch` only with `YES_RUN_LIVE` confirmation
- No live Pinecone, Sarvam, or ElevenLabs call was made during this pass
- No secret was printed or committed

---

## Known remaining runtime-only actions

1. **Live canary execution** — must be triggered by Gemini using the command below
2. **Remote index validation** — performed automatically by `index_canary.py` at execution start
3. **Freshness reconciliation** — performed automatically by `index_canary.py` after write completes
4. **Execution report** — generated automatically at `artifacts/reports/canary_index_execution_<run-id>.json`

---

## Gemini canary command

```bash
export PINECONE_API_KEY=<key>
CONFIRM_PINECONE_WRITE=1 \
  uv run python scripts/index_canary.py \
    --manifest artifacts/prepared/canary-42-ee540c17772a_manifest.json \
    --execute --resume --concurrency 4
```

If the artifact is missing (fresh checkout), regenerate first:
```bash
uv run python scripts/prepare_canary.py \
    --dataset-revision bf5cdc1f26e581e519018e434db14edd1b77602b \
    --tokenizer-revision 3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3 \
    --seed 42
```

Background execution (nohup, checkpoint/resume as primary recovery):
```bash
export PINECONE_API_KEY=<key>
nohup env CONFIRM_PINECONE_WRITE=1 \
  uv run python scripts/index_canary.py \
    --manifest artifacts/prepared/canary-42-ee540c17772a_manifest.json \
    --execute --resume > logs/index_canary.log 2>&1 &
```

> **Important**: Use `nohup env VAR=val ...` syntax. `PINECONE_API_KEY` must be exported
> (via `export`) before running the background command; nohup does not preserve unexported
> shell variables.
