# Final Pre-Index Readiness Report

**Date:** 2026-08-16  
**Audited commit:** `378fd4f616665b9db56434b04490ab248d5fb2ef` (fetched as latest main from `https://github.com/arvus3005/geeks`)  
**Working-tree status:** Clean at audit baseline; all changes applied to working tree (not committed per task constraints)

---

## Files changed (this hardening pass)

| File | Change |
|------|--------|
| `src/hhgoa_rag/pinecone_contract.py` | **NEW** — canonical contract module |
| `src/hhgoa_rag/pinecone_lifecycle.py` | Rewrote validate_index() to cover ALL contract fields; added read/write_parameters to IndexEmbed; imports from contract module |
| `src/hhgoa_rag/ingestion/schema.py` | Added unknown-field rejection; added 507-token enforcement in validate_record() |
| `src/hhgoa_rag/api/app.py` | Updated validate_index() call to new signature |
| `scripts/create_pinecone_index.py` | --execute alone now exits 2 (fail-closed); dry-run shows full canonical contract; imports from contract module |
| `scripts/ingest_prepared.py` | Imports contract from canonical module; adds contract fingerprint/version/namespace/embedded-contract validation in _load_manifest() |
| `scripts/prepare_canary.py` | physical_shard = real parquet path (not "0"); removed physical_source unofficial field; SUPPORTED_CHUNK_STRATEGIES derived from real CHUNKERS registry; "semantic" removed; manifest schema version bumped to "3" with contract fields |
| `scripts/audit_ids.py` | Complete rewrite — comprehensive ID audit with duplicate/collision/ordinal/parent-linkage/content-dedup detection |
| `bench/chunking_ablation.py` | Replaced stub with real offline comparison (passage_native, sentence_aware, fixed_token_overlap, parent_child) |
| `.github/workflows/ci.yml` | Live integration moved to workflow_dispatch only with explicit YES_RUN_LIVE confirmation; secret scanner added to normal CI |
| `tests/unit/test_preindex_hardening.py` | Added 14 new tests covering contract, schema, strategy registry, physical_shard, create gate, eval label isolation, manifest fingerprint |
| `tests/contract/test_ingestion_safety.py` | Updated test_create_index_execute_alone: now asserts exit 2 (fail-closed), not exit 0 |
| `tests/unit/test_canary_preparation.py` | Removed physical_source and dataset_repo unofficial fields from _make_record() helper; physical_shard now holds real path |
| `docs/PROJECT_SUMMARY.md` | Updated schema table; updated CI isolation section |
| `docs/PINECONE_SCHEMA.md` | **NEW** — full index schema documentation |
| `docs/INGESTION_RUNBOOK.md` | **NEW** — 10-step handoff sequence |

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

- Record schema: unchanged required fields, new enforcement of unknown-field rejection and 507-token limit in validate_record()
- Manifest schema version: bumped `"2"` → `"3"` (adds contract_version, contract_fingerprint, index_contract, index_name, index_namespace)

---

## Blockers corrected

### 1. No canonical contract module (CRITICAL)
**Before:** Contract fields duplicated as hand-written dicts in ingest_prepared.py, create_pinecone_index.py, and implicitly in pinecone_lifecycle.py.  
**After:** Single source of truth in `pinecone_contract.py`. All paths import from it.

### 2. IndexEmbed missing read/write_parameters
**Before:** `create_index_idempotent()` constructed IndexEmbed without `write_parameters` or `read_parameters`.  
**After:** Explicit `write_parameters={"input_type": "passage", "truncate": "NONE"}` and `read_parameters={"input_type": "query", "truncate": "NONE"}` passed.

### 3. validate_index() incomplete
**Before:** Only checked model, field_map, cloud, region. dimension, metric, write/read_parameters were unchecked. Missing field = silent pass.  
**After:** All canonical fields checked. Any field not returned by API produces an explicit "unverifiable contract field" error.

### 4. create_pinecone_index.py: --execute alone was exit 0
**Before:** `--execute` without CONFIRM_PINECONE_CREATE silently became a dry-run (exit 0).  
**After:** `--execute` without CONFIRM_PINECONE_CREATE=1 exits **2** (fail-closed). Never silently downgraded.

### 5. physical_shard hardcoded to "0"
**Before:** `prepare_canary.py` passed `physical_shard="0"` to every build_record() call.  
**After:** `physical_shard=rec["physical_source"]` — real parquet path (e.g. `train/hintrain.parquet`).

### 6. physical_source unofficial field appended after schema validation
**Before:** After calling `build_record()` and `validate_record()`, the code appended `record["physical_source"]` and `record["dataset_repo"]` — unofficial extra fields that would bypass schema validation.  
**After:** These lines removed. physical_shard holds the path; dataset_repo stays in the manifest.

### 7. validate_record() allowed unknown fields
**Before:** validate_record() only checked required fields were present. Any extra fields silently passed.  
**After:** Unknown top-level fields raise SchemaViolationError.

### 8. SUPPORTED_CHUNK_STRATEGIES included "semantic" (not in registry)
**Before:** SUPPORTED_CHUNK_STRATEGIES = ["passage_native", "sentence_aware", "fixed_token_overlap", "semantic"] — "semantic" is not in CHUNKERS and would raise ValueError at runtime.  
**After:** `SUPPORTED_CHUNK_STRATEGIES = sorted(CHUNKERS.keys())` — derived from real registry. "semantic" excluded.

### 9. Manifest lacked contract binding
**Before:** Manifests (version "2") had no contract_fingerprint, no index_contract, no index_namespace.  
**After:** Manifests (version "3") include contract_version, contract_fingerprint, index_contract, index_name, index_namespace. ingest_prepared.py validates all of these.

### 10. audit_ids.py was minimal
**Before:** Only checked UUIDv5 determinism on smoke fixtures.  
**After:** Full audit: duplicate IDs, text/lang conflicts, ID recomputation, cross-language collisions, ordinal errors, parent linkage, duplicate content.

### 11. chunking_ablation.py was a stub
**Before:** `print("TODO: implement")`.  
**After:** Real offline comparison of all 4 strategies with per-strategy metrics. Proxy MRR/Recall clearly labeled as offline estimates.

### 12. CI live integration always received secrets
**Before:** `integration` job ran whenever `vars.RUN_LIVE_INTEGRATION == 'true'` (a repository variable, not manual confirmation).  
**After:** `live-integration` job only runs on `workflow_dispatch` with explicit `confirm_live: YES_RUN_LIVE` input. Normal push/PR CI is unconditionally credential-free. Secret scanner added to offline CI.

---

## Reference-repository ideas adopted/not adopted

| Idea | Status | Reason |
|------|--------|--------|
| Canonical contract module | Adopted | Eliminates duplicated dicts |
| Manifest contract fingerprint | Adopted | Binds manifest to exact index |
| Unknown-field rejection in schema | Adopted | Prevents leakage of unofficial fields |
| physical_shard as real path | Adopted | Audit traceability |
| workflow_dispatch for live CI | Adopted | Credentials never in push/PR CI |
| Full MSMARCO-XI corpus now | Not adopted | Task constraint: do not begin indexing |
| Qdrant migration | Not adopted | Task constraint: keep Pinecone |
| Sarvam v3 change | Already done | Not reverted |

---

## Commands executed with exit codes

```
uv run ruff format --check src/ tests/ bench/ scripts/    exit 0 (100 files clean)
uv run ruff check src/ tests/ bench/ scripts/             exit 0 (all passed)
uv run mypy src/                                           exit 0 (0 issues, 48 files)
uv run python scripts/scan_secrets.py                     exit 0 (no secrets detected)
uv run pytest tests/unit/ tests/behavioural/ tests/contract/ -v    exit 0 (367 passed, 4.90s)
uv run python scripts/audit_ids.py --fixtures tests/fixtures/smoke_passages.json --legacy    exit 0 (PASS, 17 records)
uv run python scripts/create_pinecone_index.py --pinecone-index msmarco-xi    exit 0 (dry-run)
uv run python scripts/create_pinecone_index.py --pinecone-index msmarco-xi --execute    exit 2 (fail-closed)
```

Canary preparation: NOT RUN (would require HuggingFace dataset download — offline; Pinecone not needed, but network required). Dry-run of ingest_prepared.py: NOT RUN (no existing canary artifacts in repo).

---

## Artifact paths

| Artifact | Path |
|----------|------|
| Canonical contract module | `src/hhgoa_rag/pinecone_contract.py` |
| Lifecycle module (fixed) | `src/hhgoa_rag/pinecone_lifecycle.py` |
| Schema module (fixed) | `src/hhgoa_rag/ingestion/schema.py` |
| Create index script (fixed) | `scripts/create_pinecone_index.py` |
| Ingest prepared script (fixed) | `scripts/ingest_prepared.py` |
| Canary preparation script (fixed) | `scripts/prepare_canary.py` |
| ID audit script (upgraded) | `scripts/audit_ids.py` |
| Chunking ablation (implemented) | `bench/chunking_ablation.py` |
| Schema docs | `docs/PINECONE_SCHEMA.md` |
| Ingestion runbook | `docs/INGESTION_RUNBOOK.md` |
| CI workflow (hardened) | `.github/workflows/ci.yml` |

---

## Determinism evidence

No canary preparation was run in this session (requires HuggingFace network access to download parquet files). The preparation code's determinism is verified indirectly through:
- 367 unit tests passing including test_chunk_ordinal_total_correct_when_split, test_split_no_text_loss_or_duplication, test_split_boundary_507_passes_508_splits
- FakeTokenizer used to confirm deterministic splitting without real network I/O

---

## Chunking comparison results

No live dataset rows were available (offline, no HF network). The `bench/chunking_ablation.py` script now correctly implements all 4 strategies with real chunker routing. When run with a network-connected environment and the pinned dataset revision, it will produce per-strategy metrics in `artifacts/reports/`.

---

## ID/linkage audit result

Audit on smoke fixtures (17 records): **PASS** — 0 duplicates, 0 collisions, 0 ordinal errors.

---

## No live provider operations

- No Pinecone API call was made
- No Sarvam API call was made
- No ElevenLabs API call was made
- No HuggingFace dataset download occurred (tests use fixtures or skip)
- No secrets were read, printed, or logged
- PINECONE_API_KEY was explicitly unset for all test runs

---

## Exact Gemini indexing sequence

Per INGESTION_RUNBOOK.md:

1. Fresh-clone setup + secret scan
2. Regenerate/verify prepared artifacts (prepare_canary.py with pinned revisions)
3. Offline dry-run (ingest_prepared.py --dry-run) from repo root and from different CWD
4. Create Pinecone index (create_pinecone_index.py --execute + CONFIRM_PINECONE_CREATE=1)
5. Validate actual index contract (describe_pinecone_index.py vs canonical)
6. Run bounded smoke integration tests (pytest tests/integration/)
7. Ingest 300-record canary into pilot_v1 (ingest_prepared.py --execute + CONFIRM_PINECONE_WRITE=1)
8. Reconcile counts (reconcile_corpus.py)
9. Post-index evaluation (audit_ids.py on ingested records)
10. Stop before expansion

---

## Final verdict

**NOT READY — blocked on canary artifact generation (offline constraint)**

### Passing gates
- Index creation request contains complete canonical contract: YES
- Live ingestion validates actual index before writing: YES (code path verified)
- Manifest binds exact index contract (fingerprint + embedded contract): YES
- Schema/provenance inconsistencies resolved: YES
- Every advertised chunking strategy is real (semantic removed from CLI): YES
- All offline gates pass (367 tests, ruff, mypy, scan_secrets): YES
- No secrets tracked: YES
- No provider calls occurred: YES

### Remaining blocker
- **Prepared canary not regenerated:** `prepare_canary.py` requires downloading `train/hintrain.parquet` and `train/bentrain.parquet` from HuggingFace (network required). No HuggingFace network access was available in this session. Therefore `artifacts/prepared/` contains no new canary artifacts.
- **Dry-run of ingest_prepared.py not run:** No manifest exists to validate. The dry-run code path is correct and tested; it just cannot be demonstrated end-to-end without the prepared artifacts.
- **Determinism not demonstrated with real data:** Cannot compare two canary runs without network access.

### Verdict once canary is generated

If the operator runs `prepare_canary.py` with the pinned revisions, verifies byte-identical output on two runs, and the dry-run ingest passes — all blockers clear and the verdict upgrades to **READY FOR GEMINI INDEXING**.

The code is architecturally correct and all offline safety gates pass. The only missing artifact is the network-dependent canary JSONL.
