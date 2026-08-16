# Skeptical Pre-Index Reliability Audit & Historical Report

> **HISTORICAL AUDIT — Not the current readiness verdict.**
> This document assessed the repository at an earlier state (commit `1666db43cb22cf8bf638a16086d281440604e195`).
> Its test counts and capacity estimates are historical evidence, not current truth.
> See [docs/FINAL_PREINDEX_READINESS_REPORT.md](FINAL_PREINDEX_READINESS_REPORT.md) for the authoritative current verdict.

**Auditor:** Senior Repository Auditor & Pre-Index Reliability Engineer  
**Date:** 2026-08-16  
**Audited Branch:** `main`  
**Initial Audited Remote SHA:** `1666db43cb22cf8bf638a16086d281440604e195`  
**Remote Sync Status:** Verified identical (`origin/main` matches local `main`)  
**Working Tree Status:** Clean (untracked artifacts and caches ignored)  
**Historical Verdict:** `READY FOR GEMINI CANARY PREFLIGHT`

---

## 1. Executive Summary & Authorization Boundary

This audit independently verified the readiness of the repository `https://github.com/arvus3005/geeks` for Pinecone canary preflight execution.

### Boundary Protections Adhered To During This Audit:
- **No live API calls:** Zero network calls were made to Pinecone, Sarvam, or ElevenLabs.
- **No vector index mutations:** No index was created, altered, upserted, queried, or deleted.
- **No namespace deletion:** Automatic deletion/clearing is not implemented; namespace safety is fail-closed.
- **No secret leakage:** No environment secrets or credentials were committed, copied, printed, or exposed.
- **Architecture preserved:** Pinecone integrated-embedding architecture (`multilingual-e5-large`, 1024-dim, cosine) is preserved for the canary.

---

## 2. Files Inspected

The following 22 files were systematically audited:

1. [`README.md`](../README.md)
2. [`.env.example`](../.env.example)
3. [`.gitignore`](../.gitignore)
4. [`Makefile`](../Makefile)
5. [`pyproject.toml`](../pyproject.toml)
6. [`.github/workflows/ci.yml`](../.github/workflows/ci.yml)
7. [`docs/FINAL_PREINDEX_READINESS_REPORT.md`](FINAL_PREINDEX_READINESS_REPORT.md)
8. [`docs/PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md)
9. [`docs/INGESTION_RUNBOOK.md`](INGESTION_RUNBOOK.md)
10. [`docs/PINECONE_SCHEMA.md`](PINECONE_SCHEMA.md)
11. [`docs/DATASET_CONTRACT.md`](DATASET_CONTRACT.md)
12. [`scripts/index_canary.py`](../scripts/index_canary.py)
13. [`scripts/prepare_canary.py`](../scripts/prepare_canary.py)
14. [`scripts/ingest_prepared.py`](../scripts/ingest_prepared.py)
15. [`scripts/create_pinecone_index.py`](../scripts/create_pinecone_index.py)
16. [`scripts/describe_pinecone_index.py`](../scripts/describe_pinecone_index.py)
17. [`scripts/estimate_capacity.py`](../scripts/estimate_capacity.py)
18. [`scripts/scan_secrets.py`](../scripts/scan_secrets.py)
19. [`tests/integration/test_pinecone_smoke.py`](../tests/integration/test_pinecone_smoke.py)
20. [`src/hhgoa_rag/pinecone_contract.py`](../src/hhgoa_rag/pinecone_contract.py)
21. [`src/hhgoa_rag/pinecone_lifecycle.py`](../src/hhgoa_rag/pinecone_lifecycle.py)
22. [`src/hhgoa_rag/pinecone_store.py`](../src/hhgoa_rag/pinecone_store.py)

---

## 3. Investigation of Suspected Issues (A–G)

### A. Live Pinecone Integration Tests May Silently Skip
- **Status:** **CONFIRMED & RESOLVED**
- **Evidence:**
  - `tests/integration/test_pinecone_smoke.py:39`: `_require_opt_in()` skipped unless `PINECONE_SMOKE_TEST=1` AND `PINECONE_API_KEY` were both set.
  - `.github/workflows/ci.yml:60-65`: The `live-integration` manual job provided `PINECONE_API_KEY` and `SARVAM_API_KEY` but omitted `PINECONE_SMOKE_TEST: "1"`. Thus manual dispatch runs would execute `pytest tests/integration/ -v` and exit with code 0 while skipping all 10 tests.
  - `docs/INGESTION_RUNBOOK.md:123`: The documented command omitted `PINECONE_SMOKE_TEST=1`.
- **Changes Made:**
  - `tests/integration/test_pinecone_smoke.py`: Updated `_require_opt_in()` so that if `PINECONE_SMOKE_TEST=1` is set but `PINECONE_API_KEY` is missing/empty, it fails closed immediately (`pytest.fail(...)`) instead of skipping.
  - `.github/workflows/ci.yml`: Added `PINECONE_SMOKE_TEST: "1"` to the manual `live-integration` job env.
  - `docs/INGESTION_RUNBOOK.md` & `README.md`: Updated integration test command to include `PINECONE_SMOKE_TEST=1`.
- **Tests Added:** `TestIntegrationOptIn` in `tests/unit/test_preindex_hardening_v2.py` verifying skip on missing opt-in, fail-closed on missing key with opt-in, and pass when both are present.

---

### B. Canary Namespace Contamination Detected Only After Writes
- **Status:** **CONFIRMED & RESOLVED**
- **Evidence:**
  - `scripts/index_canary.py`: Prior to this pass, `_run()` checked `validate_index()` (Step 5) and immediately proceeded to parallel upserts in Step 6. Contamination was only evaluated in Step 7 (`reconcile_corpus` / freshness polling) *after* batches were already submitted to Pinecone.
- **Changes Made:**
  - Added **Step 5.5: Pre-write namespace preflight** in `scripts/index_canary.py`.
  - For fresh runs (`--execute` without resume or with empty completed checkpoint): Requires `pilot_v1` vector count to be exactly 0. If > 0, aborts with `CanaryError` (category `NamespaceContaminatedPreflight`) before any batch is submitted.
  - For resume runs: Checks vector count against completed batches ($N_{comp}$). If vector count > $N_{comp}$ or > 300, aborts fail-closed.
  - Provider failure during preflight: Aborts with `CanaryError` (category `PreflightProviderFailure`).
  - Strict non-destructive invariant: Never automatically clears or deletes the namespace.
- **Tests Added:** `TestPreWriteNamespacePreflight` with 5 mock tests verifying:
  1. Fresh empty namespace passes
  2. Fresh contaminated namespace aborts with 0 upserts
  3. Compatible resume passes preflight
  4. Resume with unexpected existing records aborts with 0 upserts
  5. Preflight provider network failure aborts with 0 upserts

---

### C. Conflicting Canary Identities and Commands in Documentation
- **Status:** **CONFIRMED & RESOLVED**
- **Evidence:**
  - `docs/PROJECT_SUMMARY.md:181` referenced stale `canary-42-02c06c8a0809` and listed `ingest_prepared.py` as Gemini's first command.
  - `docs/FINAL_PREINDEX_READINESS_REPORT.md` referenced `canary-42-ee540c17772a`.
  - `README.md:61` referenced `docs/DATASET_CONTRACT.md` which was missing from the repository.
- **Changes Made:**
  - Canonicalized the execution path to `scripts/index_canary.py`. `scripts/ingest_prepared.py` is clearly designated as a lower-level/legacy tool.
  - Restored `docs/DATASET_CONTRACT.md` detailing dataset provenance, configs, and leakage rules.
  - Unified all documentation and reports on `canary-42-ee540c17772a` (the deterministic manifest produced by `prepare_canary.py` with `sentence_aware`).
  - Documented how to dynamically capture manifest paths.
  - Marked `docs/FINAL_PREINDEX_READINESS_REPORT.md` as superseded.

---

### D. Full English/Hindi/Bengali Indexing Not Supported on Starter
- **Status:** **CONFIRMED & VERIFIED**
- **Evidence:**
  - `src/hhgoa_rag/ingestion/budget.py:159` unconditionally raises `StarterFullModeError` when `PINECONE_PLAN=starter`.
  - Codebase search confirmed zero Qdrant, FAISS, or Chroma implementations exist (only legacy comments).
- **Readiness Matrix Documented:**
  - Layer 1: **Offline pre-index readiness:** COMPLETE (502 tests passing, clean lint/types/security).
  - Layer 2: **300-record canary readiness:** READY FOR GEMINI PREFLIGHT.
  - Layer 3: **Bounded Starter pilot readiness:** ARCHITECTED & BOUNDED (10,000 records).
  - Layer 4: **Full English/Hindi/Bengali indexing:** NOT SUPPORTED on Starter (requires plan upgrade or local vector database).
  - Layer 5: **Complete HH Goa end-to-end product:** POST-INDEX WORK PENDING.

---

### E. Capacity Estimator Vector Dimension Discrepancy
- **Status:** **CONFIRMED & RESOLVED**
- **Evidence:**
  - `scripts/estimate_capacity.py:18` hardcoded `DENSE_DIM = 384` instead of the canonical `1024` from `hhgoa_rag.pinecone_contract.DIMENSION` (multilingual-e5-large). This underestimated dense storage by a factor of 2.67x.
- **Changes Made:**
  - Imported `CANONICAL_DIMENSION = 1024` from `src/hhgoa_rag/pinecone_contract.py`.
  - Differentiated 4 scopes: (1) 300-record canary, (2) 10,000-record bounded pilot, (3) 3-language target (en/hi/bn, ~16.4M passages, ~127.5 GB storage with safety margin), and (4) full 14-config corpus (~61.3M passages, ~478.1 GB).
  - Separated measured facts from assumptions.
  - Added explicit warning that a 200 GB Mac cannot be declared sufficient until actual record and chunk counts are measured from Parquet shards.
- **Tests Added:** `TestCapacityEstimator` verifying canonical dimension usage, proportional scaling when dimension changes, presence of all 4 scopes, and local disk warning.

---

### F. Token Limiter Sliding-Window vs Fixed-Window Implementation
- **Status:** **CONFIRMED & RESOLVED**
- **Evidence:**
  - `scripts/index_canary.py:246-250`: `_TokenRateLimiter` reset its window counter every 60 seconds (`elapsed >= window_seconds`), allowing boundary bursts of up to 2x capacity across any rolling 60-second window, despite docstrings claiming it was a sliding window.
- **Changes Made:**
  - Implemented a true sliding-window rate limiter in `_TokenRateLimiter` using `collections.deque` tracking timestamped reservation events `(timestamp, token_count)`.
  - Atomic pruning and reservation under `threading.Lock`.
  - Sleeps strictly outside the lock.
  - Injectable clock and sleeper callables for deterministic unit testing.
- **Tests Added:** `TestTokenRateLimiterBehavioural` including boundary-burst tests (t=50s and t=65s requests forced to wait until t=110s), concurrency tests, and constructor validation.

---

### G. CI and Developer Command Inconsistencies
- **Status:** **CONFIRMED & RESOLVED**
- **Evidence:**
  - `.github/workflows/ci.yml` omitted `mypy` in the offline `test` job.
  - `Makefile:41` used a weak regex `rg -l "pc-[a-zA-Z0-9]{8}..."` instead of invoking `scripts/scan_secrets.py`.
- **Changes Made:**
  - Added `uv run mypy src/` to `.github/workflows/ci.yml`.
  - Updated `Makefile` `scan-secrets` to run `uv run python scripts/scan_secrets.py` and included it in the `ci` target.

---

## 4. Canary Manifest Details (Verified Offline)

| Property | Canonical Value |
|---|---|
| **Manifest ID** | `canary-42-ee540c17772a` |
| **Dataset Repo** | `ai4bharat/MSMARCO-XI` |
| **Dataset Revision** | `bf5cdc1f26e581e519018e434db14edd1b77602b` |
| **Tokenizer Repo** | `intfloat/multilingual-e5-large` |
| **Tokenizer Revision** | `3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3` |
| **Tokenizer Fingerprint** | `2616986da866a9dc` |
| **Total Records** | `300` |
| **Per-Language Counts** | English: `100`, Hindi: `100`, Bengali: `100` |
| **Total Tokens** | `28,366` |
| **Maximum Record Token Count** | `211` (well below 507 limit) |
| **Chunking Strategy** | `sentence_aware` |
| **JSONL Size** | `395,171` bytes |
| **JSONL SHA-256** | `ca912d133c3033eca71cc86045923e0165f5b43baba6f1951a1741ff0a3a9217` |
| **Contract SHA-256 Fingerprint** | `a76947f5d5f5afb41a693501e927394705c607ff4f59b160c225ad6c2be9ddaa` |
| **Readiness** | `ready_for_write: true` (0 readiness failures, forbidden field audit PASS) |
| **Determinism** | Verified byte-for-byte identical across independent generation runs |

---

## 5. Offline Quality Gate Verification

All verification commands executed from repository root in the locked virtual environment:

```bash
$ uv sync --frozen --all-extras
Resolved 77 packages in 2ms (exit 0)

$ uv run ruff format --check src tests bench scripts
103 files already formatted (exit 0)

$ uv run ruff check src tests bench scripts
All checks passed! (exit 0)

$ uv run mypy src
Success: no issues found in 48 source files (exit 0)

$ uv run pytest tests/unit tests/behavioural tests/contract -q -rs
502 passed in 5.17s (exit 0, 0 failed, 0 skipped)

$ uv run python scripts/scan_secrets.py
Credential scan: no secrets detected in tracked source files (exit 0)

$ uv run python scripts/create_pinecone_index.py --pinecone-index msmarco-xi
[DRY-RUN] Plan: create Pinecone serverless index 'msmarco-xi' (exit 0, 0 API calls)

$ uv run python scripts/estimate_capacity.py
Capacity estimate computed successfully (canonical dimension: 1024) (exit 0)
```

---

## 6. Post-Index Product Requirements (Out of Scope for Pre-Index Pass)

The following items are post-index product deliverables. They are tracked as future work and are **not** pre-index blockers:
1. Sarvam Saaras v3 STT adapter integrated into active voice stream endpoint.
2. Audio streaming/upload endpoints for voice interaction.
3. Reranker wired into the serving path for multi-candidate reranking.
4. Generative multilingual answer synthesis stage.
5. End-to-end input/output safety guardrails.
6. Real P50/P70/P100 latency benchmarking against live infrastructure.
7. Verification of the under-200 ms end-to-end latency target.
8. Multilingual retrieval quality evaluation on real data.
9. UI and live judging demo submission artifacts.

---

## 7. Multi-Layer Readiness Matrix

| Readiness Layer | Status | Notes |
|---|---|---|
| **1. Offline Pre-Index Readiness** | **VERIFIED READY** | 502 tests passed, 0 skipped, 0 failed; clean lint/types/security. |
| **2. 300-Record Canary Readiness** | **READY FOR GEMINI PREFLIGHT** | Artifacts verified; pre-write preflight & sliding-window limiter implemented. |
| **3. Bounded Starter Pilot Readiness** | **READY (ARCHITECTURAL CEILING)** | Configured for 10,000 records / 4M tokens; budget ledger active. |
| **4. Full En/Hi/Bn Corpus Indexing** | **BLOCKED (STARTER LIMITATION)** | Requires Pinecone plan upgrade (~127.5 GB) or dedicated local vector backend. |
| **5. Complete E2E Submission** | **POST-INDEX WORK PENDING** | Requires live indexing completion and post-index pipeline integration. |

---

## 8. Gemini Execution Instructions

### Exact Next Command Gemini Should Run First (READ-ONLY PREFLIGHT):
Gemini must first run an offline dry-run to verify local environment and artifact integrity:

```bash
uv run python scripts/index_canary.py \
  --manifest artifacts/prepared/canary-42-ee540c17772a_manifest.json
```

### Mandatory Conditions Before Live Canary Execution (`--execute`):
1. The `msmarco-xi` Pinecone index must exist with the canonical contract (`multilingual-e5-large`, 1024-dim, cosine, field_map `{"text": "chunk_text"}`).
2. Namespace `pilot_v1` must be verified empty (preflight will automatically enforce this and abort if non-empty).
3. `PINECONE_API_KEY` must be exported in the shell environment.
4. `CONFIRM_PINECONE_WRITE=1` must be explicitly provided.

### Approved Live Canary Command:
```bash
export PINECONE_API_KEY=<your-key>
CONFIRM_PINECONE_WRITE=1 \
  uv run python scripts/index_canary.py \
    --manifest artifacts/prepared/canary-42-ee540c17772a_manifest.json \
    --execute --resume --concurrency 4
```

---

## Final Verdict

```
READY FOR GEMINI CANARY PREFLIGHT
```
