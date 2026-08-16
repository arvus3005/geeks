# Final Pre-Index Readiness Report (SUPERSEDED)

> **SUPERSEDED**: This historical report has been superseded by the comprehensive pre-index reliability audit in [`docs/SKEPTICAL_PREINDEX_AUDIT.md`](file:///Users/suvra/Documents/hackerhouse-goa-task-2/docs/SKEPTICAL_PREINDEX_AUDIT.md).

**Date:** 2026-08-16
**Verdict:** `SUPERSEDED BY docs/SKEPTICAL_PREINDEX_AUDIT.md`
**Historical commit:** `41b1c485a1c6e2a7c0bee39a6168d3d7e396e6f9`

---

## Readiness classification

| Layer | Status |
|---|---|
| Offline code / test readiness | READY — 489 passed, 0 skipped, 0 failed across full suite |
| Local prepared-artifact readiness | READY — `canary-42-ee540c17772a` (300 records, 28,366 tokens) |
| Runtime remote-index validation | WILL RUN LIVE during Gemini canary execution — not pre-validated here |
| Live canary completion | NOT YET — must be triggered by Gemini (no provider calls made) |

This report documents offline engineering readiness only. No live calls to Pinecone, Sarvam, or ElevenLabs were made during this pass. Live indexing, remote index schema validation, and freshness reconciliation will be performed live when `scripts/index_canary.py --execute` is executed.

---

## Summary of Corrective Changes

| Area | Resolution Details |
|---|---|
| **1. Production token-rate limiter** | Extracted `_TokenRateLimiter` class into `scripts/index_canary.py`. Replaced the defective inline state with an atomic check-and-reserve sliding-window limiter. Sleeps occur outside the lock via an injectable clock/sleeper, preventing competing workers from overwriting window state or exceeding rate ceilings upon waking. |
| **2. Canonical-contract cleanup** | Unified all production code and CLI scripts onto `src/hhgoa_rag/pinecone_contract.py` constants (`INDEX_NAME`, `NAMESPACE`, `CLOUD`, `REGION`, `MODEL`, `TEXT_FIELD`, `MAX_INPUT_TOKENS`, `MAX_BATCH_SIZE`, `FIELD_MAP`, `WRITE_PARAMETERS`, `READ_PARAMETERS`). Defined `MODEL_INPUT_LIMIT = MAX_INPUT_TOKENS` and `TEXT_RECORD_FIELD = TEXT_FIELD` as explicit derived aliases. |
| **3. Test report isolation & Git cleanliness** | Audited all test invocations of `_run_canary()` in `test_preindex_hardening_v2.py` and `test_sdk_validation.py` to pass temporary `--report-dir` paths via `tmp_path`. Added targeted `.gitignore` patterns for execution reports (`artifacts/reports/canary_index_execution_*.json` and `*.md`). |
| **4. Comprehensive behavioural tests** | Added deterministic unit and concurrency tests for `_TokenRateLimiter` using fake clocks/sleepers, verifying window rollover, ceiling adherence, reject-on-oversized, zero/negative rejection, and multi-thread race safety. Added contract resolution assertions. |
| **5. Formatting & Linting** | Standardized all code formatting via `ruff format` and verified with `ruff check` (103 files, 0 errors). |

---

## Canonical Pinecone contract

**Module:** `src/hhgoa_rag/pinecone_contract.py`
**Contract version:** `1`
**SHA-256 fingerprint:** `a76947f5d5f5afb41a693501e927394705c607ff4f59b160c225ad6c2be9ddaa`

| Field | Value | Canonical Status |
|---|---|---|
| `INDEX_NAME` | `"msmarco-xi"` | Single Source of Truth |
| `NAMESPACE` | `"pilot_v1"` | Single Source of Truth |
| `CLOUD` | `"aws"` | Single Source of Truth |
| `REGION` | `"us-east-1"` | Single Source of Truth |
| `MODEL` | `"multilingual-e5-large"` | Single Source of Truth |
| `DIMENSION` | `1024` | Single Source of Truth |
| `METRIC` | `"cosine"` | Single Source of Truth |
| `TEXT_FIELD` | `"chunk_text"` | Single Source of Truth |
| `FIELD_MAP` | `{"text": "chunk_text"}` | Immutable `MappingProxyType` |
| `WRITE_PARAMETERS` | `{"input_type": "passage", "truncate": "NONE"}` | Immutable `MappingProxyType` |
| `READ_PARAMETERS` | `{"input_type": "query", "truncate": "NONE"}` | Immutable `MappingProxyType` |
| `MAX_INPUT_TOKENS` | `507` | Single Source of Truth |
| `MAX_BATCH_SIZE` | `96` | Single Source of Truth |
| `DATASET_REPO` | `"ai4bharat/MSMARCO-XI"` | Single Source of Truth |
| `DATASET_REVISION` | `"bf5cdc1f26e581e519018e434db14edd1b77602b"` | Pinned SHA |
| `TOKENIZER_REPO` | `"intfloat/multilingual-e5-large"` | Single Source of Truth |
| `TOKENIZER_REVISION` | `"3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3"` | Pinned SHA |

---

## Remaining Justified Contract Literals Audit

A repository-wide scan confirmed that all executable production logic imports directly from `src/hhgoa_rag/pinecone_contract.py`. The only remaining string occurrences in the codebase are documented below:

| File | Line | Occurrence | Justification |
|---|---|---|---|
| `src/hhgoa_rag/ingestion/schema.py` | 32 | `# "chunk_text"` | Comment documenting `TEXT_RECORD_FIELD` alias |
| `src/hhgoa_rag/pinecone_store.py` | 11 | `{"text": "chunk_text"}` | Module docstring explanation of field mapping |
| `src/hhgoa_rag/pinecone_store.py` | 31 | `# "chunk_text"` | Comment on `TEXT_RECORD_FIELD: str = _CANONICAL_TEXT_FIELD` alias |

*Note on `budget.py`: `src/hhgoa_rag/ingestion/budget.py` imports only `FIELD_MAP` from `pinecone_contract` (re-exported as a plain dict for JSON compatibility). It does not import cloud/region/model constants as budget tracking is plan-specific rather than index-configuration-specific.*

---

## Production Token-Rate Limiter Design

The rate limiter in `scripts/index_canary.py` implements the `_TokenRateLimiter` class:
- **State encapsulation**: Protected by a `threading.Lock()`, maintaining `_window_start` (monotonic float) and `_window_tokens` (integer).
- **Atomic reservation**: `acquire(token_count)` checks remaining capacity and books tokens under lock if `_window_tokens + token_count <= tokens_per_window`.
- **Non-busy waiting**: If projected tokens exceed capacity, calculates remaining window duration, cleanly exits the lock context, and invokes `sleeper(wait)` outside the lock so other threads can proceed.
- **Atomic rollover**: Re-enters the lock after waking, verifies the current clock against `_window_start`, and rolls the window forward exactly once without unconditional resets.
- **Fail-safe validation**: Rejects non-positive token reservations and single reservations exceeding `tokens_per_window` immediately.

---

## Prepared Canary Artifact Status

| Field | Value |
|---|---|
| Manifest ID | `canary-42-ee540c17772a` |
| Records | 300 total (100 English, 100 Hindi, 100 Bengali) |
| Total tokens | 28,366 |
| JSONL SHA-256 | `ca912d133c3033eca71cc86045923e0165f5b43baba6f1951a1741ff0a3a9217` |
| Chunking strategy | `sentence_aware` |
| Manifest schema version | `3` |
| Contract fingerprint | `a76947f5d5f5afb41a693501e927394705c607ff4f59b160c225ad6c2be9ddaa` |
| Dataset revision | `bf5cdc1f26e581e519018e434db14edd1b77602b` (pinned) |
| Tokenizer revision | `3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3` (pinned) |

The prepared canary artifact is valid, verified, and unchanged.

---

## Verification & Quality Gates

All commands executed from repository root in the locked virtual environment:

```
$ uv sync --all-extras
Resolved 77 packages in 3ms
Checked 75 packages in 2ms

$ uv run ruff format --check src/ tests/ bench/ scripts/
103 files already formatted (exit 0)

$ uv run ruff check src/ tests/ bench/ scripts/
All checks passed! (exit 0)

$ uv run mypy src/
Success: no issues found in 48 source files (exit 0)

$ uv run pytest tests/unit/ tests/behavioural/ tests/contract/ -q -rs
489 passed in 6.77s (exit 0, 0 skipped, 0 failed)

$ uv run python scripts/scan_secrets.py
Credential scan: no secrets detected in tracked source files (exit 0)

$ uv run python scripts/index_canary.py --manifest artifacts/prepared/canary-42-ee540c17772a_manifest.json --report-dir /tmp/canary_report_test
DRY-RUN complete — no records were written (exit 0)
```

---

## Gemini Canary Execution Commands

### Foreground execution (interactive)
```bash
export PINECONE_API_KEY=<key>
CONFIRM_PINECONE_WRITE=1 \
  uv run python scripts/index_canary.py \
    --manifest artifacts/prepared/canary-42-ee540c17772a_manifest.json \
    --execute --resume --concurrency 4
```

### Background execution (nohup)
```bash
export PINECONE_API_KEY=<key>
nohup env CONFIRM_PINECONE_WRITE=1 \
  uv run python scripts/index_canary.py \
    --manifest artifacts/prepared/canary-42-ee540c17772a_manifest.json \
    --execute --resume > logs/index_canary.log 2>&1 &
```

> **Note**: `PINECONE_API_KEY` must be exported before executing `nohup env CONFIRM_PINECONE_WRITE=1 ...`. Checkpoints are written atomically to `artifacts/checkpoints/` after every acknowledged batch, enabling clean resumption on interruption.

---

## Final Verdict

`READY FOR GEMINI CANARY EXECUTION — NO KNOWN PRE-INDEX BLOCKERS`
