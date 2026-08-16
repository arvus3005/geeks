# Final Pre-Index Readiness Report

## Verdict

**READY FOR GEMINI LIVE 300-RECORD CANARY**

All offline gates pass with zero live provider calls, zero unexpected skips, and
zero committed secrets. The legacy live-ingestion bypass is disabled, and the
canary indexer now requires exact post-write ID-set equality before reporting
success. The live 300-record canary itself has NOT been executed or verified —
that is Gemini's live step in Antigravity IDE.

- **10,000-record Starter pilot: NOT AUTHORIZED and NOT validated.**
- **Full English/Hindi/Bengali corpus: NOT AUTHORIZED and NOT capacity-validated.**

---

## Scope

- Pre-index readiness only. No live Pinecone write was performed during this task.
- Target: 300-record canary (100 English + 100 Hindi + 100 Bengali), namespace
  `pilot_v1`, index `msmarco-xi`. Pinecone is the sole production vector store.
- `scripts/index_canary.py` remains the only live canary indexing path.
- Out of scope: 10,000-record pilot, full 14-config corpus, any live provider call.

## Commit state

- Starting commit SHA (this hardening pass): `273c290604859355b357df300836930d76cf7b95`
- Final commit SHA: recorded in the git history of `origin/main` immediately
  after this report is committed (this file cannot embed its own commit hash — a
  circular reference). See the "fix(pre-index): close final count and
  reconciliation gaps" commit at the tip of `main`.
- Tested tree state: the working tree containing the changes below, with the full
  offline gate suite rerun successfully (see "Gate results").

---

## Fixes completed in this pass (final count/reconciliation hardening)

1. **Genuine fail-closed on non-integer counts.** Both
   `PineconeStore.count_namespace()` and `index_canary._get_ns_vector_count()`
   now accept ONLY a genuine non-negative Python `int`. `bool` (checked first,
   since bool subclasses int), floats (`300.0`, `300.9`), numeric strings
   (`"300"`), negatives, `None`, and missing `vector_count` are rejected and
   never coerced. `count_namespace()` raises `PineconeProviderError`;
   `_get_ns_vector_count()` returns the "unverifiable" `None`. Provider
   exceptions remain failures. An empty/absent namespace in a valid response
   still returns 0. Parametrized tests prove every accepted/rejected type across
   preflight, polling, and reconciliation gates
   (`tests/unit/test_preindex_hardening_v5.py`).
2. **Environment-only Pinecone credentials.** Removed the `--pinecone-api-key`
   CLI flag from `reconcile_corpus.py`, `describe_pinecone_index.py`,
   `smoke_query_pinecone.py`, and `validate_pinecone_config.py`. Credentials read
   only from `PINECONE_API_KEY`; a missing/blank value exits non-zero before any
   provider construction. Tests assert the flag is absent from `--help`, rejected
   by argparse (exit 2), and that missing env fails closed.
3. **Secondary reconciliation asserts expected count.** `reconcile_corpus.py`
   gained `--expected-count` (positive integer). Exit 0 only when the verified
   namespace count equals the expected count; exit non-zero on mismatch,
   malformed/unverifiable count, provider failure, or invalid expectation. JSON
   output includes index, namespace, expected/actual count, and
   `pass`/`mismatch`/`unverifiable` status with no credentials. Documented as a
   SECONDARY check in Runbook Step 8; `index_canary.py` exact-ID reconciliation
   remains authoritative.
4. **Reproducible dependency installation.** `uv sync` reproduction commands in
   `README.md`, `Makefile`, and `docs/INGESTION_RUNBOOK.md` now use
   `uv sync --frozen --all-extras`. `uv.lock` was not modified.
5. **Removed stale repository-state statements.** `docs/PROJECT_SUMMARY.md` no
   longer claims "Nothing was committed or pushed"; it now states that offline
   code/doc changes are committed and pushed while generated artifacts remain
   git-ignored and no live write occurred.
6. **Historical audit clearly separated from current truth.**
   `docs/SKEPTICAL_PREINDEX_AUDIT.md` opens with a prominent HISTORICAL AUDIT
   notice naming the earlier assessed commit and pointing to this report as the
   authoritative verdict. Old measurements preserved.
7. **Removed misleading Qdrant terminology.** `passage_ids.py` now says
   "deterministic vector ID" / "provenance occurrence ID"; `language_routing.py`
   says "language metadata filter"; `.gitignore` and `.dockerignore` label
   `qdrant_data/` as "legacy local vector-store artefacts". Public function names
   and the deterministic ID algorithm are unchanged. Remaining references
   (`CLAUDE.md` project rules; the historical audit doc) are canonical or
   genuinely historical/comparative.
8. **Corrected background execution example.** `index_canary.py` docstring and
   the Runbook now use valid `nohup env VAR=val ...` shell with no literal
   secret, writing to `artifacts/logs/index_canary.log`.
9. **Makefile correctness.** `canary-dry-run` and `canary-execute` added to
   `.PHONY`; `$(MANIFEST)` is quoted in `canary-execute`; MANIFEST remains
   required; `index_canary.py` remains the canonical dry-run/live path; no live
   target runs during tests/install/CI/default; live execution stays guarded by
   `CONFIRM_PINECONE_WRITE=1`.
10. **This report updated** with scope, commit state, fix list, verification
    evidence, honest limitations, and final verdict.

---

## Prior hardening pass (retained for provenance)

1. **Exact post-write ID reconciliation (`scripts/index_canary.py`).**
   Step 7 no longer accepts count equality alone. After the namespace count
   reaches exactly 300, it enumerates vector IDs via
   `PineconeStore.list_vector_ids(namespace="pilot_v1")` and requires: count ==
   300 AND exactly 300 unique enumerated IDs AND the enumerated set equals the
   manifest-derived expected ID set (no missing, no extra). Failure categories:
   `PostWriteReconciliationUnverifiable` (enumeration error/unsupported/malformed/
   duplicate) and `PostWriteOwnershipMismatch` (count 300 but ID set differs).
   Count result and exact-ID result are stored separately in the report
   (`count_reconciliation`, `exact_id_reconciliation`). No upserts occur after a
   reconciliation failure. Discrepancy reporting shows only counts, never ID lists.

2. **Legacy live-ingestion path disabled (`scripts/ingest_prepared.py`).**
   `--execute` exits non-zero (2) before constructing a Pinecone client or
   reading `PINECONE_API_KEY`, even when `CONFIRM_PINECONE_WRITE=1` and
   `PINECONE_API_KEY` are present. The dead live code path was removed; the script
   remains a full offline validator/dry-run tool. Redirect messages in
   `ingest_all.py`, `ingest_shard.py`, and `resume_ingest.py` now point to
   `index_canary.py`.

3. **Deterministic canary preparation (`scripts/prepare_canary.py`).**
   The canonical manifest no longer contains `created_at` or
   `prepared_record_path_full`. Runtime metadata is written to a separate
   non-canonical `<id>_runtime.json` sidecar (excluded from identity, checksum,
   checkpoints). The manifest is serialized deterministically (sorted keys, UTF-8,
   single trailing newline). Two runs with identical inputs into different output
   directories/times produce identical manifest IDs, byte-identical JSONL, and
   byte-identical manifest JSON.

4. **Pinecone ID pagination hardening (`PineconeStore.list_vector_ids`).**
   Tracks seen pagination tokens (fail closed on a repeated token); fails closed
   when a next token yields no progress; detects duplicate IDs across pages;
   rejects non-string/empty IDs; adds a conservative maximum-page guard
   (`MAX_ENUMERATION_PAGES`); supports both `list_paginated` and iterator-style
   `list`; wraps provider exceptions as `PineconeProviderError`.

5. **Fail-closed counting (`PineconeStore.count_namespace`).**
   Returns 0 only when a valid statistics response proves the namespace is
   absent/empty. Provider exceptions, malformed responses, non-integer/boolean/
   negative counts raise `PineconeProviderError`. `scripts/reconcile_corpus.py`
   catches that error, prints "reconciliation UNVERIFIABLE", and exits non-zero.
   No safety tool treats an exception as an empty namespace.

6. **Capacity tokenizer fallback repaired (`scripts/measure_corpus_capacity.py`).**
   On tokenizer-load failure it prints an actionable error and exits non-zero. The
   previously-broken undefined-`tok` whitespace "fallback" is gone; no approximate
   token counts are substituted into canonical measured results.

7. **Makefile and CI gates.**
   `make dry-run-ingest` requires `MANIFEST=<path>` and runs
   `uv run python scripts/index_canary.py --manifest "$(MANIFEST)"`. Duplicate
   dry-run targets consolidated. `make typecheck` runs `uv run mypy src scripts`.
   CI offline job enforces `uv sync --frozen --all-extras`,
   `uv run ruff format --check .`, `uv run ruff check .`,
   `uv run mypy src scripts`, the secret scan, and unit+contract+behavioural tests.
   Live integration tests remain `workflow_dispatch`-only with `YES_RUN_LIVE`;
   normal CI receives no provider credentials.

8. **Documentation reconciled** (`docs/INGESTION_RUNBOOK.md`,
   `docs/PROJECT_SUMMARY.md`, `docs/PINECONE_SCHEMA.md`, `README.md`): removed live
   `ingest_prepared.py --execute` commands; replaced the broken
   `diff <(sha256sum ...)` determinism check with `cmp` / extracted-hash
   comparison; used full dataset revision
   `bf5cdc1f26e581e519018e434db14edd1b77602b` and tokenizer revision
   `3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3`; updated stale figures (557 tests,
   64 mypy source files); reconciled corpus terminology (source rows / unique
   passages / chunks / Pinecone vectors / measured / extrapolated) and labelled
   ~24.87M vectors / ~171.67 GB explicitly as EXTRAPOLATED.

---

## Files changed (this pass)

- `src/hhgoa_rag/pinecone_store.py` (strict int in `count_namespace`)
- `scripts/index_canary.py` (strict int in `_get_ns_vector_count`; nohup example)
- `scripts/reconcile_corpus.py` (env-only key; `--expected-count`)
- `scripts/describe_pinecone_index.py`, `scripts/smoke_query_pinecone.py` (env-only key)
- `src/hhgoa_rag/ingestion/passage_ids.py`, `src/hhgoa_rag/retrieval/language_routing.py` (terminology)
- `Makefile` (`.PHONY`, quoting, frozen install), `.gitignore`, `.dockerignore`
- `README.md`, `docs/INGESTION_RUNBOOK.md`, `docs/PROJECT_SUMMARY.md`,
  `docs/SKEPTICAL_PREINDEX_AUDIT.md`, `docs/FINAL_PREINDEX_READINESS_REPORT.md`
- `tests/unit/test_preindex_hardening_v5.py` (new — 52 tests)
- `tests/unit/test_preindex_hardening_v4.py` (updated match string for bool)

---

## Exact commands executed (offline, no credentials in scope)

```bash
uv sync --frozen --all-extras
uv run ruff format .
uv run ruff format --check .
uv run ruff check .
uv run mypy src scripts
uv run python scripts/scan_secrets.py .
uv run pytest tests/unit tests/contract tests/behavioural -q -rs
```

## Gate results (this pass, 2026-08-16)

- `uv sync --frozen --all-extras` — locked environment resolved (exit 0).
- `ruff format --check .` — 107 files already formatted (0 errors).
- `ruff check .` — All checks passed (0 errors).
- `mypy src scripts` — Success: no issues found in 64 source files.
- `scan_secrets.py .` — no secrets detected in tracked source files; exit 0.
- `pytest tests/unit tests/contract tests/behavioural -q -rs`
  (with `PINECONE_API_KEY`/`SARVAM_API_KEY`/`HF_TOKEN`/`CONFIRM_PINECONE_WRITE`
  unset) — **609 passed, 0 failed, 0 skipped** (was 557; +52 new v5 tests).

### Targeted checks

- `make -n canary-dry-run MANIFEST=test.json` — expands to
  `uv run python scripts/index_canary.py --manifest "test.json"` (quoted; no run).
- `ingest_prepared.py --execute` with `PINECONE_API_KEY` +
  `CONFIRM_PINECONE_WRITE=1` set — exit 2 before any Pinecone import.
- `--help` for reconcile/describe/smoke/validate — zero `--pinecone-api-key`
  occurrences; the flag is rejected by argparse (exit 2).
- Deterministic preparation: two `prepare_canary.py` runs
  (`HF_HUB_OFFLINE=1`, seed 42, pinned revisions) into `/tmp/run1` and
  `/tmp/run2` produced byte-identical JSONL and manifest (`cmp` exit 0); JSONL
  SHA-256 `ca912d133c3033eca71cc86045923e0165f5b43baba6f1951a1741ff0a3a9217`.

---

## Determinism evidence

Two `prepare_canary.py` runs with identical pinned inputs into different output
directories (`/tmp/run1`, `/tmp/run2`):

```
dataset-revision   bf5cdc1f26e581e519018e434db14edd1b77602b
tokenizer-revision 3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3
seed 42, chunk-strategy sentence_aware
```

- Manifest ID (both runs): `canary-42-ee540c17772a`
- JSONL SHA-256 (both runs): `ca912d133c3033eca71cc86045923e0165f5b43baba6f1951a1741ff0a3a9217`
- Manifest JSON SHA-256 (both runs): `26fe3d5213bef8424e6bf3035cce46baea0174844f87da60b6152a4703ed79c9`
- `cmp` on both the JSONL and the manifest returned byte-identical (exit 0).
- Only the non-canonical `<id>_runtime.json` sidecar differs (timestamp/abs paths),
  as designed.

---

## Confirmations

- **No live API calls.** No Pinecone index was created, listed, queried, upserted,
  or deleted. No Sarvam / ElevenLabs / Whisper calls were made.
  `index_canary.py --execute` was not run. All verification stayed offline.
- **No secrets printed or committed.** `scan_secrets.py` prints filenames only;
  `.env` is git-ignored and was not modified. No credentials appear in staged
  changes.
- **Legacy live path blocked.** `ingest_prepared.py --execute` exits 2 before any
  Pinecone import even with `CONFIRM_PINECONE_WRITE=1` and `PINECONE_API_KEY`
  present (verified by test and subprocess).
- **Post-write reconciliation requires exact ID-set equality.** The canary run is
  marked `success` only when count == 300 AND the enumerated ID set equals the
  manifest-derived expected set exactly.

---

## Remaining limitations

- The live 300-record canary has not been executed against real Pinecone; the
  post-write reconciliation logic is verified only against mocked providers.
- Pinecone latency, retrieval quality, and live correctness are unmeasured. The
  <200 ms target is unverified. No such numbers are claimed.
- The 10,000-record pilot and full corpus remain UNAUTHORIZED and unvalidated.
- Full-corpus scope figures (~24.87M vectors, ~171.67 GB) are EXTRAPOLATED from a
  bounded measured sample, not a verified full-corpus measurement.

---

## Scope verdicts (unchanged unless live evidence exists)

| Scope | Verdict |
|---|---|
| Offline gates | PASSED (independently rerun successfully) |
| 300-record canary | READY (after this hardening pass) |
| Live canary | NOT yet executed or verified |
| 10,000-record pilot | NOT validated / NOT authorized |
| Full En/Hi/Bn corpus | NOT authorized or capacity-validated |
| Voice, retrieval quality, latency benchmarking | Post-index work |
