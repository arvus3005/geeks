# Canary Readiness Report — Pre-Index Sprint

_Generated: 2026-08-16_

## Commits

- `e899ec7` — `Make Pinecone Starter canary ingestion safe and deterministic`
- `0695e3b` — `Fix canary script HF config and update operational docs`

## Files changed (24 total)

**New files:**
- `scripts/prepare_canary.py` — Offline canary preparation, zero Pinecone imports
- `scripts/scan_secrets.py` — Secret scanner, prints filenames only
- `src/hhgoa_rag/ingestion/schema.py` — Authoritative record builder/validator
- `src/hhgoa_rag/ingestion/tokenizer.py` — Real multilingual-e5-large tokenizer
- `tests/unit/test_schema.py` — Schema contract tests
- `tests/unit/test_tokenizer.py` — Tokenizer unit tests

**Modified files:**
- `src/hhgoa_rag/ingestion/engine.py` — Dedup ordering, BudgetGuard wired, batch limits, schema
- `src/hhgoa_rag/ingestion/checkpoint.py` — `num_workers` field, fail-closed on old checkpoints
- `src/hhgoa_rag/ingestion/dedup.py` — No auto-flush; flush only after Pinecone ack
- `scripts/ingest_all.py` — StarterFullModeError before API key, batch-size validation
- `scripts/ingest_shard.py` — Same Starter guard, batch-size validation
- `scripts/resume_ingest.py` — Starter guard, fail-closed on old checkpoints
- `tests/contract/test_ingestion_safety.py` — Tests for Starter gate, batch limits, old checkpoint
- `tests/contract/test_namespace_safety.py` — Dedup ordering, budget-engine, StarterFullModeError
- `tests/integration/test_pinecone_smoke.py` — Fixed namespace `smoke-fixture-v001`, Hindi/Bengali queries
- `tests/unit/test_checkpoint.py` — Added `num_workers` to fixture
- `tests/unit/test_dedup.py` — Updated for no-auto-flush semantics
- `tests/unit/test_sharding.py` — Single-worker enforcement, batch-size, checkpoint compatibility
- `pyproject.toml` + `uv.lock` — `transformers>=4.40.0`, `tokenizers>=0.19.0`, `sentencepiece>=0.2.0`
- `docs/`, `.gitignore` — Updated documentation, added archive/data ignore rules

## Dependencies added

| Package | Version | Justification |
|---------|---------|---------------|
| `transformers` | `>=4.40.0,<5` | Load tokenizer for `intfloat/multilingual-e5-large`; no torch |
| `tokenizers` | `>=0.19.0` | Fast tokenizer backend required by transformers |
| `sentencepiece` | `>=0.2.0` | Required by XLM-RoBERTa tokenizer used in multilingual-e5-large |

## Test results

**254 passed, 10 skipped**

All 10 skips are integration tests that correctly skip without `PINECONE_API_KEY` + `PINECONE_SMOKE_TEST=1`.

## Quality gates

| Gate | Result |
|------|--------|
| `uv run pytest tests/ -q -rs` | ✅ 254 passed, 10 skipped |
| `uv run ruff check src tests scripts bench` | ✅ All checks passed |
| `uv run ruff format --check src tests scripts bench` | ✅ 95 files already formatted |
| `uv run mypy src` | ✅ Success: no issues found in 47 source files |

## Tokenizer

| Property | Value |
|----------|-------|
| Name | `intfloat/multilingual-e5-large` |
| Revision | `HEAD` (pin with `--tokenizer-revision <sha>` for executable readiness) |
| Fingerprint | `8fb3093e93d4bc40` |
| Model input limit | 512 tokens (includes BOS/EOS + `passage:` prefix special tokens) |
| Source | HuggingFace Hub — downloaded successfully |

## Production-path proofs

| Proof | Evidence |
|-------|---------|
| Budget runs before provider writes | `engine.py` `flush()`: `guard.check_upsert()` → `store.upsert_records()` → `guard.commit_upsert()` |
| Ack before dedup commit | `dedup.mark_seen()` + `dedup.flush()` only after `upsert_records()` returns successfully |
| Ack before budget commit | `commit_upsert()` only after `upsert_records()` returns successfully |
| No request > 96 records | `IngestionConfig.__post_init__` validates `1 ≤ batch_size ≤ 96`; flush triggered inside the per-chunk loop |
| Byte ceiling enforced | `_estimate_batch_bytes()` checked before `store.upsert_records()` |
| Retries don't double-count | `check_upsert()` re-checked on every retry attempt; `commit_upsert()` called only after final success |
| Failed writes don't commit dedup | `dedup.flush()` (SQLite commit) never called on the failure branch |
| Old checkpoints fail closed | `IngestCheckpoint.load()` raises `RuntimeError` if `num_workers` absent from JSON |
| Single-worker enforced | `_validate_single_worker_for_starter()` called in `ingest_shard()`; all CLI entry points hardcode `num_workers=1` |
| Integration tests skip honestly | `pytest.skip()` when `PINECONE_API_KEY` or `PINECONE_SMOKE_TEST=1` absent |

## Request limits

| Limit | Value |
|-------|-------|
| Maximum records per request | 96 (Pinecone hard limit) |
| Maximum serialised request bytes | 1,800,000 (conservative ceiling below Pinecone's 2 MB limit) |

## Canary per-language record totals

❌ **Canary not yet generated** — see blocker below.

## Prepared-data checksum / manifest checksum

❌ Not available — canary not generated yet.

## Dry-run proof

```
$ uv run python scripts/ingest_all.py --mode pilot
DRY-RUN: no Pinecone writes. Missing: --execute flag, CONFIRM_PINECONE_WRITE=1 env var
{
  "mode": "pilot",
  "status": "dry_run",
  "plan": "starter",
  "note": "offline preparation only — no Pinecone writes"
}

$ PINECONE_PLAN=starter uv run python scripts/ingest_all.py --mode full --confirm-full-ingest \
    CONFIRM_FULL_INGEST=YES_I_APPROVE_FULL_CORPUS CONFIRM_PINECONE_WRITE=1
ERROR: Full-corpus ingestion is permanently blocked on Pinecone Starter plan.
PINECONE_PLAN='starter' is active. No flag or environment variable can override this.
Switch to a paid Pinecone plan to enable full-corpus ingestion.
```

No Pinecone client was constructed in either case. `PINECONE_API_KEY` was not accessed.

## Credential scan

```
$ uv run python scripts/scan_secrets.py
Credential scan: no secrets detected in tracked source files.
```

## Unresolved blockers

### BLOCKER — Dataset config structure mismatch

`ai4bharat/MSMARCO-XI` currently exposes only a `'default'` HuggingFace config rather
than the per-language configs (`'hi'`, `'bn'`, etc.) assumed by the existing engine.
The actual record structure inside `'default'` (specifically how language identity is
encoded per row) could not be confirmed due to dataset streaming latency.

Until this is resolved:

- `scripts/prepare_canary.py` will fail when trying to stream `'hi'` or `'bn'` as
  separate configs.
- The existing `engine.py` has the same assumption and will fail on real dataset access.

**Action required before running the canary:**

1. Inspect the record structure of the `'default'` config:
   ```bash
   uv run python -c "
   from datasets import load_dataset
   ds = load_dataset('ai4bharat/MSMARCO-XI', 'default', split='train', streaming=True)
   row = next(iter(ds))
   print(list(row.keys()))
   "
   ```
2. Determine how language identity is encoded (dedicated field, separate configs at
   a specific revision, etc.).
3. Either confirm a pinned revision has per-language configs, or update the parser
   to filter by language within the `'default'` config.
4. Then run:
   ```bash
   uv run python scripts/prepare_canary.py \
       --dataset-revision <pinned-sha> \
       --hf-config default \
       --seed 42
   ```

## Confirmation

| Confirmation | Status |
|---|---|
| No Pinecone index was created | ✅ Confirmed |
| No Pinecone records were written | ✅ Confirmed |
| No provider quota was consumed | ✅ Confirmed (HF tokenizer downloads are free) |

## Verdict: NOT READY

The code infrastructure is fully implemented, tested, and safe. The blocker is the
dataset config structure mismatch — the actual language-filtering approach within
`ai4bharat/MSMARCO-XI` must be confirmed before the real canary can be generated.
All other pre-index readiness defects have been corrected.

---

## Next steps (once dataset structure is confirmed)

```bash
# 1. Export a newly rotated Pinecone key
export PINECONE_API_KEY=<your-newly-rotated-key>

# 2. Generate the offline canary (no Pinecone calls)
uv run python scripts/prepare_canary.py \
    --dataset-revision <pinned-sha> \
    --hf-config <confirmed-config-name> \
    --seed 42

# 3. Inspect manifest — confirm ready_for_write: true
cat artifacts/prepared/<manifest_id>_manifest.json | python -m json.tool | grep ready_for_write

# 4. Create the integrated-embedding index (requires explicit confirmation)
export CONFIRM_PINECONE_CREATE=1
uv run python scripts/create_pinecone_index.py \
    --pinecone-index msmarco-xi \
    --cloud aws \
    --region us-east-1 \
    --execute

# 5. Validate the index
uv run python scripts/validate_pinecone_config.py

# 6. Run bounded real smoke integration tests
export PINECONE_SMOKE_TEST=1
uv run pytest tests/integration/ -v
```

Do not run the canary write command until you have:
1. Confirmed the dataset config structure and field layout
2. Generated a valid manifest with `ready_for_write: true`
3. Received team approval
