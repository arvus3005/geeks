# Pre-Index Readiness — MSMARCO-XI → Pinecone Starter

_Last updated: 2026-08-16_

## Status summary

| Component | Status |
|-----------|--------|
| Pinecone Starter plan enforcement | ✅ Implemented and tested |
| Full-corpus block on Starter | ✅ `StarterFullModeError` before API key check — no bypass possible |
| Single-worker enforcement (canary/pilot) | ✅ Implemented and tested |
| Batch size bounded 1–96 | ✅ Validated in `IngestionConfig.__post_init__` |
| Budget guard wired into write path | ✅ `BudgetGuard.check_upsert()` before every Pinecone write |
| Budget committed after ack only | ✅ `commit_upsert()` after successful upsert only |
| Dedup commit ordering | ✅ Hashes committed to SQLite only after Pinecone ack |
| Checkpoint includes `num_workers` | ✅ Old checkpoints without `num_workers` fail closed |
| Authoritative record schema | ✅ `src/hhgoa_rag/ingestion/schema.py` — `build_record()` used everywhere |
| Forbidden-field recursive check | ✅ `validate_record()` checks nested structures |
| Real tokenizer (multilingual-e5-large) | ✅ HuggingFace transformers — no torch dependency |
| Exact token accounting per record | ✅ `token_length` field in every prepared record |
| Canary preparation script | ✅ `scripts/prepare_canary.py` — zero Pinecone imports |
| Secret scanning | ✅ `scripts/scan_secrets.py` — prints filenames only |
| Fixed smoke namespace | ✅ `smoke-fixture-v001` — idempotent, not random |
| Pinecone index exists | ❌ Not created — no provider calls made |
| Records written | ❌ None — no Pinecone quota consumed |
| Live canary | ❌ Pending approval |

## No Pinecone index or records exist

As of this document, **no Pinecone index has been created and no records have been written.**
The Pinecone API has not been called. No provider quota has been consumed.

## Required operational sequence

1. **Offline preparation** — `uv run python scripts/prepare_canary.py --dataset-revision <sha>`
2. **Manifest validation** — inspect `artifacts/prepared/<id>_manifest.json`, confirm `ready_for_write: true`
3. **Pinecone index creation** — `uv run python scripts/create_pinecone_index.py --execute` + `CONFIRM_PINECONE_CREATE=1`
4. **Index validation** — `uv run python scripts/validate_pinecone_config.py`
5. **Bounded smoke test** — `PINECONE_SMOKE_TEST=1 PINECONE_API_KEY=<key> uv run pytest tests/integration/ -v`
6. **300-record canary** — pending approval (command provided at completion)
7. **Reconciliation** — `uv run python scripts/reconcile_corpus.py`
8. **Controlled pilot expansion** — planned

## Safety guarantees

- Full-corpus mode is **permanently blocked** on Starter plan (`StarterFullModeError` raised before any external call).
- No CLI flag, environment variable or confirmation token can bypass the Starter gate.
- Canary and pilot modes require `num_workers == 1`. Multi-worker sharding is deferred.
- `BudgetGuard.check_upsert()` is called before every write attempt including retries.
- `commit_upsert()` is called only after Pinecone acknowledges the batch.
- Dedup hashes are committed to SQLite only after Pinecone acknowledges the batch.
- Old checkpoints missing `num_workers` fail closed — they cannot be resumed.
- Batch size is capped at 96 records and validated at construction time.
- Request bytes are measured and compared against a conservative ceiling below Pinecone's 2 MB limit.

## Not yet live-tested (no index exists)

- Latency measurements (no index)
- Retrieval quality (no index)
- Full MSMARCO-XI corpus capacity (out of scope for Starter)
- Multi-worker parallel sharding

## Qdrant

Qdrant is **not used** in this system. The file `docs/QDRANT_SCHEMA.md` is retained as a
legacy design artefact. All active ingestion targets Pinecone with integrated
`multilingual-e5-large` embedding.
