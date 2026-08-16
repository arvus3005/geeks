# Ingestion Runbook: MSMARCO-XI → Pinecone

> **Note:** Qdrant instructions below are **legacy and unused**.
> Active ingestion targets Pinecone with integrated `multilingual-e5-large` embedding.

## Phase overview

| Phase | Command | Authorization |
|-------|---------|---------------|
| Offline preparation | `scripts/prepare_canary.py` | None — no provider calls |
| Smoke test (live) | `pytest tests/integration/` | `PINECONE_API_KEY` + `PINECONE_SMOKE_TEST=1` |
| Canary (300 records) | via prepared manifest | Explicit approval from team lead |
| Pilot expansion | `ingest_all.py --mode pilot` | Infrastructure + cost review |
| Full corpus | **Permanently blocked on Starter plan** | Paid plan + team sign-off |

**NEVER run pilot or full ingestion without explicit authorization.**
**Full-corpus mode is permanently blocked on Pinecone Starter plan — no flags can bypass this.**

## Step 1: Offline canary preparation (no network writes)

```bash
uv run python scripts/prepare_canary.py \
    --dataset-revision <pinned-hf-commit-sha> \
    --seed 42 \
    --output-json
```

Inspect the manifest at `artifacts/prepared/<id>_manifest.json`.
Confirm `ready_for_write: true` before proceeding.

## Step 2: Create Pinecone index

```bash
CONFIRM_PINECONE_CREATE=1 PINECONE_API_KEY=<key> \
    uv run python scripts/create_pinecone_index.py \
    --pinecone-index msmarco-xi \
    --execute
```

## Step 3: Validate index configuration

```bash
PINECONE_API_KEY=<key> uv run python scripts/validate_pinecone_config.py
```

## Step 4: Bounded real smoke test

```bash
PINECONE_API_KEY=<key> PINECONE_SMOKE_TEST=1 \
    uv run pytest tests/integration/ -v
```

Uses fixed namespace `smoke-fixture-v001`. Repeated runs are idempotent.
Never creates random namespaces. Never deletes the index.

## Step 5: 300-record tri-language canary (pending approval)

Command will be provided after smoke test passes and team approval is granted.
Default: 100 English + 100 Hindi + 100 Bengali records from `hi` and `bn` source configs.

## Step 6: Reconciliation

```bash
PINECONE_API_KEY=<key> uv run python scripts/reconcile_corpus.py
```

## Step 7: Pilot expansion (planned — not started)

```bash
# Single worker — multi-worker sharding is deferred
CONFIRM_PINECONE_WRITE=1 PINECONE_API_KEY=<key> \
    uv run python scripts/ingest_all.py \
    --mode pilot \
    --execute \
    --dataset-revision <sha>
```

## Resuming interrupted ingestion

```bash
uv run python scripts/resume_ingest.py --checkpoint artifacts/checkpoints/<ckpt>.json
```

Old checkpoints without `num_workers` will be rejected — fail closed.

## Deduplication strategy

- English passages: globally deduplicated across all source configs.
- Native language passages: deduplicated per-language.
- Both use SHA-256 of NFC-normalised text.
- SQLite WAL journal — hashes committed to DB only after Pinecone acknowledges the batch.

## Secret management

- Credentials come exclusively from environment variables.
- Never print, log, commit or save credential values.
- Run `uv run python scripts/scan_secrets.py` before committing.

---

## Legacy: Qdrant instructions (unused)

The instructions below are from an earlier design phase and are **not used**.
Qdrant is not deployed. Retained for reference only.

```bash
# LEGACY — DO NOT USE
docker compose up -d qdrant
uv run python scripts/create_qdrant_collection.py ...
uv run python scripts/ingest_all.py --mode smoke --config configs/smoke.yaml
```
