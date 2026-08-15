# Ingestion Runbook: MSMARCO-XI into Qdrant

## Phase Gating

| Phase | Command | Authorization needed? |
|-------|---------|----------------------|
| Smoke | `ingest_all.py --mode smoke` | No — uses local fixtures |
| Pilot | `ingest_all.py --mode pilot` | Dataset HF access required |
| Full | `ingest_all.py --mode full --confirm-full-ingest` | Infrastructure + cost approval |

**NEVER run pilot or full ingestion without explicit authorization.**

## Step 1: Smoke Run (CI-safe)

```bash
docker compose up -d qdrant
uv run python scripts/create_qdrant_collection.py --config configs/smoke.yaml --force
uv run python scripts/ingest_all.py --mode smoke --config configs/smoke.yaml
uv run python scripts/validate_qdrant_collection.py --config configs/smoke.yaml
uv run python scripts/audit_ids.py --collection msmarco_xi_passages_smoke_v001
```

## Step 2: Pilot Run (requires HF access)

```bash
# For one language config, one split:
uv run python scripts/ingest_shard.py \
  --config-lang bn \
  --split train \
  --shard 0 \
  --mode pilot \
  --collection msmarco_xi_passages_pilot_v001 \
  --checkpoint-dir artifacts/checkpoints/
```

## Step 3: Full Corpus (requires approved infrastructure)

See `estimate_capacity.py` output for infrastructure requirements before provisioning.

```bash
# CHECK ESTIMATE FIRST:
uv run python scripts/estimate_capacity.py

# Only after infrastructure approval and cost sign-off:
uv run python scripts/ingest_all.py --mode full --confirm-full-ingest
```

## Step 4: Validate and Alias Switch

```bash
uv run python scripts/validate_qdrant_collection.py --collection <TARGET>
uv run python scripts/reconcile_corpus.py --config configs/<MODE>.yaml
# Only for production collections (not smoke/pilot):
uv run python scripts/swap_collection_alias.py --target-collection <TARGET>
```

## Step 5: Snapshot

```bash
uv run python scripts/snapshot_qdrant.py --collection <TARGET>
```

## Resuming Interrupted Ingestion

```bash
uv run python scripts/resume_ingest.py --checkpoint artifacts/checkpoints/latest.json
```

## Deduplication Strategy

- English passages: globally deduplicated across all 14 Indic configs
- Translated passages: language-scoped dedup
- Both use SHA-256 of NFC-normalized text
- SQLite WAL journal, batch commits (500 records per transaction)
