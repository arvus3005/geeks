# Ingestion Runbook — MSMARCO-XI → Pinecone

**Mandatory sequence. Follow in order. Do not skip steps.**

---

## Step 1: Fresh-clone setup

```bash
git clone https://github.com/arvus3005/geeks
cd geeks
uv sync --all-extras
```

Verify no secrets in the codebase:

```bash
uv run python scripts/scan_secrets.py
```

---

## Step 2: Regenerate / verify prepared artifacts

Prepare the 300-record canary (100 English + 100 Hindi + 100 Bengali).
Dataset and tokenizer revisions are pinned.

```bash
uv run python scripts/prepare_canary.py \
    --dataset-revision bf5cdc1f26e581e519018e434db14edd1b77602b \
    --tokenizer-revision 3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3 \
    --split train \
    --seed 42 \
    --chunk-strategy sentence_aware
```

Outputs:
- `artifacts/prepared/<manifest_id>_records.jsonl`
- `artifacts/prepared/<manifest_id>_manifest.json`

Verify determinism (run twice into separate directories, compare):

```bash
uv run python scripts/prepare_canary.py --dataset-revision bf5cdc1f... \
    --tokenizer-revision 3d7cfd... --seed 42 --output-dir /tmp/run1
uv run python scripts/prepare_canary.py --dataset-revision bf5cdc1f... \
    --tokenizer-revision 3d7cfd... --seed 42 --output-dir /tmp/run2
diff <(sha256sum /tmp/run1/*_records.jsonl) <(sha256sum /tmp/run2/*_records.jsonl)
# must output: (empty — identical)
```

Run ID/linkage audit:

```bash
uv run python scripts/audit_ids.py \
    --records artifacts/prepared/<manifest_id>_records.jsonl \
    --output-file artifacts/reports/id_audit.json
```

---

## Step 3: Offline dry-run

**Preferred path** (uses the approved canary indexer with full record validation):
```bash
uv run python scripts/index_canary.py \
    --manifest artifacts/prepared/<manifest_id>_manifest.json
```

**Legacy-compatible path** (lower-level, still supported):
```bash
uv run python scripts/ingest_prepared.py \
    --manifest artifacts/prepared/<manifest_id>_manifest.json \
    --dry-run
```

No credentials are needed. No Pinecone import occurs in either dry-run path.
Exit 0 means all manifest/data validations (including `validate_record()` on every record) passed.

---

## Step 4: Create Pinecone index with explicit confirmation

**Both guards are required. `--execute` alone exits 2 (fail-closed).**

```bash
CONFIRM_PINECONE_CREATE=1 PINECONE_API_KEY=<your-key> \
    uv run python scripts/create_pinecone_index.py \
    --pinecone-index msmarco-xi \
    --execute
```

Dry-run (no API key needed):

```bash
uv run python scripts/create_pinecone_index.py --pinecone-index msmarco-xi
```

---

## Step 5: Validate actual index contract

```bash
PINECONE_API_KEY=<your-key> \
    uv run python scripts/describe_pinecone_index.py --pinecone-index msmarco-xi
```

Verify output against canonical contract in `pinecone_contract.py`:
- model = `multilingual-e5-large`
- dimension = 1024
- metric = `cosine`
- field_map = `{"text": "chunk_text"}`
- write_parameters = `{"input_type": "passage", "truncate": "NONE"}`
- read_parameters = `{"input_type": "query", "truncate": "NONE"}`
- cloud = `aws`
- region = `us-east-1`

---

## Step 6: Run bounded smoke integration tests

```bash
PINECONE_API_KEY=<your-key> PINECONE_SMOKE_TEST=1 \
    uv run pytest tests/integration/ -v -k "not full_corpus"
```

These tests are opt-in and never run in normal offline CI. They require explicit confirmation via `PINECONE_SMOKE_TEST=1`.

---

## Step 7: Ingest 300-record canary into pilot_v1

**Both guards required. `PINECONE_API_KEY` must already be exported in the environment.**

### Approved canary path (scripts/index_canary.py)

Foreground (recommended for first run):
```bash
export PINECONE_API_KEY=<your-key>
CONFIRM_PINECONE_WRITE=1 \
    uv run python scripts/index_canary.py \
    --manifest artifacts/prepared/<manifest_id>_manifest.json \
    --execute --resume --concurrency 4
```

Background execution (nohup + checkpoint resume as primary recovery):
```bash
export PINECONE_API_KEY=<your-key>
nohup env CONFIRM_PINECONE_WRITE=1 \
    uv run python scripts/index_canary.py \
    --manifest artifacts/prepared/<manifest_id>_manifest.json \
    --execute --resume > logs/index_canary.log 2>&1 &
```

> **Important**: Use `nohup env VAR=val ...` syntax so environment variables are
> correctly passed when `nohup` detaches from the shell. `PINECONE_API_KEY` must
> already be exported (via `export`) before running the background command.

`index_canary.py` will:
1. Validate manifest (checksum, contract fingerprint, all provenance fields)
2. Verify data file (checksum)
3. Validate all records via `validate_record()` — before any Pinecone client is constructed
4. Validate remote index contract against canonical values
5. Only then submit batches (max 96 records, 225,000 tokens/min ceiling)
6. Save checkpoint after each acknowledged batch (resumable)
7. Perform freshness reconciliation — must reach exactly 300 vectors

### Legacy-compatible alternative (scripts/ingest_prepared.py)

```bash
PINECONE_API_KEY=<your-key> CONFIRM_PINECONE_WRITE=1 \
    uv run python scripts/ingest_prepared.py \
    --manifest artifacts/prepared/<manifest_id>_manifest.json \
    --namespace pilot_v1 \
    --execute
```

---


## Step 8: Reconcile counts

```bash
PINECONE_API_KEY=<your-key> \
    uv run python scripts/reconcile_corpus.py \
    --pinecone-index msmarco-xi \
    --namespace pilot_v1 \
    --expected 300
```

---

## Step 9: Post-index evaluation

Run offline ID audit on the ingested records:

```bash
uv run python scripts/audit_ids.py \
    --records artifacts/prepared/<manifest_id>_records.jsonl
```

---

## Step 10: Stop before expansion

Do NOT proceed to full corpus indexing until:
- Canary count reconciles (300 records in pilot_v1)
- ID audit passes
- All quality gates documented
- Full corpus budget approved (Pinecone plan upgrade required for >300 records)

---

## Safety rules

- `CONFIRM_PINECONE_CREATE=1` + `--execute` required for index creation
- `CONFIRM_PINECONE_WRITE=1` + `--execute` required for ingestion
- `--execute` alone exits 2 (never silently downgraded to dry-run)
- Namespace `pilot_v1` is the only permitted namespace for canary writes
- Full corpus mode on Starter plan is permanently blocked (no bypass)
- Live integration tests only via `workflow_dispatch` in GitHub Actions
