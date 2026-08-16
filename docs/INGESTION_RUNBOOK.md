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

```bash
# From repo root
uv run python scripts/ingest_prepared.py \
    --manifest artifacts/prepared/<manifest_id>_manifest.json \
    --dry-run

# From a different working directory (must work portably)
cd /tmp
uv run python --project /path/to/geeks \
    /path/to/geeks/scripts/ingest_prepared.py \
    --manifest /path/to/geeks/artifacts/prepared/<manifest_id>_manifest.json \
    --dry-run
```

No credentials are needed. No Pinecone import occurs. Exit 0 means all
manifest/data validations passed.

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
PINECONE_API_KEY=<your-key> \
    uv run pytest tests/integration/ -v -k "not full_corpus"
```

These tests are opt-in and never run in normal CI.

---

## Step 7: Ingest 300-record canary into pilot_v1

**Both guards required.**

```bash
PINECONE_API_KEY=<your-key> CONFIRM_PINECONE_WRITE=1 \
    uv run python scripts/ingest_prepared.py \
    --manifest artifacts/prepared/<manifest_id>_manifest.json \
    --namespace pilot_v1 \
    --execute
```

The script will:
1. Validate manifest (checksum, forbidden fields, contract fingerprint, namespace)
2. Verify data file (checksum)
3. Validate all records via `validate_record()`
4. Describe and validate actual remote index against canonical contract
5. Abort if any field missing/unverifiable/incompatible
6. Only then submit batches (max 96 records, max 1.8 MB each)

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
