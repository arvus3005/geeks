# HH Goa 2026 Task 2 — Multilingual RAG System

## Status: Pre-index hardening complete — ready for Gemini 300-record canary preflight

### Multi-Layer Scope & Readiness Matrix
| Layer | Scope | Status | Notes |
|---|---|---|---|
| **1. Offline Code & Test Gates** | Full offline test suite | **PASSED** | 500+ unit, contract, and behavioral tests pass with zero live calls |
| **2. 300-Record Canary** | 100 en + 100 hi + 100 bn | **READY FOR GEMINI PREFLIGHT** | Exact deterministic IDs, fail-closed preflight & resume ownership verification |
| **3. Bounded Starter Pilot** | 10,000 records / 4M tokens | **NOT YET VALIDATED** | Architectural ceiling only; unvalidated pending live canary verification |
| **4. Full En/Hi/Bn Corpus** | ~16.4M passages (~170 GB) | **NOT AUTHORIZED OR CAPACITY-VALIDATED** | Exceeds Starter plan; full corpus local disk requirement unverified |
| **5. Post-Index Product Pipeline** | STT, Voice UI, Reranking serving | **POST-INDEX WORK PENDING** | Requires live index completion and pipeline integration |

### What is implemented and tested offline
- 4 chunking strategies (`passage_native`, `sentence_aware`, `fixed_token_overlap`, `semantic_experimental`)
- `PineconeReranker` with SDK mock coverage; `RetrievalOnlyPassthrough`
- Resumable canary indexer (`scripts/index_canary.py`) with:
  - Fail-closed namespace preflight (aborts on missing/malformed stats)
  - Exact resume ownership verification using deterministic UUIDv5 vector IDs
  - Thread-safe sliding-window token rate limiter (225k tokens/min ceiling)
  - Atomic incremental checkpoints per acknowledged batch
  - Pre-write and post-write freshness reconciliation
- SQLite-backed content deduplication with WAL
- Budget limits and safety gates: dry-run by default; full-mode permanently blocked on Starter plan
- Contract tests: forbidden evaluation fields (`query`, `Answer`, etc.) cannot enter Pinecone records
- Secret scanner (`scripts/scan_secrets.py`) and zero credential leakage

### What is NOT yet done (post-index work)
- Real Pinecone index creation and live vector ingestion
- Live latency and retrieval quality measurement (<200 ms target is unverified)
- Sarvam STT voice pipeline integration
- Generative multilingual answer synthesis

### Architecture
- Vector store: **Pinecone Starter** (AWS us-east-1, serverless integrated embedding)
- Embedding model: `multilingual-e5-large` (server-side; 1024-dim, cosine, field_map `{"text": "chunk_text"}`)
- Reranker: `bge-reranker-v2-m3` (Pinecone inference API)
- Languages: English + 14 Indic MSMARCO-XI configs (`as, bn, gu, hi, kn, ml, mr, ne, or, pa, sa, ta, te, ur`)

### Quick start (offline — no credentials needed)
```bash
# 1. Install dependencies
uv sync --all-extras

# 2. Run offline test suite
uv run pytest tests/unit tests/contract tests/behavioural -q

# 3. Regenerate prepared canary artifacts (deterministic)
uv run python scripts/prepare_canary.py \
    --dataset-revision bf5cdc1f26e581e519018e434db14edd1b77602b \
    --tokenizer-revision 3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3 \
    --split train \
    --seed 42 \
    --chunk-strategy sentence_aware

# 4. Run offline canary dry-run
uv run python scripts/index_canary.py \
    --manifest artifacts/prepared/canary-42-ee540c17772a_manifest.json
```

### Running real integration tests (opt-in, requires PINECONE_API_KEY)
```bash
export PINECONE_API_KEY=<your-key>
PINECONE_SMOKE_TEST=1 uv run pytest tests/integration -v
```

### Key guardrails
- Credentials must come only from environment variables — never CLI arguments or config files
- Full-corpus ingestion is permanently blocked while `PINECONE_PLAN=starter`
- Budget limits are enforced before every API call; retries cannot bypass them
- The under-200 ms latency target is unverified — no fabricated numbers are reported

See `docs/FINAL_PREINDEX_READINESS_REPORT.md` for the canonical pre-index readiness report.
See `docs/INGESTION_RUNBOOK.md` for the live ingestion procedure.
See `docs/PINECONE_SCHEMA.md` for the Pinecone record schema.
See `docs/DATASET_CONTRACT.md` for the data leakage boundary rules.
See `docs/PROJECT_SUMMARY.md` for the project overview.
