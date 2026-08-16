# HH Goa 2026 Task 2 — Multilingual RAG System

## Status: Pre-index readiness — ready for bounded Pinecone Starter pilot

### What is implemented and tested
- 4 chunking strategies (passage-native, sentence-aware, fixed-token-overlap, semantic-experimental)
- `PineconeReranker` with SDK mock coverage; `RetrievalOnlyPassthrough`
- Resumable per-shard ingestion engine with crash-safe checkpointing
- Central budget enforcement (`BudgetGuard`): token/record/storage/rerank limits; fail-closed
- SQLite-backed content deduplication with WAL
- Safety gates: dry-run by default; full-mode permanently blocked on Starter plan
- Contract tests: forbidden fields (query, Answer, etc.) cannot enter Pinecone records
- **Offline test suite passes with no live provider credentials; real Pinecone integration tests are opt-in (separate CI job).**

### What is NOT yet done (requires live indexing session)
- Real Pinecone index creation (blocked until credentials and pilot manifest are approved)
- Real data ingestion — no records have been indexed
- Live latency measurement — the <200 ms target is unverified until measured
- Quality evaluation against a real index

### Architecture
- Vector store: **Pinecone Starter** (AWS us-east-1, serverless integrated embedding)
- Embedding model: `multilingual-e5-large` (server-side; no local model weights)
- Reranker: `bge-reranker-v2-m3` (Pinecone inference API)
- Pipeline: Pinecone Top-8 retrieval → bge reranker → Top-3 → answer stage
- STT: Sarvam (planned; not yet integrated in this session)
- Languages: English + 14 Indic MSMARCO-XI configs:
  `as, bn, gu, hi, kn, ml, mr, ne, or, pa, sa, ta, te, ur`

### Planned pilot allocation (bounded sample — NOT full corpus)
| Language group     | Token budget | Records (est.) |
|--------------------|-------------|----------------|
| English            | 25% (~1M)   | ~2,500         |
| Hindi              | 15% (~600K) | ~1,500         |
| Bengali            | 15% (~600K) | ~1,500         |
| 12 other Indic     | 45% (~150K each) | ~375 each |
| **Total cap**      | **4M tokens** | **10,000 records** |

### Quick start (offline — no credentials needed)
```bash
uv sync --all-extras
uv run pytest tests/unit tests/contract tests/behavioural -q
uv run python scripts/create_pinecone_index.py --pinecone-index msmarco-xi     # dry-run
uv run python scripts/ingest_all.py --mode pilot                                # dry-run
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

See `docs/INGESTION_RUNBOOK.md` for the live ingestion procedure.
See `docs/PINECONE_SCHEMA.md` for the Pinecone record schema.
See `docs/DATASET_CONTRACT.md` for the data leakage boundary rules.
See `docs/PROJECT_SUMMARY.md` for the project overview.
