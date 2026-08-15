# HH Goa 2026 Task 2 — Multilingual RAG System

## Status: Phase 2 (Ingestion Foundation) — In Progress

### What works
- Smoke ingestion: all 15 language codes (14 Indic + en), deterministic point IDs, hybrid retrieval
- Unit tests: passage IDs, normalizer, dedup, chunkers, sparse encoding, fake embedder
- Contract tests: leakage prevention, collection safety, alias protection
- Behavioural tests: retrieval edge cases (no Qdrant required)
- Integration tests: smoke Qdrant workflow (requires local Docker)

### Quick start
```bash
uv sync --frozen --all-extras
docker compose up -d qdrant
uv run python scripts/create_qdrant_collection.py --config configs/smoke.yaml --force
uv run python scripts/ingest_all.py --mode smoke --config configs/smoke.yaml
uv run python scripts/validate_qdrant_collection.py --config configs/smoke.yaml
uv run pytest tests/unit tests/contract tests/behavioural -q
uv run pytest tests/integration -q  # requires Qdrant
```

### Architecture
- Dense: `intfloat/multilingual-e5-small` (384-dim, cosine)
- Sparse: stable SHA-256 token IDs (production: will use FastEmbed BM25)
- Retrieval: Qdrant hybrid RRF (dense + sparse prefetch)
- Full corpus: 11.4M rows × 14 Indic configs = ~160M passage occurrences (estimate)

See `docs/INGESTION_RUNBOOK.md` for full ingestion procedure.
See `docs/DATASET_CONTRACT.md` for leakage boundary rules.
See `docs/QDRANT_SCHEMA.md` for collection schema.
