# Pre-Index Readiness Report

**Date:** 2026-08-16
**Status:** READY for bounded Pinecone Starter pilot — pending live index creation

## What is ready
- [x] Central budget enforcement module with all Starter limits
- [x] Dry-run default on all scripts; full-mode permanently blocked on Starter
- [x] `--pinecone-api-key` CLI arg removed from all scripts; env-only credentials
- [x] Config-file API key fallback removed
- [x] 4 chunkers implemented and tested (passage-native, sentence-aware, fixed-token, semantic-experimental)
- [x] `PineconeReranker` with candidate-K slicing, usage tracking, fail-closed behaviour
- [x] Resumable ingestion engine with crash-safe checkpoint ordering
- [x] SQLite deduplication with WAL transactions
- [x] Contract tests proving forbidden fields cannot enter Pinecone records
- [x] 208 offline tests pass; 8 integration tests opt-in skipped
- [x] Ruff clean, format clean, mypy clean

## What is NOT done (live indexing session)
- [ ] Real Pinecone index does not exist — no records indexed
- [ ] Pilot manifest not yet generated
- [ ] Live latency unmeasured — <200 ms target unverified
- [ ] Full quality evaluation pending

## Starter operational limits enforced
| Limit              | Ceiling enforced |
|--------------------|-----------------|
| Embedding tokens   | 4,000,000       |
| Records            | 10,000          |
| Storage            | 1.5 GB          |
| Rerank requests    | 500/month       |

## Projected pilot allocation
| Language        | Token share | Records (est.) |
|-----------------|------------|----------------|
| English         | 25%        | ~2,500         |
| Hindi           | 15%        | ~1,500         |
| Bengali         | 15%        | ~1,500         |
| 12 other Indic  | 3.75% each | ~375 each      |

## Safe next action (live indexing session)
1. Set `PINECONE_API_KEY` environment variable.
2. Run: `python scripts/create_pinecone_index.py --pinecone-index msmarco-xi --execute`
   with `CONFIRM_PINECONE_CREATE=1`.
3. Validate the index exists with: `python scripts/describe_pinecone_index.py --pinecone-index msmarco-xi`
4. Generate the pilot manifest.
5. Run pilot ingestion with `--execute` and `CONFIRM_PINECONE_WRITE=1`.
6. Measure live P50/P70/P100 latency.

**Do NOT perform any of the above steps during the pre-index session.**
