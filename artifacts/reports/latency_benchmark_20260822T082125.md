# Latency Benchmark — 20260822T082125

- **Target**: local-in-process
- **Generated**: 2026-08-22T08:21:25.969563+00:00
- **Queries**: 120 (120 ok, 0 failed)
- **Decisions**: {'allow': 74, 'abstain': 46}

## End-to-end backend latency (ms)

| P50 | P70 | P95 | P99 | P100 | Mean |
|---|---|---|---|---|---|
| 41.4 | 51.5 | 72.2 | 98.7 | 106.9 | 41.4 |

## Per-stage P50 / P100 (ms)

| Stage | P50 | P100 |
|---|---|---|
| answer_extract | 27.3 | 75.3 |
| evidence_check | 0.0 | 0.0 |
| grounding_verify | 0.0 | 0.1 |
| input_guard | 0.0 | 0.0 |
| language_detect | 0.0 | 0.0 |
| local_hybrid_retrieve | 11.1 | 28.2 |
| pinecone_retrieve | 0.0 | 0.0 |
| query_embed | 2.1 | 6.0 |
| rerank | 0.0 | 0.0 |
| serialize | 0.0 | 0.0 |
| total_backend | 41.4 | 106.9 |
