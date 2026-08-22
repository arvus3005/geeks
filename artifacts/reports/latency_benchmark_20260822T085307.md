# Latency Benchmark — 20260822T085307

- **Target**: local-in-process
- **Generated**: 2026-08-22T08:53:07.788298+00:00
- **Queries**: 120 (120 ok, 0 failed)
- **Decisions**: {'allow': 74, 'abstain': 46}

## End-to-end backend latency (ms)

| P50 | P70 | P95 | P99 | P100 | Mean |
|---|---|---|---|---|---|
| 41.8 | 51.0 | 78.3 | 103.9 | 105.7 | 42.1 |

## Per-stage P50 / P100 (ms)

| Stage | P50 | P100 |
|---|---|---|
| answer_extract | 28.4 | 75.6 |
| evidence_check | 0.0 | 0.0 |
| grounding_verify | 0.0 | 0.1 |
| input_guard | 0.0 | 0.1 |
| language_detect | 0.0 | 0.0 |
| local_hybrid_retrieve | 11.4 | 29.7 |
| pinecone_retrieve | 0.0 | 0.0 |
| query_embed | 1.9 | 5.3 |
| rerank | 0.0 | 0.0 |
| serialize | 0.0 | 0.0 |
| total_backend | 41.8 | 105.7 |
