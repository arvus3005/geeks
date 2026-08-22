# Latency Benchmark — 20260822T072809

- **Target**: local-in-process
- **Generated**: 2026-08-22T07:28:09.513749+00:00
- **Queries**: 120 (120 ok, 0 failed)
- **Decisions**: {'allow': 73, 'abstain': 47}

## End-to-end backend latency (ms)

| P50 | P70 | P95 | P99 | P100 | Mean |
|---|---|---|---|---|---|
| 56.2 | 67.8 | 109.7 | 120.1 | 121.3 | 55.8 |

## Per-stage P50 / P100 (ms)

| Stage | P50 | P100 |
|---|---|---|
| answer_extract | 38.1 | 113.5 |
| evidence_check | 0.0 | 0.0 |
| grounding_verify | 0.0 | 0.1 |
| input_guard | 0.0 | 0.1 |
| language_detect | 0.0 | 0.0 |
| local_hybrid_retrieve | 10.8 | 32.9 |
| pinecone_retrieve | 0.0 | 0.0 |
| query_embed | 2.0 | 6.8 |
| rerank | 0.0 | 0.0 |
| serialize | 0.0 | 0.0 |
| total_backend | 56.2 | 121.3 |
