# Latency Benchmark — 20260822T081412

- **Target**: local-in-process
- **Generated**: 2026-08-22T08:14:12.061969+00:00
- **Queries**: 120 (120 ok, 0 failed)
- **Decisions**: {'allow': 73, 'abstain': 47}

## End-to-end backend latency (ms)

| P50 | P70 | P95 | P99 | P100 | Mean |
|---|---|---|---|---|---|
| 40.2 | 49.6 | 65.7 | 78.9 | 95.5 | 39.9 |

## Per-stage P50 / P100 (ms)

| Stage | P50 | P100 |
|---|---|---|
| answer_extract | 26.3 | 60.5 |
| evidence_check | 0.0 | 0.0 |
| grounding_verify | 0.0 | 0.1 |
| input_guard | 0.0 | 0.0 |
| language_detect | 0.0 | 0.0 |
| local_hybrid_retrieve | 10.6 | 30.9 |
| pinecone_retrieve | 0.0 | 0.0 |
| query_embed | 2.0 | 4.0 |
| rerank | 0.0 | 0.0 |
| serialize | 0.0 | 0.0 |
| total_backend | 40.2 | 95.5 |
