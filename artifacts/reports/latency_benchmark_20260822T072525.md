# Latency Benchmark — 20260822T072525

- **Target**: local-in-process
- **Generated**: 2026-08-22T07:25:25.691111+00:00
- **Queries**: 120 (120 ok, 0 failed)
- **Decisions**: {'allow': 54, 'abstain': 66}

## End-to-end backend latency (ms)

| P50 | P70 | P95 | P99 | P100 | Mean |
|---|---|---|---|---|---|
| 48.0 | 62.2 | 81.6 | 87.9 | 99.9 | 48.5 |

## Per-stage P50 / P100 (ms)

| Stage | P50 | P100 |
|---|---|---|
| answer_extract | 35.4 | 81.1 |
| evidence_check | 0.0 | 0.0 |
| grounding_verify | 0.0 | 0.1 |
| input_guard | 0.0 | 0.0 |
| language_detect | 0.0 | 0.0 |
| local_hybrid_retrieve | 10.9 | 41.3 |
| pinecone_retrieve | 0.0 | 0.0 |
| query_embed | 2.1 | 3.6 |
| rerank | 0.0 | 0.0 |
| serialize | 0.0 | 0.0 |
| total_backend | 48.0 | 99.9 |
