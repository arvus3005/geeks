# Latency Benchmark — 20260822T072939

- **Target**: local-in-process
- **Generated**: 2026-08-22T07:29:39.157373+00:00
- **Queries**: 120 (120 ok, 0 failed)
- **Decisions**: {'allow': 73, 'abstain': 47}

## End-to-end backend latency (ms)

| P50 | P70 | P95 | P99 | P100 | Mean |
|---|---|---|---|---|---|
| 48.4 | 56.7 | 77.7 | 90.4 | 103.8 | 47.2 |

## Per-stage P50 / P100 (ms)

| Stage | P50 | P100 |
|---|---|---|
| answer_extract | 34.4 | 78.9 |
| evidence_check | 0.0 | 0.0 |
| grounding_verify | 0.0 | 0.1 |
| input_guard | 0.0 | 0.0 |
| language_detect | 0.0 | 0.0 |
| local_hybrid_retrieve | 10.9 | 26.7 |
| pinecone_retrieve | 0.0 | 0.0 |
| query_embed | 2.1 | 3.5 |
| rerank | 0.0 | 0.0 |
| serialize | 0.0 | 0.0 |
| total_backend | 48.4 | 103.8 |
