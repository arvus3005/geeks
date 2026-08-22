# Latency Benchmark — 20260822T072843

- **Target**: local-in-process
- **Generated**: 2026-08-22T07:28:43.633764+00:00
- **Queries**: 120 (120 ok, 0 failed)
- **Decisions**: {'allow': 72, 'abstain': 48}

## End-to-end backend latency (ms)

| P50 | P70 | P95 | P99 | P100 | Mean |
|---|---|---|---|---|---|
| 47.7 | 54.6 | 75.6 | 87.8 | 104.0 | 45.4 |

## Per-stage P50 / P100 (ms)

| Stage | P50 | P100 |
|---|---|---|
| answer_extract | 34.1 | 79.1 |
| evidence_check | 0.0 | 0.0 |
| grounding_verify | 0.0 | 0.1 |
| input_guard | 0.0 | 0.0 |
| language_detect | 0.0 | 0.0 |
| local_hybrid_retrieve | 9.7 | 24.8 |
| pinecone_retrieve | 0.0 | 0.0 |
| query_embed | 2.0 | 3.6 |
| rerank | 0.0 | 0.0 |
| serialize | 0.0 | 0.0 |
| total_backend | 47.7 | 104.0 |
