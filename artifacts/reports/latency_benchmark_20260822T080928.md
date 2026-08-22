# Latency Benchmark — 20260822T080928

- **Target**: local-in-process
- **Generated**: 2026-08-22T08:09:28.519751+00:00
- **Queries**: 120 (120 ok, 0 failed)
- **Decisions**: {'allow': 73, 'abstain': 47}

## End-to-end backend latency (ms)

| P50 | P70 | P95 | P99 | P100 | Mean |
|---|---|---|---|---|---|
| 51.9 | 68.4 | 102.9 | 123.5 | 124.2 | 54.7 |

## Per-stage P50 / P100 (ms)

| Stage | P50 | P100 |
|---|---|---|
| answer_extract | 35.7 | 84.8 |
| evidence_check | 0.0 | 0.0 |
| grounding_verify | 0.0 | 0.1 |
| input_guard | 0.0 | 0.1 |
| language_detect | 0.0 | 0.0 |
| local_hybrid_retrieve | 14.1 | 66.2 |
| pinecone_retrieve | 0.0 | 0.0 |
| query_embed | 2.1 | 5.6 |
| rerank | 0.0 | 0.0 |
| serialize | 0.0 | 0.0 |
| total_backend | 51.9 | 124.2 |
