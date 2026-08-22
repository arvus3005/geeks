# Latency Benchmark — 20260822T150119

- **Target**: local-in-process
- **Generated**: 2026-08-22T15:01:19.265750+00:00
- **Queries**: 120 (120 ok, 0 failed)
- **Decisions**: {'allow': 54, 'abstain': 66}

## End-to-end backend latency (ms)

| P50 | P70 | P95 | P99 | P100 | Mean |
|---|---|---|---|---|---|
| 39.2 | 50.1 | 72.7 | 82.1 | 94.7 | 40.4 |

## Per-stage P50 / P100 (ms)

| Stage | P50 | P100 |
|---|---|---|
| answer_extract | 29.8 | 63.1 |
| evidence_check | 0.0 | 0.0 |
| grounding_verify | 0.0 | 0.1 |
| input_guard | 0.0 | 0.0 |
| language_detect | 0.0 | 0.0 |
| local_hybrid_retrieve | 11.8 | 39.2 |
| pinecone_retrieve | 0.0 | 0.0 |
| query_embed | 2.0 | 5.2 |
| rerank | 0.0 | 0.0 |
| serialize | 0.0 | 0.0 |
| total_backend | 39.2 | 94.7 |
