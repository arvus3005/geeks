# Latency Benchmark — 20260822T051718

- **Target**: local-in-process
- **Generated**: 2026-08-22T05:17:18.532052+00:00
- **Queries**: 120 (120 ok, 0 failed)
- **Decisions**: {'allow': 118, 'abstain': 2}

## End-to-end backend latency (ms)

| P50 | P70 | P95 | P99 | P100 | Mean |
|---|---|---|---|---|---|
| 13.4 | 17.2 | 28.3 | 32.4 | 38.3 | 14.1 |

## Per-stage P50 / P100 (ms)

| Stage | P50 | P100 |
|---|---|---|
| answer_extract | 0.0 | 0.3 |
| evidence_check | 0.0 | 0.0 |
| grounding_verify | 0.0 | 0.2 |
| input_guard | 0.0 | 0.0 |
| language_detect | 0.0 | 0.0 |
| local_hybrid_retrieve | 10.3 | 34.4 |
| pinecone_retrieve | 0.0 | 0.0 |
| query_embed | 2.0 | 3.3 |
| rerank | 0.0 | 0.0 |
| serialize | 0.0 | 0.0 |
| total_backend | 13.4 | 38.3 |
