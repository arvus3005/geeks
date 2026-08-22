# Latency Benchmark — 20260822T082324

- **Target**: local-in-process
- **Generated**: 2026-08-22T08:23:24.243598+00:00
- **Queries**: 120 (120 ok, 0 failed)
- **Decisions**: {'allow': 74, 'abstain': 46}

## End-to-end backend latency (ms)

| P50 | P70 | P95 | P99 | P100 | Mean |
|---|---|---|---|---|---|
| 40.9 | 52.3 | 78.5 | 98.0 | 107.6 | 41.8 |

## Per-stage P50 / P100 (ms)

| Stage | P50 | P100 |
|---|---|---|
| answer_extract | 27.7 | 77.0 |
| evidence_check | 0.0 | 0.0 |
| grounding_verify | 0.0 | 0.4 |
| input_guard | 0.0 | 0.0 |
| language_detect | 0.0 | 0.0 |
| local_hybrid_retrieve | 11.3 | 32.4 |
| pinecone_retrieve | 0.0 | 0.0 |
| query_embed | 1.9 | 3.8 |
| rerank | 0.0 | 0.0 |
| serialize | 0.0 | 0.0 |
| total_backend | 40.9 | 107.6 |
