# Latency Benchmark — 20260822T072225

- **Target**: local-in-process
- **Generated**: 2026-08-22T07:22:25.783085+00:00
- **Queries**: 120 (120 ok, 0 failed)
- **Decisions**: {'allow': 73, 'abstain': 47}

## End-to-end backend latency (ms)

| P50 | P70 | P95 | P99 | P100 | Mean |
|---|---|---|---|---|---|
| 53.0 | 72.7 | 95.4 | 104.2 | 122.6 | 56.1 |

## Per-stage P50 / P100 (ms)

| Stage | P50 | P100 |
|---|---|---|
| answer_extract | 39.1 | 87.6 |
| evidence_check | 0.0 | 0.0 |
| grounding_verify | 0.0 | 0.1 |
| input_guard | 0.0 | 0.0 |
| language_detect | 0.0 | 0.0 |
| local_hybrid_retrieve | 11.3 | 56.9 |
| pinecone_retrieve | 0.0 | 0.0 |
| query_embed | 2.1 | 4.1 |
| rerank | 0.0 | 0.0 |
| serialize | 0.0 | 0.0 |
| total_backend | 53.0 | 122.6 |
