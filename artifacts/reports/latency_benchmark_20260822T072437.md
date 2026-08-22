# Latency Benchmark — 20260822T072437

- **Target**: local-in-process
- **Generated**: 2026-08-22T07:24:37.120823+00:00
- **Queries**: 120 (120 ok, 0 failed)
- **Decisions**: {'allow': 73, 'abstain': 47}

## End-to-end backend latency (ms)

| P50 | P70 | P95 | P99 | P100 | Mean |
|---|---|---|---|---|---|
| 48.1 | 56.6 | 75.8 | 88.2 | 103.2 | 46.2 |

## Per-stage P50 / P100 (ms)

| Stage | P50 | P100 |
|---|---|---|
| answer_extract | 33.8 | 79.3 |
| evidence_check | 0.0 | 0.0 |
| grounding_verify | 0.0 | 0.1 |
| input_guard | 0.0 | 0.0 |
| language_detect | 0.0 | 0.0 |
| local_hybrid_retrieve | 10.8 | 25.7 |
| pinecone_retrieve | 0.0 | 0.0 |
| query_embed | 2.0 | 3.3 |
| rerank | 0.0 | 0.0 |
| serialize | 0.0 | 0.0 |
| total_backend | 48.1 | 103.1 |
