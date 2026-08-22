# Latency Benchmark — 20260822T072907

- **Target**: local-in-process
- **Generated**: 2026-08-22T07:29:07.570339+00:00
- **Queries**: 120 (120 ok, 0 failed)
- **Decisions**: {'allow': 72, 'abstain': 48}

## End-to-end backend latency (ms)

| P50 | P70 | P95 | P99 | P100 | Mean |
|---|---|---|---|---|---|
| 46.8 | 53.8 | 74.6 | 85.0 | 101.2 | 44.8 |

## Per-stage P50 / P100 (ms)

| Stage | P50 | P100 |
|---|---|---|
| answer_extract | 33.7 | 78.0 |
| evidence_check | 0.0 | 0.0 |
| grounding_verify | 0.0 | 0.1 |
| input_guard | 0.0 | 0.0 |
| language_detect | 0.0 | 0.0 |
| local_hybrid_retrieve | 9.5 | 24.2 |
| pinecone_retrieve | 0.0 | 0.0 |
| query_embed | 2.0 | 3.3 |
| rerank | 0.0 | 0.0 |
| serialize | 0.0 | 0.0 |
| total_backend | 46.8 | 101.2 |
