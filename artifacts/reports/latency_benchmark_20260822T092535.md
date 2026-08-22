# Latency Benchmark — 20260822T092535

- **Target**: local-in-process
- **Generated**: 2026-08-22T09:25:35.612537+00:00
- **Queries**: 120 (120 ok, 0 failed)
- **Decisions**: {'allow': 75, 'abstain': 45}

## End-to-end backend latency (ms)

| P50 | P70 | P95 | P99 | P100 | Mean |
|---|---|---|---|---|---|
| 39.4 | 50.0 | 73.1 | 102.6 | 105.7 | 40.7 |

## Per-stage P50 / P100 (ms)

| Stage | P50 | P100 |
|---|---|---|
| answer_extract | 27.2 | 72.6 |
| evidence_check | 0.0 | 0.0 |
| grounding_verify | 0.0 | 0.0 |
| input_guard | 0.0 | 0.0 |
| language_detect | 0.0 | 0.0 |
| local_hybrid_retrieve | 10.9 | 35.2 |
| pinecone_retrieve | 0.0 | 0.0 |
| query_embed | 1.9 | 3.5 |
| rerank | 0.0 | 0.0 |
| serialize | 0.0 | 0.0 |
| total_backend | 39.4 | 105.7 |
