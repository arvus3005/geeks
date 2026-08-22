# Latency Benchmark — 20260822T072352

- **Target**: local-in-process
- **Generated**: 2026-08-22T07:23:52.388585+00:00
- **Queries**: 120 (120 ok, 0 failed)
- **Decisions**: {'allow': 72, 'abstain': 48}

## End-to-end backend latency (ms)

| P50 | P70 | P95 | P99 | P100 | Mean |
|---|---|---|---|---|---|
| 46.5 | 53.7 | 73.7 | 88.1 | 101.2 | 44.6 |

## Per-stage P50 / P100 (ms)

| Stage | P50 | P100 |
|---|---|---|
| answer_extract | 33.7 | 77.4 |
| evidence_check | 0.0 | 0.0 |
| grounding_verify | 0.0 | 0.1 |
| input_guard | 0.0 | 0.0 |
| language_detect | 0.0 | 0.0 |
| local_hybrid_retrieve | 10.4 | 33.2 |
| pinecone_retrieve | 0.0 | 0.0 |
| query_embed | 2.0 | 3.4 |
| rerank | 0.0 | 0.0 |
| serialize | 0.0 | 0.0 |
| total_backend | 46.5 | 101.2 |
