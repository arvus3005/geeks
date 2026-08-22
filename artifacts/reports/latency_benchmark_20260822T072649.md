# Latency Benchmark — 20260822T072649

- **Target**: local-in-process
- **Generated**: 2026-08-22T07:26:49.990792+00:00
- **Queries**: 120 (120 ok, 0 failed)
- **Decisions**: {'allow': 73, 'abstain': 47}

## End-to-end backend latency (ms)

| P50 | P70 | P95 | P99 | P100 | Mean |
|---|---|---|---|---|---|
| 47.6 | 56.4 | 77.7 | 86.6 | 105.5 | 46.3 |

## Per-stage P50 / P100 (ms)

| Stage | P50 | P100 |
|---|---|---|
| answer_extract | 33.8 | 77.6 |
| evidence_check | 0.0 | 0.0 |
| grounding_verify | 0.0 | 0.1 |
| input_guard | 0.0 | 0.0 |
| language_detect | 0.0 | 0.0 |
| local_hybrid_retrieve | 10.0 | 30.6 |
| pinecone_retrieve | 0.0 | 0.0 |
| query_embed | 2.0 | 3.3 |
| rerank | 0.0 | 0.0 |
| serialize | 0.0 | 0.0 |
| total_backend | 47.6 | 105.5 |
