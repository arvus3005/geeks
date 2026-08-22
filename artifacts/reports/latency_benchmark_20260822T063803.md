# Latency Benchmark — 20260822T063803

- **Target**: deployed:https://hyphen-onyx-sprig.ngrok-free.dev
- **Generated**: 2026-08-22T06:38:03.232295+00:00
- **Queries**: 120 (120 ok, 0 failed)
- **Decisions**: {'allow': 73, 'abstain': 47}

## End-to-end backend latency (ms)

| P50 | P70 | P95 | P99 | P100 | Mean |
|---|---|---|---|---|---|
| 141.5 | 157.2 | 206.9 | 369.9 | 575.9 | 150.2 |

## Per-stage P50 / P100 (ms)

| Stage | P50 | P100 |
|---|---|---|
| answer_extract | 43.5 | 421.4 |
| evidence_check | 0.0 | 0.0 |
| grounding_verify | 0.0 | 0.1 |
| input_guard | 0.0 | 0.1 |
| language_detect | 0.0 | 0.0 |
| local_hybrid_retrieve | 12.0 | 22.2 |
| pinecone_retrieve | 0.0 | 0.0 |
| query_embed | 2.7 | 6.2 |
| rerank | 0.0 | 0.0 |
| serialize | 0.0 | 0.0 |
| total_backend | 57.3 | 450.9 |
| wall_clock_incl_network | 141.5 | 575.9 |
