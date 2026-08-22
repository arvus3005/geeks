# Latency Benchmark — 20260822T051730

- **Target**: deployed:https://9ce4-202-142-124-101.ngrok-free.app
- **Generated**: 2026-08-22T05:17:30.965708+00:00
- **Queries**: 120 (120 ok, 0 failed)
- **Decisions**: {'allow': 118, 'abstain': 2}

## End-to-end backend latency (ms)

| P50 | P70 | P95 | P99 | P100 | Mean |
|---|---|---|---|---|---|
| 97.4 | 102.2 | 113.8 | 115.4 | 118.5 | 98.4 |

## Per-stage P50 / P100 (ms)

| Stage | P50 | P100 |
|---|---|---|
| answer_extract | 0.0 | 0.2 |
| evidence_check | 0.0 | 0.0 |
| grounding_verify | 0.0 | 0.1 |
| input_guard | 0.0 | 0.0 |
| language_detect | 0.0 | 0.0 |
| local_hybrid_retrieve | 15.3 | 27.6 |
| pinecone_retrieve | 0.0 | 0.0 |
| query_embed | 3.7 | 7.3 |
| rerank | 0.0 | 0.0 |
| serialize | 0.0 | 0.0 |
| total_backend | 20.7 | 35.5 |
| wall_clock_incl_network | 97.4 | 118.5 |
