# Latency Benchmark — 20260822T053913

- **Target**: deployed:https://9ce4-202-142-124-101.ngrok-free.app
- **Generated**: 2026-08-22T05:39:13.622088+00:00
- **Queries**: 120 (120 ok, 0 failed)
- **Decisions**: {'abstain': 37, 'allow': 83}

## End-to-end backend latency (ms)

| P50 | P70 | P95 | P99 | P100 | Mean |
|---|---|---|---|---|---|
| 101.0 | 107.5 | 120.0 | 126.7 | 179.6 | 102.0 |

## Per-stage P50 / P100 (ms)

| Stage | P50 | P100 |
|---|---|---|
| answer_extract | 0.0 | 0.2 |
| evidence_check | 0.0 | 0.0 |
| grounding_verify | 0.0 | 0.1 |
| input_guard | 0.0 | 0.0 |
| language_detect | 0.0 | 0.0 |
| local_hybrid_retrieve | 17.6 | 37.5 |
| pinecone_retrieve | 0.0 | 0.0 |
| query_embed | 4.1 | 7.9 |
| rerank | 0.0 | 0.0 |
| serialize | 0.0 | 0.0 |
| total_backend | 24.7 | 47.3 |
| wall_clock_incl_network | 101.0 | 179.6 |
