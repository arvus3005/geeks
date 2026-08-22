# Latency Benchmark — 20260822T051554

- **Target**: deployed:https://9ce4-202-142-124-101.ngrok-free.app
- **Generated**: 2026-08-22T05:15:54.145906+00:00
- **Queries**: 120 (120 ok, 0 failed)
- **Decisions**: {'allow': 118, 'abstain': 2}

## End-to-end backend latency (ms)

| P50 | P70 | P95 | P99 | P100 | Mean |
|---|---|---|---|---|---|
| 140.7 | 193.9 | 272.1 | 343.7 | 367.7 | 156.9 |

## Per-stage P50 / P100 (ms)

| Stage | P50 | P100 |
|---|---|---|
| answer_extract | 0.0 | 0.1 |
| evidence_check | 0.0 | 0.0 |
| grounding_verify | 0.0 | 0.1 |
| input_guard | 0.0 | 0.0 |
| language_detect | 0.0 | 0.0 |
| local_hybrid_retrieve | 63.2 | 279.4 |
| pinecone_retrieve | 0.0 | 0.0 |
| query_embed | 3.0 | 6.4 |
| rerank | 0.0 | 0.0 |
| serialize | 0.0 | 0.0 |
| total_backend | 67.2 | 286.3 |
| wall_clock_incl_network | 140.7 | 367.7 |
