# Latency Benchmark — 20260822T061403

- **Target**: deployed:https://hyphen-onyx-sprig.ngrok-free.dev
- **Generated**: 2026-08-22T06:14:03.978589+00:00
- **Queries**: 120 (120 ok, 0 failed)
- **Decisions**: {'abstain': 44, 'allow': 76}

## End-to-end backend latency (ms)

| P50 | P70 | P95 | P99 | P100 | Mean |
|---|---|---|---|---|---|
| 119.8 | 128.0 | 149.9 | 155.0 | 164.7 | 121.7 |

## Per-stage P50 / P100 (ms)

| Stage | P50 | P100 |
|---|---|---|
| answer_extract | 11.9 | 43.0 |
| evidence_check | 0.0 | 0.0 |
| grounding_verify | 0.0 | 0.1 |
| input_guard | 0.0 | 0.0 |
| language_detect | 0.0 | 0.0 |
| local_hybrid_retrieve | 17.4 | 38.8 |
| pinecone_retrieve | 0.0 | 0.0 |
| query_embed | 3.5 | 24.9 |
| rerank | 0.0 | 0.0 |
| serialize | 0.0 | 0.0 |
| total_backend | 32.8 | 77.7 |
| wall_clock_incl_network | 119.8 | 164.7 |
