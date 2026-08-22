# Latency Benchmark — 20260822T062213

- **Target**: deployed:https://hyphen-onyx-sprig.ngrok-free.dev
- **Generated**: 2026-08-22T06:22:13.343551+00:00
- **Queries**: 120 (120 ok, 0 failed)
- **Decisions**: {'abstain': 45, 'allow': 75}

## End-to-end backend latency (ms)

| P50 | P70 | P95 | P99 | P100 | Mean |
|---|---|---|---|---|---|
| 110.5 | 125.2 | 159.1 | 389.1 | 454.9 | 122.0 |

## Per-stage P50 / P100 (ms)

| Stage | P50 | P100 |
|---|---|---|
| answer_extract | 13.9 | 71.3 |
| evidence_check | 0.0 | 0.0 |
| grounding_verify | 0.0 | 0.1 |
| input_guard | 0.0 | 0.0 |
| language_detect | 0.0 | 0.0 |
| local_hybrid_retrieve | 16.1 | 35.7 |
| pinecone_retrieve | 0.0 | 0.0 |
| query_embed | 3.1 | 30.2 |
| rerank | 0.0 | 0.0 |
| serialize | 0.0 | 0.0 |
| total_backend | 33.2 | 95.9 |
| wall_clock_incl_network | 110.5 | 454.9 |
