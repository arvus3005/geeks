# Latency Benchmark — 20260822T060421

- **Target**: deployed:https://hyphen-onyx-sprig.ngrok-free.dev
- **Generated**: 2026-08-22T06:04:21.727798+00:00
- **Queries**: 120 (120 ok, 0 failed)
- **Decisions**: {'abstain': 97, 'allow': 23}

## End-to-end backend latency (ms)

| P50 | P70 | P95 | P99 | P100 | Mean |
|---|---|---|---|---|---|
| 111.2 | 114.8 | 140.3 | 155.9 | 413.7 | 114.6 |

## Per-stage P50 / P100 (ms)

| Stage | P50 | P100 |
|---|---|---|
| answer_extract | 11.9 | 40.7 |
| evidence_check | 0.0 | 0.0 |
| grounding_verify | 0.0 | 0.1 |
| input_guard | 0.0 | 0.0 |
| language_detect | 0.0 | 0.0 |
| local_hybrid_retrieve | 17.2 | 36.9 |
| pinecone_retrieve | 0.0 | 0.0 |
| query_embed | 3.3 | 7.3 |
| rerank | 0.0 | 0.0 |
| serialize | 0.0 | 0.0 |
| total_backend | 30.6 | 71.9 |
| wall_clock_incl_network | 111.2 | 413.7 |
