# Latency Benchmark — 20260822T063231

- **Target**: deployed:https://hyphen-onyx-sprig.ngrok-free.dev
- **Generated**: 2026-08-22T06:32:31.941048+00:00
- **Queries**: 120 (120 ok, 0 failed)
- **Decisions**: {'abstain': 97, 'allow': 23}

## End-to-end backend latency (ms)

| P50 | P70 | P95 | P99 | P100 | Mean |
|---|---|---|---|---|---|
| 121.4 | 144.0 | 209.4 | 230.9 | 238.6 | 133.2 |

## Per-stage P50 / P100 (ms)

| Stage | P50 | P100 |
|---|---|---|
| answer_extract | 21.2 | 127.0 |
| evidence_check | 0.0 | 0.0 |
| grounding_verify | 0.0 | 0.1 |
| input_guard | 0.0 | 0.0 |
| language_detect | 0.0 | 0.0 |
| local_hybrid_retrieve | 15.2 | 34.5 |
| pinecone_retrieve | 0.0 | 0.0 |
| query_embed | 3.0 | 25.6 |
| rerank | 0.0 | 0.0 |
| serialize | 0.0 | 0.0 |
| total_backend | 40.5 | 154.4 |
| wall_clock_incl_network | 121.4 | 238.6 |
