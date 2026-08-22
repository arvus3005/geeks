# Latency Benchmark — 20260822T063648

- **Target**: deployed:https://hyphen-onyx-sprig.ngrok-free.dev
- **Generated**: 2026-08-22T06:36:48.336614+00:00
- **Queries**: 120 (120 ok, 0 failed)
- **Decisions**: {'abstain': 69, 'allow': 51}

## End-to-end backend latency (ms)

| P50 | P70 | P95 | P99 | P100 | Mean |
|---|---|---|---|---|---|
| 118.3 | 137.0 | 184.7 | 209.7 | 217.5 | 124.6 |

## Per-stage P50 / P100 (ms)

| Stage | P50 | P100 |
|---|---|---|
| answer_extract | 25.3 | 113.2 |
| evidence_check | 0.0 | 0.0 |
| grounding_verify | 0.0 | 0.1 |
| input_guard | 0.0 | 0.0 |
| language_detect | 0.0 | 0.0 |
| local_hybrid_retrieve | 12.6 | 22.4 |
| pinecone_retrieve | 0.0 | 0.0 |
| query_embed | 3.0 | 7.1 |
| rerank | 0.0 | 0.0 |
| serialize | 0.0 | 0.0 |
| total_backend | 40.2 | 137.4 |
| wall_clock_incl_network | 118.3 | 217.5 |
