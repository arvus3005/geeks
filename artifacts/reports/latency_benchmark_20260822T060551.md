# Latency Benchmark — 20260822T060551

- **Target**: deployed:https://hyphen-onyx-sprig.ngrok-free.dev
- **Generated**: 2026-08-22T06:05:51.268071+00:00
- **Queries**: 120 (120 ok, 0 failed)
- **Decisions**: {'abstain': 37, 'allow': 83}

## End-to-end backend latency (ms)

| P50 | P70 | P95 | P99 | P100 | Mean |
|---|---|---|---|---|---|
| 104.2 | 112.8 | 120.0 | 131.3 | 398.7 | 107.6 |

## Per-stage P50 / P100 (ms)

| Stage | P50 | P100 |
|---|---|---|
| answer_extract | 0.0 | 0.2 |
| evidence_check | 0.0 | 0.0 |
| grounding_verify | 0.0 | 0.1 |
| input_guard | 0.0 | 0.0 |
| language_detect | 0.0 | 0.1 |
| local_hybrid_retrieve | 17.4 | 36.1 |
| pinecone_retrieve | 0.0 | 0.0 |
| query_embed | 4.6 | 16.7 |
| rerank | 0.0 | 0.0 |
| serialize | 0.0 | 0.0 |
| total_backend | 24.1 | 47.7 |
| wall_clock_incl_network | 104.2 | 398.7 |
