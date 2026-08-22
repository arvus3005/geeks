# Latency Benchmark — 20260822T065527

- **Target**: deployed:https://hyphen-onyx-sprig.ngrok-free.dev
- **Generated**: 2026-08-22T06:55:27.786555+00:00
- **Queries**: 120 (120 ok, 0 failed)
- **Decisions**: {'allow': 73, 'abstain': 47}

## End-to-end backend latency (ms)

| P50 | P70 | P95 | P99 | P100 | Mean |
|---|---|---|---|---|---|
| 127.6 | 145.4 | 175.0 | 179.3 | 186.2 | 129.8 |

## Per-stage P50 / P100 (ms)

| Stage | P50 | P100 |
|---|---|---|
| answer_extract | 42.7 | 86.5 |
| evidence_check | 0.0 | 0.0 |
| grounding_verify | 0.0 | 0.1 |
| input_guard | 0.0 | 0.0 |
| language_detect | 0.0 | 0.0 |
| local_hybrid_retrieve | 12.1 | 23.9 |
| pinecone_retrieve | 0.0 | 0.0 |
| query_embed | 2.6 | 7.0 |
| rerank | 0.0 | 0.0 |
| serialize | 0.0 | 0.0 |
| total_backend | 56.2 | 112.9 |
| wall_clock_incl_network | 127.6 | 186.2 |
