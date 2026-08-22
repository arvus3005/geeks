# Latency Benchmark — 20260822T065922

- **Target**: deployed:https://hyphen-onyx-sprig.ngrok-free.dev
- **Generated**: 2026-08-22T06:59:22.127659+00:00
- **Queries**: 120 (120 ok, 0 failed)
- **Decisions**: {'allow': 73, 'abstain': 47}

## End-to-end backend latency (ms)

| P50 | P70 | P95 | P99 | P100 | Mean |
|---|---|---|---|---|---|
| 138.1 | 157.0 | 194.6 | 516.2 | 1866.4 | 157.2 |

## Per-stage P50 / P100 (ms)

| Stage | P50 | P100 |
|---|---|---|
| answer_extract | 42.9 | 90.8 |
| evidence_check | 0.0 | 0.0 |
| grounding_verify | 0.0 | 0.1 |
| input_guard | 0.0 | 0.1 |
| language_detect | 0.0 | 0.0 |
| local_hybrid_retrieve | 15.0 | 35.2 |
| pinecone_retrieve | 0.0 | 0.0 |
| query_embed | 2.6 | 8.6 |
| rerank | 0.0 | 0.0 |
| serialize | 0.0 | 0.0 |
| total_backend | 59.5 | 115.8 |
| wall_clock_incl_network | 138.1 | 1866.4 |
