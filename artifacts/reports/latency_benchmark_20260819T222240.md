# Latency Benchmark — 20260819T222240

- **Target**: deployed:https://hhgoa-rag-d3fw.onrender.com
- **Generated**: 2026-08-19T22:22:40.621121+00:00
- **Queries**: 100 (100 ok, 0 failed)
- **Decisions**: {'allow': 98, 'abstain': 2}

## End-to-end backend latency (ms)

| P50 | P70 | P95 | P99 | P100 | Mean |
|---|---|---|---|---|---|
| 326.3 | 355.7 | 413.2 | 511.8 | 511.8 | 338.2 |

## Per-stage P50 / P100 (ms)

| Stage | P50 | P100 |
|---|---|---|
| answer_extract | 0.1 | 15.8 |
| evidence_check | 0.0 | 0.0 |
| grounding_verify | 0.1 | 0.2 |
| input_guard | 0.0 | 0.1 |
| language_detect | 0.0 | 0.1 |
| pinecone_retrieve | 31.1 | 128.0 |
| query_embed | 7.6 | 52.3 |
| rerank | 0.0 | 0.0 |
| serialize | 0.0 | 0.0 |
| total_backend | 42.4 | 170.2 |
| wall_clock_incl_network | 326.3 | 511.8 |
