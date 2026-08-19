# Latency Benchmark — 20260819T205335

- **Target**: deployed:https://hhgoa-rag-d3fw.onrender.com
- **Generated**: 2026-08-19T20:53:35.347757+00:00
- **Queries**: 32 (32 ok, 0 failed)
- **Decisions**: {'allow': 31, 'abstain': 1}

## End-to-end backend latency (ms)

| P50 | P70 | P95 | P99 | P100 | Mean |
|---|---|---|---|---|---|
| 307.0 | 332.0 | 410.8 | 427.1 | 427.1 | 325.5 |

## Per-stage P50 / P100 (ms)

| Stage | P50 | P100 |
|---|---|---|
| answer_extract | 0.1 | 0.1 |
| evidence_check | 0.0 | 0.0 |
| grounding_verify | 0.1 | 0.1 |
| input_guard | 0.0 | 0.1 |
| language_detect | 0.0 | 0.0 |
| pinecone_retrieve | 33.5 | 93.7 |
| query_embed | 7.4 | 116.8 |
| rerank | 0.0 | 0.0 |
| serialize | 0.0 | 0.0 |
| total_backend | 42.9 | 151.6 |
| wall_clock_incl_network | 307.0 | 427.1 |
