# Latency Benchmark — 20260819T200348

- **Target**: local-in-process
- **Generated**: 2026-08-19T20:03:48.695129+00:00
- **Queries**: 32 (32 ok, 0 failed)
- **Decisions**: {'allow': 32}

## End-to-end backend latency (ms)

| P50 | P70 | P95 | P99 | P100 | Mean |
|---|---|---|---|---|---|
| 302.7 | 308.0 | 612.3 | 1087.7 | 1087.7 | 340.3 |

## Per-stage P50 / P100 (ms)

| Stage | P50 | P100 |
|---|---|---|
| answer_extract | 0.0 | 0.2 |
| evidence_check | 0.0 | 0.0 |
| grounding_verify | 0.0 | 0.1 |
| input_guard | 0.0 | 0.1 |
| language_detect | 0.0 | 0.0 |
| pinecone_retrieve | 292.7 | 1079.1 |
| query_embed | 5.3 | 8.4 |
| rerank | 0.0 | 0.0 |
| serialize | 0.0 | 0.0 |
| total_backend | 302.7 | 1087.7 |
