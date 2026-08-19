# Latency Benchmark — 20260819T204853

- **Target**: local-in-process
- **Generated**: 2026-08-19T20:48:53.050316+00:00
- **Queries**: 32 (32 ok, 0 failed)
- **Decisions**: {'allow': 31, 'abstain': 1}

## End-to-end backend latency (ms)

| P50 | P70 | P95 | P99 | P100 | Mean |
|---|---|---|---|---|---|
| 334.8 | 350.6 | 1779.7 | 1913.4 | 1913.4 | 475.1 |

## Per-stage P50 / P100 (ms)

| Stage | P50 | P100 |
|---|---|---|
| answer_extract | 0.1 | 0.2 |
| evidence_check | 0.0 | 0.0 |
| grounding_verify | 0.1 | 0.2 |
| input_guard | 0.0 | 0.1 |
| language_detect | 0.0 | 0.0 |
| pinecone_retrieve | 327.2 | 1904.7 |
| query_embed | 4.9 | 9.3 |
| rerank | 0.0 | 0.0 |
| serialize | 0.0 | 0.0 |
| total_backend | 334.8 | 1913.4 |
