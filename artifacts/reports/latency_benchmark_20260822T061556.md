# Latency Benchmark — 20260822T061556

- **Target**: deployed:https://hyphen-onyx-sprig.ngrok-free.dev
- **Generated**: 2026-08-22T06:15:56.226947+00:00
- **Queries**: 120 (120 ok, 0 failed)
- **Decisions**: {'abstain': 43, 'allow': 77}

## End-to-end backend latency (ms)

| P50 | P70 | P95 | P99 | P100 | Mean |
|---|---|---|---|---|---|
| 114.2 | 126.9 | 159.0 | 169.2 | 181.1 | 118.9 |

## Per-stage P50 / P100 (ms)

| Stage | P50 | P100 |
|---|---|---|
| answer_extract | 12.8 | 69.5 |
| evidence_check | 0.0 | 0.0 |
| grounding_verify | 0.0 | 0.2 |
| input_guard | 0.0 | 0.1 |
| language_detect | 0.0 | 0.0 |
| local_hybrid_retrieve | 15.8 | 45.9 |
| pinecone_retrieve | 0.0 | 0.0 |
| query_embed | 3.2 | 15.2 |
| rerank | 0.0 | 0.0 |
| serialize | 0.0 | 0.0 |
| total_backend | 31.8 | 98.9 |
| wall_clock_incl_network | 114.2 | 181.1 |
