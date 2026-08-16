# Chunking Strategy Ablation Report

_Generated: 2026-08-16T14:53:15.402276+00:00_

> **Note**: MRR/Recall figures are OFFLINE PROXY metrics using token-overlap ranking,
> not live Pinecone vector search results.

## Summary

| Strategy | Passages | Chunks | Expansion | Token P95 | MRR@10 (proxy) | R@10 (proxy) | Build(s) |
|---|---|---|---|---|---|---|---|
| passage_native | 2000 | 1994 | 0.997 | 182 | 1.0000 | 1.0000 | 0.284 |
| sentence_aware | 2000 | 2120 | 1.06 | 177 | 1.0000 | 1.0000 | 0.274 |
| fixed_token_overlap | 2000 | 2090 | 1.045 | 184 | 1.0000 | 1.0000 | 0.292 |
| parent_child | 2000 | 4114 | 2.057 | 180 | 1.0000 | 1.0000 | 0.502 |

## Detailed Metrics

### passage_native

- **passages_evaluated**: 2000
- **chunks_produced**: 1994
- **expansion_factor**: 0.997
- **rejected_passages**: 12
- **duplicate_ids**: 9
- **duplicate_texts**: 9
- **token_p50**: 90
- **token_p70**: 107
- **token_p95**: 182
- **token_p100**: 328
- **token_mean**: 98.8
- **meta_bytes_mean**: 290.9
- **meta_bytes_max**: 314
- **planned_pinecone_requests**: 1999
- **projected_storage_bytes**: 8167424
- **build_time_s**: 0.284
- **mrr_at_10_proxy**: 1.0
- **recall_at_5_proxy**: 1.0
- **recall_at_10_proxy**: 1.0
- **recall_at_20_proxy**: 1.0

### sentence_aware

- **passages_evaluated**: 2000
- **chunks_produced**: 2120
- **expansion_factor**: 1.06
- **rejected_passages**: 2
- **duplicate_ids**: 51
- **duplicate_texts**: 52
- **token_p50**: 90
- **token_p70**: 108
- **token_p95**: 177
- **token_p100**: 372
- **token_mean**: 99.8
- **meta_bytes_mean**: 290.3
- **meta_bytes_max**: 314
- **planned_pinecone_requests**: 2135
- **projected_storage_bytes**: 8683520
- **build_time_s**: 0.274
- **mrr_at_10_proxy**: 1.0
- **recall_at_5_proxy**: 1.0
- **recall_at_10_proxy**: 1.0
- **recall_at_20_proxy**: 1.0

### fixed_token_overlap

- **passages_evaluated**: 2000
- **chunks_produced**: 2090
- **expansion_factor**: 1.045
- **rejected_passages**: 4
- **duplicate_ids**: 46
- **duplicate_texts**: 46
- **token_p50**: 91
- **token_p70**: 108
- **token_p95**: 184
- **token_p100**: 292
- **token_mean**: 100.6
- **meta_bytes_mean**: 290.4
- **meta_bytes_max**: 314
- **planned_pinecone_requests**: 2080
- **projected_storage_bytes**: 8560640
- **build_time_s**: 0.292
- **mrr_at_10_proxy**: 1.0
- **recall_at_5_proxy**: 1.0
- **recall_at_10_proxy**: 1.0
- **recall_at_20_proxy**: 1.0

### parent_child

- **passages_evaluated**: 2000
- **chunks_produced**: 4114
- **expansion_factor**: 2.057
- **rejected_passages**: 8
- **duplicate_ids**: 1962
- **duplicate_texts**: 1963
- **token_p50**: 90
- **token_p70**: 107
- **token_p95**: 180
- **token_p100**: 372
- **token_mean**: 99.3
- **meta_bytes_mean**: 290.6
- **meta_bytes_max**: 314
- **planned_pinecone_requests**: 4218
- **projected_storage_bytes**: 16850944
- **build_time_s**: 0.502
- **mrr_at_10_proxy**: 1.0
- **recall_at_5_proxy**: 1.0
- **recall_at_10_proxy**: 1.0
- **recall_at_20_proxy**: 1.0
