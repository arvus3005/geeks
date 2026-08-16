# Chunking Strategy Ablation Report

_Generated: 2026-08-16T13:03:28.652825+00:00_

> **Note**: MRR/Recall figures are OFFLINE PROXY metrics using token-overlap ranking,
> not live Pinecone vector search results.

## Summary

| Strategy | Passages | Chunks | Expansion | Token P95 | MRR@10 (proxy) | R@10 (proxy) | Build(s) |
|---|---|---|---|---|---|---|---|
| passage_native | 1000 | 999 | 0.999 | 176 | 0.9500 | 1.0000 | 0.216 |
| sentence_aware | 1000 | 1039 | 1.039 | 161 | 0.9500 | 1.0000 | 0.199 |
| fixed_token_overlap | 1000 | 1032 | 1.032 | 179 | 0.9500 | 1.0000 | 0.203 |
| parent_child | 1000 | 2038 | 2.038 | 168 | 0.9333 | 1.0000 | 0.381 |

## Detailed Metrics

### passage_native

- **passages_evaluated**: 1000
- **chunks_produced**: 999
- **expansion_factor**: 0.999
- **rejected_passages**: 2
- **duplicate_ids**: 2
- **duplicate_texts**: 2
- **token_p50**: 91
- **token_p70**: 109
- **token_p95**: 176
- **token_p100**: 328
- **token_mean**: 100.7
- **meta_bytes_mean**: 290.4
- **meta_bytes_max**: 314
- **planned_pinecone_requests**: 486
- **projected_storage_bytes**: 4091904
- **build_time_s**: 0.216
- **mrr_at_10_proxy**: 0.95
- **recall_at_5_proxy**: 1.0
- **recall_at_10_proxy**: 1.0
- **recall_at_20_proxy**: 1.0

### sentence_aware

- **passages_evaluated**: 1000
- **chunks_produced**: 1039
- **expansion_factor**: 1.039
- **rejected_passages**: 2
- **duplicate_ids**: 2
- **duplicate_texts**: 2
- **token_p50**: 91
- **token_p70**: 108
- **token_p95**: 161
- **token_p100**: 274
- **token_mean**: 98.0
- **meta_bytes_mean**: 290.3
- **meta_bytes_max**: 314
- **planned_pinecone_requests**: 506
- **projected_storage_bytes**: 4255744
- **build_time_s**: 0.199
- **mrr_at_10_proxy**: 0.95
- **recall_at_5_proxy**: 1.0
- **recall_at_10_proxy**: 1.0
- **recall_at_20_proxy**: 1.0

### fixed_token_overlap

- **passages_evaluated**: 1000
- **chunks_produced**: 1032
- **expansion_factor**: 1.032
- **rejected_passages**: 0
- **duplicate_ids**: 2
- **duplicate_texts**: 2
- **token_p50**: 92
- **token_p70**: 109
- **token_p95**: 179
- **token_p100**: 274
- **token_mean**: 100.9
- **meta_bytes_mean**: 290.4
- **meta_bytes_max**: 314
- **planned_pinecone_requests**: 502
- **projected_storage_bytes**: 4227072
- **build_time_s**: 0.203
- **mrr_at_10_proxy**: 0.95
- **recall_at_5_proxy**: 1.0
- **recall_at_10_proxy**: 1.0
- **recall_at_20_proxy**: 1.0

### parent_child

- **passages_evaluated**: 1000
- **chunks_produced**: 2038
- **expansion_factor**: 2.038
- **rejected_passages**: 3
- **duplicate_ids**: 952
- **duplicate_texts**: 952
- **token_p50**: 91
- **token_p70**: 108
- **token_p95**: 168
- **token_p100**: 328
- **token_mean**: 99.3
- **meta_bytes_mean**: 290.3
- **meta_bytes_max**: 314
- **planned_pinecone_requests**: 1025
- **projected_storage_bytes**: 8347648
- **build_time_s**: 0.381
- **mrr_at_10_proxy**: 0.9333
- **recall_at_5_proxy**: 1.0
- **recall_at_10_proxy**: 1.0
- **recall_at_20_proxy**: 1.0
