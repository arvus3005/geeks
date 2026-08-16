# Chunking Strategy Ablation Report

_Generated: 2026-08-16T14:53:29.405482+00:00_

> **Note**: MRR/Recall figures are OFFLINE PROXY metrics using token-overlap ranking,
> not live Pinecone vector search results.

## Summary

| Strategy | Passages | Chunks | Expansion | Token P95 | MRR@10 (proxy) | R@10 (proxy) | Build(s) |
|---|---|---|---|---|---|---|---|
| passage_native | 2000 | 2000 | 1.0 | 148 | 0.3086 | 1.0000 | 0.25 |
| sentence_aware | 2000 | 2083 | 1.042 | 140 | 0.3053 | 1.0000 | 0.224 |
| fixed_token_overlap | 2000 | 2015 | 1.008 | 148 | 0.3086 | 1.0000 | 0.217 |
| parent_child | 2000 | 4083 | 2.042 | 143 | 0.2381 | 0.7843 | 0.393 |

## Detailed Metrics

### passage_native

- **passages_evaluated**: 2000
- **chunks_produced**: 2000
- **expansion_factor**: 1.0
- **rejected_passages**: 0
- **duplicate_ids**: 10
- **duplicate_texts**: 10
- **token_p50**: 75
- **token_p70**: 86
- **token_p95**: 148
- **token_p100**: 303
- **token_mean**: 82.8
- **meta_bytes_mean**: 136.3
- **meta_bytes_max**: 166
- **planned_pinecone_requests**: 2005
- **projected_storage_bytes**: 8192000
- **build_time_s**: 0.25
- **mrr_at_10_proxy**: 0.3086
- **recall_at_5_proxy**: 0.7843
- **recall_at_10_proxy**: 1.0
- **recall_at_20_proxy**: 1.0

### sentence_aware

- **passages_evaluated**: 2000
- **chunks_produced**: 2083
- **expansion_factor**: 1.042
- **rejected_passages**: 0
- **duplicate_ids**: 10
- **duplicate_texts**: 13
- **token_p50**: 75
- **token_p70**: 86
- **token_p95**: 140
- **token_p100**: 238
- **token_mean**: 81.0
- **meta_bytes_mean**: 136.2
- **meta_bytes_max**: 166
- **planned_pinecone_requests**: 2091
- **projected_storage_bytes**: 8531968
- **build_time_s**: 0.224
- **mrr_at_10_proxy**: 0.3053
- **recall_at_5_proxy**: 0.7549
- **recall_at_10_proxy**: 1.0
- **recall_at_20_proxy**: 1.0

### fixed_token_overlap

- **passages_evaluated**: 2000
- **chunks_produced**: 2015
- **expansion_factor**: 1.008
- **rejected_passages**: 0
- **duplicate_ids**: 10
- **duplicate_texts**: 10
- **token_p50**: 75
- **token_p70**: 86
- **token_p95**: 148
- **token_p100**: 238
- **token_mean**: 82.5
- **meta_bytes_mean**: 136.3
- **meta_bytes_max**: 166
- **planned_pinecone_requests**: 2021
- **projected_storage_bytes**: 8253440
- **build_time_s**: 0.217
- **mrr_at_10_proxy**: 0.3086
- **recall_at_5_proxy**: 0.7843
- **recall_at_10_proxy**: 1.0
- **recall_at_20_proxy**: 1.0

### parent_child

- **passages_evaluated**: 2000
- **chunks_produced**: 4083
- **expansion_factor**: 2.042
- **rejected_passages**: 0
- **duplicate_ids**: 1872
- **duplicate_texts**: 1875
- **token_p50**: 75
- **token_p70**: 86
- **token_p95**: 143
- **token_p100**: 303
- **token_mean**: 81.9
- **meta_bytes_mean**: 136.2
- **meta_bytes_max**: 166
- **planned_pinecone_requests**: 4184
- **projected_storage_bytes**: 16723968
- **build_time_s**: 0.393
- **mrr_at_10_proxy**: 0.2381
- **recall_at_5_proxy**: 0.549
- **recall_at_10_proxy**: 0.7843
- **recall_at_20_proxy**: 1.0
