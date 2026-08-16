# Chunking Strategy Ablation Report

_Generated: 2026-08-16T13:04:29.247940+00:00_

> **Note**: MRR/Recall figures are OFFLINE PROXY metrics using token-overlap ranking,
> not live Pinecone vector search results.

## Summary

| Strategy | Passages | Chunks | Expansion | Token P95 | MRR@10 (proxy) | R@10 (proxy) | Build(s) |
|---|---|---|---|---|---|---|---|
| passage_native | 1000 | 999 | 0.999 | 193 | 1.0000 | 1.0000 | 0.125 |
| sentence_aware | 1000 | 1037 | 1.037 | 183 | 1.0000 | 1.0000 | 0.116 |
| fixed_token_overlap | 1000 | 1012 | 1.012 | 197 | 1.0000 | 1.0000 | 0.115 |
| parent_child | 1000 | 2036 | 2.036 | 187 | 1.0000 | 1.0000 | 0.217 |

## Detailed Metrics

### passage_native

- **passages_evaluated**: 1000
- **chunks_produced**: 999
- **expansion_factor**: 0.999
- **rejected_passages**: 2
- **duplicate_ids**: 2
- **duplicate_texts**: 2
- **token_p50**: 100
- **token_p70**: 120
- **token_p95**: 193
- **token_p100**: 452
- **token_mean**: 110.5
- **meta_bytes_mean**: 302.4
- **meta_bytes_max**: 316
- **planned_pinecone_requests**: 486
- **projected_storage_bytes**: 4091904
- **build_time_s**: 0.125
- **mrr_at_10_proxy**: 1.0
- **recall_at_5_proxy**: 1.0
- **recall_at_10_proxy**: 1.0
- **recall_at_20_proxy**: 1.0

### sentence_aware

- **passages_evaluated**: 1000
- **chunks_produced**: 1037
- **expansion_factor**: 1.037
- **rejected_passages**: 2
- **duplicate_ids**: 2
- **duplicate_texts**: 2
- **token_p50**: 100
- **token_p70**: 120
- **token_p95**: 183
- **token_p100**: 384
- **token_mean**: 107.6
- **meta_bytes_mean**: 302.4
- **meta_bytes_max**: 316
- **planned_pinecone_requests**: 506
- **projected_storage_bytes**: 4247552
- **build_time_s**: 0.116
- **mrr_at_10_proxy**: 1.0
- **recall_at_5_proxy**: 1.0
- **recall_at_10_proxy**: 1.0
- **recall_at_20_proxy**: 1.0

### fixed_token_overlap

- **passages_evaluated**: 1000
- **chunks_produced**: 1012
- **expansion_factor**: 1.012
- **rejected_passages**: 0
- **duplicate_ids**: 4
- **duplicate_texts**: 4
- **token_p50**: 100
- **token_p70**: 120
- **token_p95**: 197
- **token_p100**: 384
- **token_mean**: 111.0
- **meta_bytes_mean**: 302.4
- **meta_bytes_max**: 316
- **planned_pinecone_requests**: 491
- **projected_storage_bytes**: 4145152
- **build_time_s**: 0.115
- **mrr_at_10_proxy**: 1.0
- **recall_at_5_proxy**: 1.0
- **recall_at_10_proxy**: 1.0
- **recall_at_20_proxy**: 1.0

### parent_child

- **passages_evaluated**: 1000
- **chunks_produced**: 2036
- **expansion_factor**: 2.036
- **rejected_passages**: 3
- **duplicate_ids**: 963
- **duplicate_texts**: 963
- **token_p50**: 100
- **token_p70**: 120
- **token_p95**: 187
- **token_p100**: 452
- **token_mean**: 109.1
- **meta_bytes_mean**: 302.4
- **meta_bytes_max**: 316
- **planned_pinecone_requests**: 1025
- **projected_storage_bytes**: 8339456
- **build_time_s**: 0.217
- **mrr_at_10_proxy**: 1.0
- **recall_at_5_proxy**: 1.0
- **recall_at_10_proxy**: 1.0
- **recall_at_20_proxy**: 1.0
