# Chunking Strategy Ablation Report

_Generated: 2026-08-16T14:53:33.495799+00:00_

> **Note**: MRR/Recall figures are OFFLINE PROXY metrics using token-overlap ranking,
> not live Pinecone vector search results.

## Summary

| Strategy | Passages | Chunks | Expansion | Token P95 | MRR@10 (proxy) | R@10 (proxy) | Build(s) |
|---|---|---|---|---|---|---|---|
| passage_native | 2000 | 1996 | 0.998 | 197 | 1.0000 | 1.0000 | 0.274 |
| sentence_aware | 2000 | 2083 | 1.042 | 189 | 1.0000 | 1.0000 | 0.275 |
| fixed_token_overlap | 2000 | 2021 | 1.01 | 201 | 1.0000 | 1.0000 | 0.267 |
| parent_child | 2000 | 4079 | 2.039 | 192 | 1.0000 | 1.0000 | 0.495 |

## Detailed Metrics

### passage_native

- **passages_evaluated**: 2000
- **chunks_produced**: 1996
- **expansion_factor**: 0.998
- **rejected_passages**: 8
- **duplicate_ids**: 8
- **duplicate_texts**: 8
- **token_p50**: 99
- **token_p70**: 116
- **token_p95**: 197
- **token_p100**: 421
- **token_mean**: 108.1
- **meta_bytes_mean**: 302.3
- **meta_bytes_max**: 322
- **planned_pinecone_requests**: 2001
- **projected_storage_bytes**: 8175616
- **build_time_s**: 0.274
- **mrr_at_10_proxy**: 1.0
- **recall_at_5_proxy**: 1.0
- **recall_at_10_proxy**: 1.0
- **recall_at_20_proxy**: 1.0

### sentence_aware

- **passages_evaluated**: 2000
- **chunks_produced**: 2083
- **expansion_factor**: 1.042
- **rejected_passages**: 3
- **duplicate_ids**: 16
- **duplicate_texts**: 16
- **token_p50**: 99
- **token_p70**: 116
- **token_p95**: 189
- **token_p100**: 283
- **token_mean**: 106.4
- **meta_bytes_mean**: 302.1
- **meta_bytes_max**: 322
- **planned_pinecone_requests**: 2103
- **projected_storage_bytes**: 8531968
- **build_time_s**: 0.275
- **mrr_at_10_proxy**: 1.0
- **recall_at_5_proxy**: 1.0
- **recall_at_10_proxy**: 1.0
- **recall_at_20_proxy**: 1.0

### fixed_token_overlap

- **passages_evaluated**: 2000
- **chunks_produced**: 2021
- **expansion_factor**: 1.01
- **rejected_passages**: 1
- **duplicate_ids**: 8
- **duplicate_texts**: 8
- **token_p50**: 99
- **token_p70**: 117
- **token_p95**: 201
- **token_p100**: 436
- **token_mean**: 109.4
- **meta_bytes_mean**: 302.2
- **meta_bytes_max**: 324
- **planned_pinecone_requests**: 2023
- **projected_storage_bytes**: 8278016
- **build_time_s**: 0.267
- **mrr_at_10_proxy**: 1.0
- **recall_at_5_proxy**: 1.0
- **recall_at_10_proxy**: 1.0
- **recall_at_20_proxy**: 1.0

### parent_child

- **passages_evaluated**: 2000
- **chunks_produced**: 4079
- **expansion_factor**: 2.039
- **rejected_passages**: 7
- **duplicate_ids**: 1938
- **duplicate_texts**: 1938
- **token_p50**: 99
- **token_p70**: 116
- **token_p95**: 192
- **token_p100**: 421
- **token_mean**: 107.2
- **meta_bytes_mean**: 302.2
- **meta_bytes_max**: 322
- **planned_pinecone_requests**: 4189
- **projected_storage_bytes**: 16707584
- **build_time_s**: 0.495
- **mrr_at_10_proxy**: 1.0
- **recall_at_5_proxy**: 1.0
- **recall_at_10_proxy**: 1.0
- **recall_at_20_proxy**: 1.0
