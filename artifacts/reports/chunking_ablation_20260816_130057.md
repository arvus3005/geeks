# Chunking Strategy Ablation Report

_Generated: 2026-08-16T13:00:57.735140+00:00_

> **Note**: MRR/Recall figures are OFFLINE PROXY metrics using token-overlap ranking,
> not live Pinecone vector search results.

## Summary

| Strategy | Passages | Chunks | Expansion | Token P95 | MRR@10 (proxy) | R@10 (proxy) | Build(s) |
|---|---|---|---|---|---|---|---|
| passage_native | 0 | 0 | 0.0 | 0.0 | N/A | N/A | 0.0 |
| sentence_aware | 0 | 0 | 0.0 | 0.0 | N/A | N/A | 0.0 |
| fixed_token_overlap | 0 | 0 | 0.0 | 0.0 | N/A | N/A | 0.0 |
| parent_child | 0 | 0 | 0.0 | 0.0 | N/A | N/A | 0.0 |

## Detailed Metrics

### passage_native

- **passages_evaluated**: 0
- **chunks_produced**: 0
- **expansion_factor**: 0.0
- **rejected_passages**: 0
- **duplicate_ids**: 0
- **duplicate_texts**: 0
- **token_p50**: 0.0
- **token_p70**: 0.0
- **token_p95**: 0.0
- **token_p100**: 0
- **token_mean**: 0.0
- **meta_bytes_mean**: 0.0
- **meta_bytes_max**: 0
- **planned_pinecone_requests**: 0
- **projected_storage_bytes**: 0
- **build_time_s**: 0.0
- **mrr_at_10_proxy**: None
- **recall_at_5_proxy**: None
- **recall_at_10_proxy**: None
- **recall_at_20_proxy**: None

### sentence_aware

- **passages_evaluated**: 0
- **chunks_produced**: 0
- **expansion_factor**: 0.0
- **rejected_passages**: 0
- **duplicate_ids**: 0
- **duplicate_texts**: 0
- **token_p50**: 0.0
- **token_p70**: 0.0
- **token_p95**: 0.0
- **token_p100**: 0
- **token_mean**: 0.0
- **meta_bytes_mean**: 0.0
- **meta_bytes_max**: 0
- **planned_pinecone_requests**: 0
- **projected_storage_bytes**: 0
- **build_time_s**: 0.0
- **mrr_at_10_proxy**: None
- **recall_at_5_proxy**: None
- **recall_at_10_proxy**: None
- **recall_at_20_proxy**: None

### fixed_token_overlap

- **passages_evaluated**: 0
- **chunks_produced**: 0
- **expansion_factor**: 0.0
- **rejected_passages**: 0
- **duplicate_ids**: 0
- **duplicate_texts**: 0
- **token_p50**: 0.0
- **token_p70**: 0.0
- **token_p95**: 0.0
- **token_p100**: 0
- **token_mean**: 0.0
- **meta_bytes_mean**: 0.0
- **meta_bytes_max**: 0
- **planned_pinecone_requests**: 0
- **projected_storage_bytes**: 0
- **build_time_s**: 0.0
- **mrr_at_10_proxy**: None
- **recall_at_5_proxy**: None
- **recall_at_10_proxy**: None
- **recall_at_20_proxy**: None

### parent_child

- **passages_evaluated**: 0
- **chunks_produced**: 0
- **expansion_factor**: 0.0
- **rejected_passages**: 0
- **duplicate_ids**: 0
- **duplicate_texts**: 0
- **token_p50**: 0.0
- **token_p70**: 0.0
- **token_p95**: 0.0
- **token_p100**: 0
- **token_mean**: 0.0
- **meta_bytes_mean**: 0.0
- **meta_bytes_max**: 0
- **planned_pinecone_requests**: 0
- **projected_storage_bytes**: 0
- **build_time_s**: 0.0
- **mrr_at_10_proxy**: None
- **recall_at_5_proxy**: None
- **recall_at_10_proxy**: None
- **recall_at_20_proxy**: None
