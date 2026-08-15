# Dataset Contract: MSMARCO-XI

## Source
- **Repository**: `ai4bharat/MSMARCO-XI` on Hugging Face
- **Configs**: 14 Indic language configs (`as`, `bn`, `gu`, `hi`, `kn`, `ml`, `mr`, `ne`, `or`, `pa`, `sa`, `ta`, `te`, `ur`)
- **Splits**: `train`, `validation`

## Record Schema
Each record contains:
- `query_id`: string
- `query`: Indic-language query — **FORBIDDEN in payloads/embeddings**
- `Eng_Query`: English query — **FORBIDDEN**
- `Answer`: Indic answer — **FORBIDDEN**
- `Eng_Answer`: English answer — **FORBIDDEN**
- `query_type`: string — **FORBIDDEN**
- `passages`: dict with:
  - `English_passages`: list of English passage strings
  - `Translated_passages`: list of Indic-language passage strings
  - `is_selected`: list of relevance labels — **FORBIDDEN in payloads**

## CRITICAL: Leakage Boundary

These fields MUST NEVER enter:
- Production Qdrant point payloads
- Dense vector inputs
- Sparse vector inputs
- Reranking features

```python
FORBIDDEN_FIELDS = {"query", "Answer", "Eng_Query", "Eng_Answer", "query_type", "is_selected"}
```

They are for **offline evaluation only**.

## Passage Types

1. **English passages** (`passage_language = "en"`) — deduplicated globally across all 14 configs
2. **Translated passages** (`passage_language = config_language`) — language-scoped dedup

## Quality Filters

- Minimum 10 chars after NFC normalization and whitespace collapse
- Non-string passages are rejected and logged

## Scale (snapshot estimate)
- ~11.4M rows × 14 configs
- ~160M passage occurrences before dedup
- Expected unique passages after global dedup: TBD from pilot
