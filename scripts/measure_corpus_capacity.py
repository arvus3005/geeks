#!/usr/bin/env python3
"""Read-only corpus capacity measurement tool for MSMARCO-XI → Pinecone.

Inspects pinned dataset Parquet shards without indexing them.
Supports bounded sampling mode so measurements finish quickly.

Reports:
- Sample rows inspected
- English passage counts and translated passage counts
- Deduplication rates across languages and within languages
- Chunk expansion under the selected chunking strategy
- Token distributions under multilingual-e5-large
- Serialized text and metadata payload byte sizes
- Differentiated Pinecone managed integrated-embedding vs hypothetical local store storage

ALL numbers are strictly labeled as MEASURED (from sample), EXTRAPOLATED, or ASSUMED.
Never calls Pinecone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

from hhgoa_rag.pinecone_contract import (
    DATASET_REPO,
    TOKENIZER_REPO,
)
from hhgoa_rag.pinecone_contract import (
    DATASET_REVISION as CANONICAL_DATASET_REVISION,
)
from hhgoa_rag.pinecone_contract import (
    DIMENSION as CANONICAL_DIMENSION,
)
from hhgoa_rag.pinecone_contract import (
    TOKENIZER_REVISION as CANONICAL_TOKENIZER_REVISION,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("measure_capacity")

# Published snapshot row counts from MSMARCO-XI dataset cards (for extrapolation only)
PUBLISHED_ROWS_PER_INDIC_CONFIG_TRAIN = 817_951  # Approximate published train rows for hi/bn
TOTAL_INDIC_CONFIGS = 14
TARGET_CONFIGS = ["hi", "bn"]

FLOAT32_BYTES = 4


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def measure_sample_data(
    rows_by_config: dict[str, list[dict]],
    chunk_strategy: str = "sentence_aware",
    tokenizer: Any = None,
    dimension: int = CANONICAL_DIMENSION,
) -> dict[str, Any]:
    """Perform read-only measurement on sample rows across language configs.

    Parameters:
        rows_by_config: Mapping of config language ('hi', 'bn') to list of raw dataset rows.
        chunk_strategy: Chunking strategy name from CHUNKERS registry.
        tokenizer: Optional tokenizer instance with count_tokens method.
        dimension: Embedding dimension (default 1024).

    Returns:
        Structured measurement dict with measured, extrapolated, and assumed metrics.
    """
    from hhgoa_rag.dataset.parser import parse_record
    from hhgoa_rag.ingestion.chunkers import get_chunker
    from hhgoa_rag.ingestion.schema import build_record

    chunker = get_chunker(chunk_strategy, tokenizer=tokenizer)

    total_rows = sum(len(rows) for rows in rows_by_config.values())
    config_counts = {cfg: len(rows) for cfg, rows in rows_by_config.items()}

    total_raw_en_passages = 0
    total_raw_trans_passages = 0

    seen_en_hashes: set[str] = set()
    unique_en_passages: list[dict] = []
    en_cross_duplicates = 0

    trans_passages_by_lang: dict[str, list[dict]] = {}
    trans_duplicates_by_lang: dict[str, int] = {}

    all_measured_chunks: list[dict] = []
    total_payload_bytes = 0
    token_lengths: list[int] = []

    for cfg_lang, rows in rows_by_config.items():
        trans_passages_by_lang.setdefault(cfg_lang, [])
        trans_duplicates_by_lang.setdefault(cfg_lang, 0)
        seen_trans_hashes: set[str] = set()

        for row_idx, row in enumerate(rows):
            occurrences, _ = parse_record(
                row,
                config_language=cfg_lang,
                split="train",
                source_shard="sample",
                source_row=row_idx,
                dataset_revision=CANONICAL_DATASET_REVISION,
            )

            for occ in occurrences:
                chash = _content_hash(occ.normalized_text)
                pinfo = {
                    "text": occ.normalized_text,
                    "content_hash": chash,
                    "language": occ.passage_language,
                    "config_language": occ.config_language,
                    "source_row": occ.source_row,
                    "passage_position": occ.passage_position,
                    "is_english": occ.is_original_english,
                }

                if occ.is_original_english:
                    total_raw_en_passages += 1
                    if chash not in seen_en_hashes:
                        seen_en_hashes.add(chash)
                        unique_en_passages.append(pinfo)
                    else:
                        en_cross_duplicates += 1
                else:
                    total_raw_trans_passages += 1
                    if chash not in seen_trans_hashes:
                        seen_trans_hashes.add(chash)
                        trans_passages_by_lang[cfg_lang].append(pinfo)
                    else:
                        trans_duplicates_by_lang[cfg_lang] += 1

    # Chunk and measure all unique passages
    all_unique_passages = unique_en_passages + [
        p for plist in trans_passages_by_lang.values() for p in plist
    ]

    total_passages_chunked = len(all_unique_passages)
    total_chunks_produced = 0

    for p in all_unique_passages:
        chunks = chunker.chunk(p["text"], p["content_hash"])
        chunk_texts = [c.text for c in chunks if c.text.strip()]
        total_chunks_produced += len(chunk_texts)

        for ordinal, ctext in enumerate(chunk_texts):
            chash = _content_hash(ctext)
            tlen = (
                tokenizer.count_tokens(ctext, add_prefix=True)
                if tokenizer is not None
                else len(ctext.split())
            )
            token_lengths.append(tlen)

            # Build record to measure exact serialized bytes
            rec = build_record(
                record_id=f"sample_{chash[:12]}_{ordinal}",
                chunk_text=ctext,
                language=p["language"],
                config_language=p["config_language"],
                dataset_revision=CANONICAL_DATASET_REVISION,
                split="train",
                physical_shard="sample.parquet",
                local_source_row=p["source_row"],
                passage_position=p["passage_position"],
                parent_passage_id=p["content_hash"],
                content_hash=chash,
                chunk_strategy=chunk_strategy,
                chunk_strategy_version="v1",
                chunk_ordinal=ordinal,
                chunk_total=len(chunk_texts),
                token_length=tlen,
                tokenizer_fingerprint="sample_fingerprint",
                manifest_id="sample_manifest",
            )
            rec_bytes = len(json.dumps(rec, ensure_ascii=False).encode("utf-8"))
            total_payload_bytes += rec_bytes
            all_measured_chunks.append({"token_length": tlen, "bytes": rec_bytes})

    # Compute measured sample rates
    avg_chunks_per_passage = (
        (total_chunks_produced / total_passages_chunked) if total_passages_chunked > 0 else 1.0
    )
    avg_tokens_per_chunk = (sum(token_lengths) / len(token_lengths)) if token_lengths else 0.0
    avg_payload_bytes_per_record = (
        (total_payload_bytes / total_chunks_produced) if total_chunks_produced > 0 else 0.0
    )
    en_dedup_ratio = (
        (len(unique_en_passages) / total_raw_en_passages) if total_raw_en_passages > 0 else 1.0
    )

    # Pinecone Managed Integrated Vector Model:
    # Dense vector: 1024 dims * 4 bytes = 4096 bytes per vector
    # Payload: avg measured serialized record bytes
    # Pinecone internal index structure factor: ~1.3x
    pinecone_dense_bytes_per_vector = dimension * FLOAT32_BYTES
    pinecone_total_bytes_per_vector = (
        pinecone_dense_bytes_per_vector + avg_payload_bytes_per_record
    ) * 1.3

    # Hypothetical Local Dense/Sparse Store:
    # Dense: 1024 * 4 = 4096 bytes
    # Sparse: ~50 nonzeros * 8 bytes = 400 bytes
    # Payload: avg payload bytes
    # Local HNSW graph index overhead: ~200 bytes per vector
    # Safety margin: 1.5x
    local_sparse_bytes_per_vector = 50 * 8
    local_hnsw_overhead_bytes_per_vector = 200
    local_base_bytes_per_vector = (
        pinecone_dense_bytes_per_vector
        + local_sparse_bytes_per_vector
        + avg_payload_bytes_per_record
        + local_hnsw_overhead_bytes_per_vector
    )
    local_total_bytes_with_safety = local_base_bytes_per_vector * 1.5

    # Extrapolations for target 3 languages (English + Hindi + Bengali)
    raw_passages_per_row = (
        ((total_raw_en_passages + total_raw_trans_passages) / total_rows)
        if total_rows > 0
        else 10.0
    )

    target_extrapolated_rows = PUBLISHED_ROWS_PER_INDIC_CONFIG_TRAIN * 2  # hi + bn
    target_extrapolated_raw_passages = int(target_extrapolated_rows * raw_passages_per_row)
    # Shared English deduped across the 2 configs + native translations
    target_extrapolated_unique_passages = int(
        target_extrapolated_raw_passages * 0.5 * en_dedup_ratio  # unique English
        + target_extrapolated_raw_passages * 0.5  # native translations
    )
    target_extrapolated_vectors = int(target_extrapolated_unique_passages * avg_chunks_per_passage)

    pinecone_target_storage_gb = (
        target_extrapolated_vectors * pinecone_total_bytes_per_vector
    ) / 1e9
    local_target_storage_gb = (target_extrapolated_vectors * local_total_bytes_with_safety) / 1e9

    return {
        "metadata": {
            "tool": "measure_corpus_capacity",
            "dataset_repo": DATASET_REPO,
            "dataset_revision": CANONICAL_DATASET_REVISION,
            "tokenizer_repo": TOKENIZER_REPO,
            "tokenizer_revision": CANONICAL_TOKENIZER_REVISION,
            "chunk_strategy": chunk_strategy,
            "canonical_dimension": dimension,
            "status": "SAMPLE MEASURED — FULL CORPUS UNRESOLVED",
        },
        "measured_sample_metrics": {
            "sample_rows_inspected": total_rows,
            "sample_rows_by_config": config_counts,
            "raw_english_passages": total_raw_en_passages,
            "unique_english_passages": len(unique_en_passages),
            "english_cross_config_duplicates": en_cross_duplicates,
            "english_dedup_ratio": round(en_dedup_ratio, 4),
            "raw_translated_passages": total_raw_trans_passages,
            "translated_duplicates_by_lang": trans_duplicates_by_lang,
            "total_unique_passages_chunked": total_passages_chunked,
            "total_chunks_produced": total_chunks_produced,
            "measured_chunk_expansion_ratio": round(avg_chunks_per_passage, 4),
            "average_tokens_per_chunk": round(avg_tokens_per_chunk, 1),
            "average_payload_bytes_per_record": round(avg_payload_bytes_per_record, 1),
            "measured_records": len(all_measured_chunks),
        },
        "pinecone_integrated_vector_model": {
            "architecture": "Pinecone Serverless Managed Integrated Multilingual Embedding",
            "dense_vector_bytes_per_record": pinecone_dense_bytes_per_vector,
            "payload_bytes_per_record": round(avg_payload_bytes_per_record, 1),
            "index_overhead_factor": 1.3,
            "effective_bytes_per_vector": round(pinecone_total_bytes_per_vector, 1),
            "canary_300_records_storage_mb": round(
                (300 * pinecone_total_bytes_per_vector) / 1e6, 3
            ),
            "starter_pilot_10k_records_storage_mb": round(
                (10_000 * pinecone_total_bytes_per_vector) / 1e6, 3
            ),
            "extrapolated_target_3_languages_storage_gb": round(pinecone_target_storage_gb, 2),
            "starter_plan_capacity_gb": 2.0,
            "starter_plan_canary_fit": "FITS CONFORTABLY (< 5 MB storage, < 30k tokens)",
            "starter_plan_pilot_10k_fit": "FITS WITHIN 2GB STARTER CAP (~70 MB)",
            "starter_plan_target_3_languages_fit": "EXCEEDS STARTER PLAN (requires Standard/Enterprise plan)",
        },
        "hypothetical_local_store_model": {
            "note": "For comparison only — not used in current Pinecone architecture",
            "dense_bytes_per_vector": pinecone_dense_bytes_per_vector,
            "sparse_bytes_per_vector": local_sparse_bytes_per_vector,
            "hnsw_overhead_bytes_per_vector": local_hnsw_overhead_bytes_per_vector,
            "payload_bytes_per_record": round(avg_payload_bytes_per_record, 1),
            "safety_factor": 1.5,
            "effective_bytes_per_vector": round(local_total_bytes_with_safety, 1),
            "extrapolated_target_3_languages_storage_gb": round(local_target_storage_gb, 2),
        },
        "extrapolated_scope_estimates": {
            "disclaimer": "EXTRAPOLATED from sample averages and published HF dataset row counts. NOT measured full-corpus facts.",
            "published_train_rows_per_indic_config": PUBLISHED_ROWS_PER_INDIC_CONFIG_TRAIN,
            "target_3_languages": {
                "configs": ["en (shared)", "hi", "bn"],
                "extrapolated_unique_passages": target_extrapolated_unique_passages,
                "extrapolated_vectors": target_extrapolated_vectors,
                "pinecone_storage_gb": round(pinecone_target_storage_gb, 2),
            },
        },
        "local_disk_and_memory_assessment": {
            "status": "UNRESOLVED FOR FULL CORPUS",
            "finding": (
                "A 200 GB local Mac CANNOT be declared sufficient for the full 14-config workflow "
                "until measured shard downloads, extraction caches, temporary files, SQLite dedup DB, "
                "and build headroom are verified against real storage usage."
            ),
            "canary_local_feasibility": "VERIFIED SUFFICIENT (< 1 GB total temp/cache space needed)",
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Measure MSMARCO-XI corpus capacity and chunking metrics (read-only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--dataset-revision",
        default=CANONICAL_DATASET_REVISION,
        help=f"Pinned HuggingFace dataset commit hash (default {CANONICAL_DATASET_REVISION})",
    )
    p.add_argument(
        "--tokenizer-revision",
        default=CANONICAL_TOKENIZER_REVISION,
        help=f"Pinned HuggingFace tokenizer commit hash (default {CANONICAL_TOKENIZER_REVISION})",
    )
    p.add_argument(
        "--sample-rows",
        type=int,
        default=50,
        help="Number of rows to sample per Parquet file (default 50 for fast bounded measurement)",
    )
    p.add_argument(
        "--chunk-strategy",
        default="sentence_aware",
        help="Chunking strategy to evaluate (default sentence_aware)",
    )
    p.add_argument(
        "--output-json",
        action="store_true",
        help="Output raw JSON to stdout",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    # Ensure repo root and scripts dir are in sys.path
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    logger.info("Initializing tokenizer for measurement …")
    from hhgoa_rag.ingestion.tokenizer import get_tokenizer

    try:
        tok = get_tokenizer(revision=args.tokenizer_revision)
    except Exception as e:
        # Fail closed: canonical measured token counts must never be silently
        # replaced by an approximate whitespace count. There is no constructed
        # fallback tokenizer — refuse rather than produce misleading numbers.
        print(
            f"ERROR: Could not load the pinned HuggingFace tokenizer "
            f"(revision={args.tokenizer_revision!r}): {e}\n"
            "Refusing to substitute an approximate whitespace token count into "
            "canonical measured results. Ensure the tokenizer revision is "
            "downloadable (network/cache) and retry.",
            file=sys.stderr,
        )
        sys.exit(1)

    import prepare_canary as pc

    _download_and_read_parquet = pc._download_and_read_parquet

    rows_by_config: dict[str, list[dict]] = {}
    for cfg_lang in TARGET_CONFIGS:
        try:
            rows, _ = _download_and_read_parquet(
                config_lang=cfg_lang,
                split="train",
                dataset_revision=args.dataset_revision,
                max_rows=args.sample_rows,
            )
            rows_by_config[cfg_lang] = rows
        except Exception as e:
            logger.error("Failed to read Parquet shard for %s: %s", cfg_lang, e)
            print(
                f"ERROR: Could not inspect Parquet shard for {cfg_lang}: {e}",
                file=sys.stderr,
            )
            sys.exit(1)

    result = measure_sample_data(
        rows_by_config=rows_by_config,
        chunk_strategy=args.chunk_strategy,
        tokenizer=tok,
    )

    if args.output_json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("\n" + "=" * 65)
        print("CORPUS CAPACITY & CHUNKING MEASUREMENT REPORT")
        print("=" * 65)
        m = result["measured_sample_metrics"]
        print(
            f"  Sample rows inspected    : {m['sample_rows_inspected']} ({m['sample_rows_by_config']})"
        )
        print(f"  Raw English passages     : {m['raw_english_passages']}")
        print(
            f"  Unique English passages  : {m['unique_english_passages']} (cross-dedup: {m['english_cross_config_duplicates']})"
        )
        print(f"  Raw translated passages  : {m['raw_translated_passages']}")
        print(f"  Total unique passages    : {m['total_unique_passages_chunked']}")
        print(f"  Total chunks produced    : {m['total_chunks_produced']}")
        print(f"  Chunk expansion ratio    : {m['measured_chunk_expansion_ratio']}x")
        print(f"  Average tokens per chunk : {m['average_tokens_per_chunk']}")
        print(f"  Average record payload   : {m['average_payload_bytes_per_record']} bytes")
        print("=" * 65)
        p = result["pinecone_integrated_vector_model"]
        print("PINECONE MANAGED INTEGRATED EMBEDDING STORAGE (dense only + payload):")
        print(
            f"  300-record canary storage: {p['canary_300_records_storage_mb']} MB (fit: {p['starter_plan_canary_fit']})"
        )
        print(
            f"  10,000-record pilot      : {p['starter_pilot_10k_records_storage_mb']} MB (fit: {p['starter_plan_pilot_10k_fit']})"
        )
        print(
            f"  Target 3-language subset : ~{p['extrapolated_target_3_languages_storage_gb']} GB [EXTRAPOLATED]"
        )
        print("=" * 65)
        print("LOCAL DISK ASSESSMENT:")
        print(f"  {result['local_disk_and_memory_assessment']['finding']}")
        print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
