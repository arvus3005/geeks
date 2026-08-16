#!/usr/bin/env python3
"""Estimate storage and compute requirements for MSMARCO-XI ingestion.

Canonical vector dimension is imported directly from hhgoa_rag.pinecone_contract
(DIMENSION = 1024 for multilingual-e5-large).

Differentiates:
  1. Full MSMARCO-XI corpus (all 14 Indic configs + English)
  2. Target 3-language subset (English + Hindi + Bengali)
  3. Bounded Starter pilot (10,000 records)
  4. Canary baseline (300 records)
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from hhgoa_rag.pinecone_contract import DIMENSION as CANONICAL_DIMENSION

# ── Empirical/Snapshot Constants & Assumptions ───────────────────────────────
TOTAL_SNAPSHOT_ROWS = 11_451_314
TOTAL_CONFIGS = 14
TARGET_CONFIGS = 3  # English (shared), Hindi, Bengali
AVG_PASSAGES_PER_ROW_ASSUMPTION = 10  # 5 English + 5 translated per source row
FLOAT32_BYTES = 4
PAYLOAD_BYTES_ESTIMATE = 500  # text chunk + metadata
HNSW_INDEX_OVERHEAD_BYTES = 200  # graph index overhead per vector
SPARSE_NONZERO_ASSUMPTION = 50
SPARSE_BYTES_PER_NONZERO = 8  # 4 bytes int32 idx + 4 bytes float32 val
SAFETY_FACTOR = 1.5


def calculate_estimates(
    dimension: int = CANONICAL_DIMENSION,
    snapshot_rows: int = TOTAL_SNAPSHOT_ROWS,
    total_configs: int = TOTAL_CONFIGS,
    target_configs: int = TARGET_CONFIGS,
    avg_passages_per_row: int = AVG_PASSAGES_PER_ROW_ASSUMPTION,
    safety_factor: float = SAFETY_FACTOR,
) -> dict[str, Any]:
    """Compute structured storage and compute estimates for multiple scope targets."""
    # Dedup formulas based on dataset structure
    # Full corpus (all 14 configs)
    full_unique_eng = (snapshot_rows * avg_passages_per_row // 2) // total_configs
    full_unique_trans = snapshot_rows * avg_passages_per_row // 2
    full_total_passages = full_unique_eng + full_unique_trans

    # Target subset (English + Hindi + Bengali = 3 configs)
    # Rows for 2 Indic configs (Hindi + Bengali) out of 14:
    target_rows = snapshot_rows * target_configs // total_configs
    target_unique_eng = (target_rows * avg_passages_per_row // 2) // target_configs
    target_unique_trans = target_rows * avg_passages_per_row // 2
    target_total_passages = target_unique_eng + target_unique_trans

    # Bounded pilot (Starter max)
    pilot_passages = 10_000

    # Canary (exact)
    canary_passages = 300

    def _scope_storage(passages: int) -> dict[str, Any]:
        dense_b = passages * dimension * FLOAT32_BYTES
        sparse_b = passages * SPARSE_NONZERO_ASSUMPTION * SPARSE_BYTES_PER_NONZERO
        payload_b = passages * PAYLOAD_BYTES_ESTIMATE
        hnsw_b = passages * HNSW_INDEX_OVERHEAD_BYTES
        total_b = dense_b + sparse_b + payload_b + hnsw_b
        return {
            "passages": passages,
            "dense_vectors_gb": round(dense_b / 1e9, 3),
            "sparse_vectors_gb": round(sparse_b / 1e9, 3),
            "payload_gb": round(payload_b / 1e9, 3),
            "hnsw_index_overhead_gb": round(hnsw_b / 1e9, 3),
            "total_base_gb": round(total_b / 1e9, 3),
            "with_safety_margin_gb": round(total_b * safety_factor / 1e9, 3),
        }

    def _scope_compute(passages: int) -> dict[str, Any]:
        return {
            "estimated_hours_cpu_50_rps": round(passages / 50 / 3600, 2),
            "estimated_hours_gpu_500_rps": round(passages / 500 / 3600, 2),
        }

    return {
        "metadata": {
            "canonical_dimension": dimension,
            "dimension_source": "src/hhgoa_rag/pinecone_contract.py (multilingual-e5-large)",
            "safety_factor": safety_factor,
            "status": "ESTIMATE ONLY — separates measured inputs from assumptions",
        },
        "assumptions_vs_measured": {
            "measured_inputs": {
                "vector_dimension": dimension,
                "canary_exact_records": canary_passages,
                "pinecone_starter_storage_limit_gb": 2.0,
                "pinecone_starter_monthly_token_limit": 5_000_000,
            },
            "unmeasured_assumptions": {
                "snapshot_rows": snapshot_rows,
                "total_configs": total_configs,
                "target_configs": target_configs,
                "avg_passages_per_row": avg_passages_per_row,
                "sparse_nonzeros_per_vector": SPARSE_NONZERO_ASSUMPTION,
                "payload_bytes_per_passage": PAYLOAD_BYTES_ESTIMATE,
                "hnsw_bytes_per_vector": HNSW_INDEX_OVERHEAD_BYTES,
                "note_on_local_disk": (
                    "A 200 GB Mac cannot be declared sufficient for full or 3-language "
                    "indexing until actual English/Hindi/Bengali record and chunk counts "
                    "are measured from Parquet shards."
                ),
            },
        },
        "scopes": {
            "canary_300_records": {
                "description": "Exact 300-record canary (100 en + 100 hi + 100 bn)",
                "storage": _scope_storage(canary_passages),
                "compute": _scope_compute(canary_passages),
                "starter_plan_fit": "FITS CONFORTABLY (< 5 MB storage, < 30k tokens)",
            },
            "bounded_pilot_10k_records": {
                "description": "Pinecone Starter 10,000-record operational pilot ceiling",
                "storage": _scope_storage(pilot_passages),
                "compute": _scope_compute(pilot_passages),
                "starter_plan_fit": "FITS WITHIN STARTER 2GB CAP (~0.07 GB)",
            },
            "target_3_languages_en_hi_bn": {
                "description": "English, Hindi and Bengali subsets (3 / 14 configs)",
                "estimated_unique_passages": target_total_passages,
                "storage": _scope_storage(target_total_passages),
                "compute": _scope_compute(target_total_passages),
                "starter_plan_fit": "EXCEEDS STARTER PLAN — requires Pinecone Standard/Enterprise or dedicated local vector store",
            },
            "full_corpus_14_languages": {
                "description": "Entire MSMARCO-XI dataset (14 Indic configs + English)",
                "estimated_unique_passages": full_total_passages,
                "storage": _scope_storage(full_total_passages),
                "compute": _scope_compute(full_total_passages),
                "starter_plan_fit": "EXCEEDS STARTER PLAN — requires Pinecone Standard/Enterprise or dedicated local vector store",
            },
        },
        "infrastructure_profiles": [
            {
                "name": "canary_and_pilot",
                "scope": "Canary (300) to Pilot (10,000)",
                "ram_gb": 8,
                "disk_gb": 20,
                "suitability": "Local developer Mac / laptop is fully sufficient",
            },
            {
                "name": "target_3_languages",
                "scope": "English + Hindi + Bengali (~12M passages)",
                "ram_gb": 32,
                "disk_gb": 200,
                "suitability": (
                    "200 GB disk may be tight during shard extraction and build headroom. "
                    "Cannot declare 200 GB sufficient until exact chunk counts are measured."
                ),
            },
            {
                "name": "full_corpus_all_languages",
                "scope": "All 14 Indic configs (~61M passages)",
                "ram_gb": 128,
                "disk_gb": 1000,
                "gpu": "A10 / A100 recommended for embedding throughput",
                "suitability": "Dedicated indexing server or cloud cluster required",
            },
        ],
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Estimate storage and compute requirements for MSMARCO-XI ingestion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--dimension",
        type=int,
        default=CANONICAL_DIMENSION,
        help=f"Dense vector dimension (default {CANONICAL_DIMENSION} from contract)",
    )
    p.add_argument(
        "--snapshot-rows",
        type=int,
        default=TOTAL_SNAPSHOT_ROWS,
        help=f"Source snapshot rows (default {TOTAL_SNAPSHOT_ROWS})",
    )
    p.add_argument(
        "--safety-factor",
        type=float,
        default=SAFETY_FACTOR,
        help=f"Storage safety margin factor (default {SAFETY_FACTOR})",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    report = calculate_estimates(
        dimension=args.dimension,
        snapshot_rows=args.snapshot_rows,
        safety_factor=args.safety_factor,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
