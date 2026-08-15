#!/usr/bin/env python3
"""Estimate storage and compute requirements for full MSMARCO-XI ingestion."""

import json


def main():
    # Based on snapshot: 11,451,314 rows, 55.6 GB source, 14 Indic configs
    SNAPSHOT_ROWS = 11_451_314
    CONFIGS = 14
    AVG_PASSAGES_PER_ROW = 10  # both English and translated

    # Dedup estimates
    EST_UNIQUE_ENG = SNAPSHOT_ROWS * AVG_PASSAGES_PER_ROW // 2 // CONFIGS
    EST_UNIQUE_TRANS = SNAPSHOT_ROWS * AVG_PASSAGES_PER_ROW // 2
    EST_TOTAL_UNIQUE_PASSAGES = EST_UNIQUE_ENG + EST_UNIQUE_TRANS

    DENSE_DIM = 384
    FLOAT32_BYTES = 4
    dense_bytes = EST_TOTAL_UNIQUE_PASSAGES * DENSE_DIM * FLOAT32_BYTES

    AVG_SPARSE_NZ = 50
    sparse_bytes = EST_TOTAL_UNIQUE_PASSAGES * AVG_SPARSE_NZ * 8
    payload_bytes = EST_TOTAL_UNIQUE_PASSAGES * 500
    hnsw_bytes = EST_TOTAL_UNIQUE_PASSAGES * 200

    total_bytes = dense_bytes + sparse_bytes + payload_bytes + hnsw_bytes
    safety_factor = 1.5

    report = {
        "warning": "ESTIMATE ONLY — based on snapshot metadata, not measured pilot data",
        "snapshot_rows": SNAPSHOT_ROWS,
        "estimated_unique_passages": EST_TOTAL_UNIQUE_PASSAGES,
        "storage": {
            "dense_vectors_gb": round(dense_bytes / 1e9, 1),
            "sparse_vectors_gb": round(sparse_bytes / 1e9, 1),
            "payload_gb": round(payload_bytes / 1e9, 1),
            "hnsw_gb": round(hnsw_bytes / 1e9, 1),
            "total_estimated_gb": round(total_bytes / 1e9, 1),
            "with_safety_margin_gb": round(total_bytes * safety_factor / 1e9, 1),
        },
        "compute": {
            "embedding_throughput_passages_per_sec_cpu": 50,
            "estimated_hours_cpu": round(EST_TOTAL_UNIQUE_PASSAGES / 50 / 3600, 1),
            "embedding_throughput_passages_per_sec_gpu": 500,
            "estimated_hours_gpu": round(EST_TOTAL_UNIQUE_PASSAGES / 500 / 3600, 1),
        },
        "infrastructure_profiles": [
            {
                "name": "single_node_pilot",
                "ram_gb": 32,
                "fast_disk_gb": 200,
                "note": "For pilot subset only (~1M passages)",
            },
            {
                "name": "minimum_full_index",
                "ram_gb": 64,
                "fast_disk_gb": 500,
                "gpu": "optional T4 or better",
                "note": "Minimum viable for full corpus",
            },
            {
                "name": "recommended_judging",
                "ram_gb": 128,
                "fast_disk_gb": 1000,
                "gpu": "A10 or better",
                "note": "Comfortable headroom for full corpus + snapshots",
            },
        ],
        "CAUTION": "DO NOT provision infrastructure until cost is approved",
    }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
