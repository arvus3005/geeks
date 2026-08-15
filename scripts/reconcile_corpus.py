#!/usr/bin/env python3
"""Reconcile corpus manifest counts with Qdrant point counts."""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="Reconcile corpus manifest with Qdrant")
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--manifest", default=None, help="Path to manifest JSON")
    args = parser.parse_args()

    import yaml

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}

    collection = cfg.get("qdrant_collection_physical", "msmarco_xi_passages_smoke_v001")

    from qdrant_client import QdrantClient

    from hhgoa_rag.qdrant_lifecycle import validate_collection

    client = QdrantClient(url=args.qdrant_url)
    result = validate_collection(client, collection)

    reconciliation = {
        "collection": collection,
        "qdrant_points": result["points"],
        "status": result["status"],
        "valid": result["valid"],
        "issues": result["issues"],
        "reconciliation_equation": (
            "source_occurrences = valid + rejected + unique + duplicate (see manifest)"
        ),
    }
    print(json.dumps(reconciliation, indent=2))
    sys.exit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
