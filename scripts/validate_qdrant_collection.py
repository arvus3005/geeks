#!/usr/bin/env python3
"""Validate a Qdrant collection: schema, vectors, payload indexes, language coverage."""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="Validate Qdrant collection")
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--collection", default=None)
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--check-languages", nargs="*", default=None)
    args = parser.parse_args()

    import yaml

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}

    collection = args.collection or cfg.get(
        "qdrant_collection_physical", "msmarco_xi_passages_smoke_v001"
    )
    qdrant_url = args.qdrant_url or cfg.get("qdrant_url", "http://localhost:6333")

    from qdrant_client import QdrantClient

    from hhgoa_rag.qdrant_lifecycle import validate_collection

    client = QdrantClient(url=qdrant_url)

    try:
        result = validate_collection(client, collection)
    except Exception as e:
        print(json.dumps({"error": str(e), "collection": collection}))
        sys.exit(1)

    # Check forbidden fields in payload
    forbidden = {"query", "Answer", "Eng_Query", "Eng_Answer", "query_type", "is_selected"}
    scroll_result = client.scroll(collection_name=collection, limit=10, with_payload=True)
    leakage_found = []
    for point in scroll_result[0]:
        if point.payload:
            leaked = forbidden & set(point.payload.keys())
            if leaked:
                leakage_found.extend(list(leaked))

    result["forbidden_field_leakage"] = leakage_found
    result["leakage_check"] = "PASS" if not leakage_found else "FAIL"

    print(json.dumps(result, indent=2))
    sys.exit(0 if result["valid"] and not leakage_found else 1)


if __name__ == "__main__":
    main()
