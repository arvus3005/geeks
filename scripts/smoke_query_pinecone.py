#!/usr/bin/env python3
"""Run a test query against the Pinecone smoke namespace and display results."""

import argparse
import json
import os
import sys


def main() -> None:
    p = argparse.ArgumentParser(description="Query Pinecone smoke namespace")
    p.add_argument("--pinecone-api-key", default=None)
    p.add_argument("--pinecone-index", default=os.environ.get("PINECONE_INDEX", "msmarco-xi"))
    p.add_argument("--query", default="What is the capital of India?")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--language-filter", default=None, help="Comma-separated language codes, e.g. en,hi,bn")
    p.add_argument("--namespace", default="smoke")
    p.add_argument("--output-json", action="store_true")
    args = p.parse_args()

    api_key = args.pinecone_api_key or os.environ.get("PINECONE_API_KEY")
    if not api_key:
        print("ERROR: PINECONE_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    from pinecone import Pinecone

    from hhgoa_rag.pinecone_store import PineconeStore

    pc = Pinecone(api_key=api_key)
    store = PineconeStore(pc.Index(args.pinecone_index), embed_model="multilingual-e5-large")

    filter_dict = None
    if args.language_filter:
        langs = [lang.strip() for lang in args.language_filter.split(",")]
        filter_dict = {"language": {"$in": langs}}

    hits = store.search(
        query_text=args.query,
        top_k=args.top_k,
        namespace=args.namespace,
        filter=filter_dict,
    )

    result = {
        "query": args.query,
        "namespace": args.namespace,
        "top_k": args.top_k,
        "hits": [{"id": h.id, "score": h.score, "language": h.language, "text": h.text[:150]} for h in hits],
    }
    if args.output_json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Query: {args.query!r}  ns={args.namespace}  hits={len(hits)}")
        for h in hits:
            print(f"  [{h.score:.3f}] {h.language} — {h.text[:100]}")


if __name__ == "__main__":
    main()
