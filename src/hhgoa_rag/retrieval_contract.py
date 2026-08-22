"""Canonical, backend-neutral constants shared across ingestion, retrieval,
and answer extraction.

Was named pinecone_contract.py through 2026-08-21 (had grown into a much
larger Pinecone-specific index contract — dimension, cloud/region, rerank
params, batch limits — none of which apply now that Pinecone is retired
from serving, see README's indexing-status section). Renamed and trimmed
2026-08-22 to just what's still genuinely shared, rather than leaving
unused Pinecone-shaped fields sitting in a file with "pinecone" in its
name.
"""

from __future__ import annotations

from typing import Final

# The field name passage text is stored/read under across the local hybrid
# store's passage payloads (src/hhgoa_rag/retrieval/sharded_local_hybrid_store.py
# sets this on every hit's metadata dict) and what extract_answer /
# verify_grounding read.
TEXT_FIELD: Final[str] = "chunk_text"

DATASET_REPO: Final[str] = "ai4bharat/MSMARCO-XI"
