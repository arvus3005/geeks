"""Embedder module for the HH Goa 2026 Task 2 eval harness
(rag-local-eval-loop, see docs/wiring-in-the-eval-loop.pdf and
TARGET_INTERFACE.md in that suite's own repo).

Thin wrapper around this project's real production embedder
(hhgoa_rag.retrieval.local_embedder — int8 ONNX + native SentencePiece,
intfloat/multilingual-e5-small) so the eval suite scores the exact same
embedding path the live API uses, not a separate reimplementation.

Call-pattern mapping (confirmed by reading the eval suite's own source,
not assumed from its docs): `embed()` is called in batches to build the
suite's own throwaway evaluation corpus (eval/index_build.py), i.e. the
PASSAGE side; `embed_one()` is called once per query at retrieval time
(eval/pipeline.py), i.e. the QUERY side. e5 models are asymmetrically
instruction-tuned ("query: " vs "passage: " prefixes) — mapping the two
eval-suite calls onto this project's own embed_passages_batch /
embed_query (rather than using one function for both) is what keeps this
faithful to how the live system actually embeds text, not just
dimensionally compatible.
"""

from __future__ import annotations

import numpy as np

from hhgoa_rag.retrieval.local_embedder import EMBED_DIM, _lazy_load, embed_passages_batch, embed_query


def get_model():
    """Called once by the eval suite; only the side effect (loading the
    ONNX session + SentencePiece tokenizer once, per CLAUDE.md) matters."""
    _lazy_load()
    return None


def embed_one(text: str) -> np.ndarray:
    return np.asarray(embed_query(text), dtype=np.float32)


def embed(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, EMBED_DIM), dtype=np.float32)
    return np.asarray(embed_passages_batch(texts), dtype=np.float32)
