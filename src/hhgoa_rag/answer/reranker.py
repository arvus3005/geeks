"""Self-hosted cross-encoder relevance scorer — jinaai/jina-reranker-v2-base-
multilingual, int8 ONNX (~280MB on disk, ~0.85GB resident once loaded).

Why this exists, and why it's not just more embedding-similarity tuning
--------------------------------------------------------------------------
extract_answer's earlier guardrail (query/passage embedding cosine
similarity, calibrated on real multilingual data) could not reliably tell
"this passage genuinely answers the question" from "this passage is about
the same general topic but doesn't answer it" -- both score similarly
high, because bi-encoder similarity only compares two SEPARATELY-computed
vectors, never lets the query and passage tokens actually interact.
Confirmed via a competing team's own real ablation (see
reference/shruti-main/docs/BUILD_LOG.md): top-1 cosine similarity tops out
at AUC=0.713 for exactly this discrimination, and pushing the threshold
harder trades real answerable-query recall for it, not free lunch.

A cross-encoder is architecturally the right tool: it runs the query and
passage through the model TOGETHER (concatenated, with cross-attention),
so it can actually judge "does this text answer that question" rather
than "are these two topics similar". Measured on this project's own known
problem cases (2026-08-22): relevant pair scored 2.46, a topically-
adjacent-but-wrong pair scored -0.58 (negative), a genuinely unrelated
pair scored -3.62 -- a clean, wide separation cosine similarity never
produced (its own equivalent scores for the same pairs sat within ~0.05
of each other). ~5-6ms per (query, passage) score on CPU.

Same base architecture (XLM-RoBERTa) as this project's own embedder
(local_embedder.py uses multilingual-e5-small, also XLM-R-based), so
real multilingual coverage across all 6 indexed languages is not a
new, unverified assumption -- it's the same tokenizer family already
trusted for retrieval.

Loaded once at process startup, never per-request, per CLAUDE.md.
"""

from __future__ import annotations

import logging
import os
import threading

logger = logging.getLogger(__name__)

MODEL_REPO = "jinaai/jina-reranker-v2-base-multilingual"
_ONNX_FILENAME = "onnx/model_int8.onnx"
# config.json reports max_position_embeddings=1026; RoBERTa-family models
# reserve the first couple of positions, so 1024 is the real usable
# sequence length. Real production passages can run longer than this
# session's short calibration samples -- an unbounded encode() crashed
# with "indices element out of data bounds" on a real production passage
# the first time this was tested end-to-end, not just in isolation.
_MAX_SEQ_LEN = 1024

# Character-level cap applied BEFORE tokenization, not just the token-level
# truncation above. Found via a real measured tail-latency case: a Tamil
# query scored 450.9ms backend (vs P50=57ms) on a real production
# benchmark -- longer text costs more to tokenize AND to run through the
# model, and some scripts (Tamil among them) produce more subword pieces
# per character than Latin text, so a fixed token budget alone doesn't
# bound wall-clock the way a character cap does. A competing team
# (reference/hhgoa-task2-main/docs/BUILD_LOG.md) measured the same fix
# directly: capping encoder input at 256 chars cut their P99 by ~2.4x for
# ~9ms of P50 cost and negligible quality loss (1/60 answers changed).
# 500 is looser than their 256 since MSMARCO passages carry more real
# content per candidate than a single sentence does.
_MAX_CHARS = 500

_session = None
_tokenizer = None
_lock = threading.Lock()


def _lazy_load() -> None:
    global _session, _tokenizer
    if _session is not None:
        return
    with _lock:
        if _session is not None:
            return
        _build()


def _build() -> None:
    global _session, _tokenizer
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    logger.info("Loading reranker model (int8 ONNX, %s)…", MODEL_REPO)

    onnx_path = hf_hub_download(repo_id=MODEL_REPO, filename=_ONNX_FILENAME)
    tokenizer_path = hf_hub_download(repo_id=MODEL_REPO, filename="tokenizer.json")

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = int(os.environ.get("RERANKER_THREADS", "1"))
    _session = ort.InferenceSession(
        onnx_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
    )
    _tokenizer = Tokenizer.from_file(tokenizer_path)
    _tokenizer.enable_truncation(max_length=_MAX_SEQ_LEN)
    logger.info("Reranker model ready.")


def score(query: str, passage: str) -> float:
    """Higher = more relevant. Raw cross-encoder logit, not a probability --
    thresholds are calibrated against this raw scale (see extractive.py),
    not [0,1]. Positive roughly means "this passage answers the query";
    very negative means "this passage is not about the query at all"."""
    import numpy as np

    _lazy_load()
    assert _session is not None and _tokenizer is not None

    encoding = _tokenizer.encode(query[:_MAX_CHARS], passage[:_MAX_CHARS])
    input_ids = np.array([encoding.ids], dtype=np.int64)
    attention_mask = np.array([encoding.attention_mask], dtype=np.int64)
    (logits,) = _session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
    return float(logits[0][0])
