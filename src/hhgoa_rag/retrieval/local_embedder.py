"""Local embedding for both indexing (passages) and querying, using
intfloat/multilingual-e5-small via ONNX Runtime + native SentencePiece.

Why e5-small instead of e5-large, and why ONNX + SentencePiece instead of
torch + transformers
---------------------------------------------------------------------------
Both indexing and querying were originally done through Pinecone's
server-side integrated embedding (multilingual-e5-large). That hit the
account's monthly embedding-token quota (429 RESOURCE_EXHAUSTED) — see git
history — so embedding moved local. A torch + transformers + e5-large stack
was tried first and measured at ~1.5-2GB steady-state RSS regardless of
quantization (int8 weights, ONNX, arena tuning — all tried, all landed in
that range), which doesn't fit a 512MB free-tier container.

Two things, tested empirically, get this down to ~407.5MB total:

1. **ONNX Runtime instead of torch/transformers.** onnxruntime's own import
   footprint is ~55MB vs. torch's ~195MB, and it needs no autograd/nn.Module
   graph machinery.
2. **Raw sentencepiece.SentencePieceProcessor instead of HF's
   tokenizers.Tokenizer(tokenizer.json).** Both wrap the *same* 250,002-token
   XLM-RoBERTa vocabulary, but the JSON-based `tokenizers` library builds a
   generic BPE merge-rule structure that measured ~440MB resident; the native
   SentencePiece binary format (its own purpose-built trie) measured
   ~121.7MB for the identical vocabulary — a ~3.6x difference for the exact
   same data.
3. **e5-small instead of e5-large.** Even with both fixes above, e5-large's
   ONNX session alone still costs ~1.46GB (its own size, not tokenizer or
   runtime overhead — confirmed by holding the tokenizer fixed and swapping
   only the model). e5-small's quantized ONNX session costs ~285MB instead.

Combined: onnxruntime (55MB) + SentencePiece (121.7MB) + e5-small int8 ONNX
(~230MB) = 407.5MB measured steady-state, with real headroom under 512MB.

This means passages must be embedded with e5-small too — it is NOT
dimension- or space-compatible with the original e5-large vectors. See
scripts/reindex_e5small.py for the one-time passage re-embedding migration.

Loaded once at process startup, never per-request, per CLAUDE.md.
"""

from __future__ import annotations

import logging
import os
import threading

logger = logging.getLogger(__name__)

MODEL_REPO = "intfloat/multilingual-e5-small"
ONNX_REPO = "Xenova/multilingual-e5-small"
ONNX_FILE = "onnx/model_int8.onnx"
EMBED_DIM = 384
_CACHE_DIR = ".cache/huggingface"
_XLMR_BOS, _XLMR_EOS = 0, 2  # <s>, </s> — XLM-RoBERTa SentencePiece special tokens
# XLM-RoBERTa's "fairseq offset": ids 0-3 are reserved for <s>/<pad>/</s>/<unk>
# ahead of the raw SentencePiece vocabulary, so every piece id from
# SentencePieceProcessor.encode() must be shifted by +1 to land on the
# correct row of the model's embedding table. Verified against
# transformers.AutoTokenizer's reference output — omitting this offset
# still produces syntactically valid ids (just off by one row each), so it
# fails silently: no error, no crash, just semantically scrambled vectors.
_FAIRSEQ_OFFSET = 1

_session = None
_sp = None
_load_lock = threading.Lock()


def _lazy_load() -> None:
    global _session, _sp
    if _session is not None:
        return

    # Double-checked locking: the bulk reindex script calls embed_passages_batch
    # from a thread pool, and without this, concurrent first-callers would each
    # redundantly download and construct their own ONNX session/tokenizer.
    with _load_lock:
        if _session is not None:
            return
        _build()


def _build() -> None:
    global _session, _sp

    import onnxruntime as ort
    import sentencepiece as spm
    from huggingface_hub import hf_hub_download

    logger.info("Loading local embedder: %s (ONNX int8) + native SentencePiece", MODEL_REPO)

    sp_model_path = hf_hub_download(
        MODEL_REPO, "sentencepiece.bpe.model", cache_dir=_CACHE_DIR
    )
    _sp = spm.SentencePieceProcessor(model_file=sp_model_path)

    onnx_path = hf_hub_download(ONNX_REPO, ONNX_FILE, cache_dir=_CACHE_DIR)
    opts = ort.SessionOptions()
    # Small, memory-constrained deployment target: one thread is both more
    # predictable at the tail and avoids ORT spawning a thread per core.
    # Offline bulk-indexing scripts (not the serving path) can override this
    # via HHGOA_ONNX_INTRA_THREADS to use all available cores instead.
    threads = int(os.environ.get("HHGOA_ONNX_INTRA_THREADS", "1"))
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    _session = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])

    logger.info("Local embedder ready (dim=%d)", EMBED_DIM)


def _embed(text: str) -> list[float]:
    import numpy as np

    _lazy_load()
    assert _session is not None and _sp is not None

    ids = [_XLMR_BOS, *(p + _FAIRSEQ_OFFSET for p in _sp.encode(text)), _XLMR_EOS]
    input_ids = np.array([ids], dtype=np.int64)
    attention_mask = np.ones_like(input_ids)
    feed = {"input_ids": input_ids, "attention_mask": attention_mask}
    input_names = {i.name for i in _session.get_inputs()}
    if "token_type_ids" in input_names:
        feed["token_type_ids"] = np.zeros_like(input_ids)

    (last_hidden,) = _session.run(None, feed)
    mask = attention_mask[:, :, None].astype(np.float32)
    pooled = (last_hidden * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
    norm = np.linalg.norm(pooled, axis=1, keepdims=True)
    normalized = pooled / np.clip(norm, 1e-12, None)

    vec = normalized[0].tolist()
    if len(vec) != EMBED_DIM:
        raise ValueError(f"Expected {EMBED_DIM}-dim embedding, got {len(vec)}")
    return vec


def embed_query(text: str) -> list[float]:
    """E5 models require the 'query: ' prefix at query time (distinct from
    'passage: ' at indexing time) — mixing these up silently degrades
    retrieval quality without raising an error.
    """
    return _embed(f"query: {text}")


def embed_passage(text: str) -> list[float]:
    return _embed(f"passage: {text}")


def embed_passages_batch(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Batch passage embedding for bulk (re)indexing — padded batches are far
    faster than one ONNX session.run() call per passage at 57k-passage scale.
    """
    import numpy as np

    _lazy_load()
    assert _session is not None and _sp is not None
    input_names = {i.name for i in _session.get_inputs()}

    max_position_tokens = 512 - 2  # reserve room for BOS/EOS within the model's 512-position limit

    out: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        encoded = [
            [
                _XLMR_BOS,
                *(
                    p + _FAIRSEQ_OFFSET
                    for p in _sp.encode(f"passage: {t}")[:max_position_tokens]
                ),
                _XLMR_EOS,
            ]
            for t in chunk
        ]
        max_len = max(len(e) for e in encoded)

        input_ids = np.zeros((len(chunk), max_len), dtype=np.int64)
        attention_mask = np.zeros((len(chunk), max_len), dtype=np.int64)
        for i, e in enumerate(encoded):
            input_ids[i, : len(e)] = e
            attention_mask[i, : len(e)] = 1

        feed = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in input_names:
            feed["token_type_ids"] = np.zeros_like(input_ids)

        (last_hidden,) = _session.run(None, feed)
        mask = attention_mask[:, :, None].astype(np.float32)
        pooled = (last_hidden * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
        norm = np.linalg.norm(pooled, axis=1, keepdims=True)
        normalized = pooled / np.clip(norm, 1e-12, None)
        out.extend(normalized.tolist())

    return out
