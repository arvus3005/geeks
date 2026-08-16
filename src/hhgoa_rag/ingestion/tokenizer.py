"""Real multilingual-E5 tokenizer for exact token accounting.

Loads the tokenizer for `intfloat/multilingual-e5-large` from HuggingFace.
No model weights, no PyTorch — tokenizer only.

Usage
-----
    tok = get_tokenizer()                    # cached singleton
    n = tok.count_tokens("query: hello")    # includes BOS/EOS special tokens
    fp = tok.fingerprint                     # reproducible revision string

Model input limit: 512 tokens (including special tokens) per multilingual-e5-large.
Text that exceeds this limit after tokenization must be split before indexing.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

TOKENIZER_REPO = "intfloat/multilingual-e5-large"

# The model was trained with max 512 tokens; inputs exceeding this are truncated
# by the model.  We reject or split passages above this limit.
MODEL_INPUT_LIMIT = 512

# e5 models use a query/passage prefix; the prefix itself consumes tokens.
# "query: " → 3 tokens; "passage: " → 3 tokens.
# For indexed passages we use the "passage: " prefix when counting.
PASSAGE_PREFIX = "passage: "

_lock = threading.Lock()
_cached: _TokenizerWrapper | None = None


@dataclass(frozen=True)
class TokenizerInfo:
    repo: str
    revision: str
    fingerprint: str
    model_input_limit: int


class _TokenizerWrapper:
    def __init__(self, hf_tokenizer: Any, revision: str) -> None:
        self._tok = hf_tokenizer
        self.revision = revision
        self.model_input_limit = MODEL_INPUT_LIMIT

        # Fingerprint: hash of vocab size + eos/bos ids + revision
        fp_str = (
            f"{self._tok.vocab_size}|{self._tok.bos_token_id}|{self._tok.eos_token_id}|{revision}"
        )
        self.fingerprint = hashlib.sha256(fp_str.encode()).hexdigest()[:16]

    def count_tokens(self, text: str, add_prefix: bool = True) -> int:
        """Count tokens including special tokens (BOS/EOS).

        `add_prefix=True` prepends 'passage: ' to match the embedding model's
        expected input format for indexed passages.
        """
        input_text = (PASSAGE_PREFIX + text) if add_prefix else text
        encoded = self._tok(
            input_text,
            add_special_tokens=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return len(encoded["input_ids"])

    def fits_model_limit(self, text: str, add_prefix: bool = True) -> bool:
        return self.count_tokens(text, add_prefix=add_prefix) <= self.model_input_limit

    @property
    def info(self) -> TokenizerInfo:
        return TokenizerInfo(
            repo=TOKENIZER_REPO,
            revision=self.revision,
            fingerprint=self.fingerprint,
            model_input_limit=self.model_input_limit,
        )


def get_tokenizer(revision: str | None = None) -> _TokenizerWrapper:
    """Return a cached tokenizer instance.  Fails closed if the tokenizer cannot load.

    Parameters
    ----------
    revision:
        Exact HuggingFace commit hash.  When None the current HEAD is used and
        the fingerprint will reflect whatever commit is resolved; pass an explicit
        revision for reproducible preparation runs.
    """
    global _cached
    with _lock:
        if _cached is not None:
            return _cached
        _cached = _load(revision)
        return _cached


def _load(revision: str | None) -> _TokenizerWrapper:
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise RuntimeError(
            "transformers package is required for token accounting. "
            "Install it with: uv add 'transformers>=4.40.0' tokenizers sentencepiece"
        ) from e

    logger.info("Loading tokenizer %s (revision=%s) …", TOKENIZER_REPO, revision or "HEAD")
    try:
        tok = AutoTokenizer.from_pretrained(
            TOKENIZER_REPO,
            revision=revision,
            use_fast=True,
        )
    except Exception as e:
        raise RuntimeError(
            f"Cannot load tokenizer '{TOKENIZER_REPO}' (revision={revision!r}). "
            "Check HuggingFace access and network. "
            f"Original error: {e}"
        ) from e

    resolved_rev = getattr(tok, "name_or_path", TOKENIZER_REPO)
    logger.info("Tokenizer loaded: %s", resolved_rev)
    return _TokenizerWrapper(tok, revision or "HEAD")
