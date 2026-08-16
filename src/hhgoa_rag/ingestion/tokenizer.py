"""Real multilingual-E5 tokenizer for exact token accounting.

Loads the tokenizer for `intfloat/multilingual-e5-large` from HuggingFace.
No model weights, no PyTorch — tokenizer only.

Usage
-----
    tok = get_tokenizer()                    # cached (HEAD revision)
    tok = get_tokenizer(revision="abc123")  # cached by revision SHA
    n = tok.count_tokens("query: hello")    # includes BOS/EOS special tokens
    fp = tok.fingerprint                     # reproducible revision string

Model input limit: 512 tokens (including special tokens) per multilingual-e5-large.
Text that exceeds this limit after tokenization must be split before indexing.

Caching is revision-keyed: passing a different revision SHA returns (and caches)
a distinct wrapper instance. This prevents stale-revision cross-contamination
when calling get_tokenizer() with different revision arguments in the same process.

HF cache location: .cache/huggingface (writable, .gitignored).
"""

from __future__ import annotations

import hashlib
import logging
import os
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

# Revision-keyed cache: maps resolved_revision_str → _TokenizerWrapper
_lock = threading.Lock()
_cache: dict[str, _TokenizerWrapper] = {}

# Configure writable HF cache location
_HF_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    ".cache",
    "huggingface",
)


@dataclass(frozen=True)
class TokenizerInfo:
    repo: str
    revision: str
    fingerprint: str
    model_input_limit: int


class _TokenizerWrapper:
    def __init__(self, hf_tokenizer: Any, resolved_revision: str) -> None:
        self._tok = hf_tokenizer
        self.revision = resolved_revision
        self.model_input_limit = MODEL_INPUT_LIMIT

        # Fingerprint: hash of vocab size + eos/bos ids + resolved revision
        fp_str = (
            f"{self._tok.vocab_size}"
            f"|{self._tok.bos_token_id}"
            f"|{self._tok.eos_token_id}"
            f"|{resolved_revision}"
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
    """Return a cached tokenizer instance keyed by revision.

    Parameters
    ----------
    revision:
        Exact HuggingFace commit hash.  When None the current HEAD is used.
        Two calls with different revision values return different (cached) instances.
        Passing revision=None always returns the same HEAD instance.

    Raises RuntimeError if the tokenizer cannot be loaded — never silently skips.
    """
    cache_key = revision if revision is not None else "__HEAD__"
    with _lock:
        if cache_key in _cache:
            return _cache[cache_key]
        wrapper = _load(revision)
        _cache[cache_key] = wrapper
        return wrapper


def _load(revision: str | None) -> _TokenizerWrapper:
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise RuntimeError(
            "transformers package is required for token accounting. "
            "Install it with: uv add 'transformers>=4.40.0' tokenizers sentencepiece"
        ) from e

    logger.info("Loading tokenizer %s (revision=%s) …", TOKENIZER_REPO, revision or "HEAD")

    # Configure writable cache directory
    os.makedirs(_HF_CACHE_DIR, exist_ok=True)

    try:
        tok = AutoTokenizer.from_pretrained(
            TOKENIZER_REPO,
            revision=revision,
            use_fast=True,
            cache_dir=_HF_CACHE_DIR,
        )
    except Exception as e:
        raise RuntimeError(
            f"Cannot load tokenizer '{TOKENIZER_REPO}' (revision={revision!r}). "
            "Check HuggingFace access and network. "
            f"Original error: {e}"
        ) from e

    # Resolve the actual revision string stored in the tokenizer
    # Use the passed revision (or a sentinel for HEAD) as the cache key component
    resolved_rev = revision if revision is not None else "HEAD"
    logger.info(
        "Tokenizer loaded: %s revision=%s fingerprint-input=%s",
        TOKENIZER_REPO,
        resolved_rev,
        resolved_rev,
    )
    return _TokenizerWrapper(tok, resolved_rev)
