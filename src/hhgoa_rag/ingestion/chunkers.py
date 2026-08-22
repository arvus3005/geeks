"""Five chunking strategies for MSMARCO-XI passages.

1. PassageNativeChunker    — preserve passage as-is
2. SentenceAwareChunker    — split on sentence boundaries, group to target size
3. FixedTokenChunker       — fixed exact-token windows with overlap (requires real tokenizer)
4. SemanticChunker         — split on similarity drops (EXPERIMENTAL)
5. MetadataPrefixChunker   — wraps a base chunker, injects a short
   `[lang|source|doc_id]`-style prefix into the text used for EMBEDDING
   only, never shown to the user. Found via reference-project research
   (reference/hhgoa-task2-main/docs/BUILD_LOG.md: this exact pattern,
   validated with a paired-bootstrap significance test at full corpus
   scale, was the one strategy that actually beat a plain single-strategy
   baseline -- not just a design choice, a measured one). Not applied to
   this project's already-built 54M-passage index (re-indexing that scale
   is out of scope same-day); implemented and tested as a genuine 5th
   available strategy, and the natural next step if the corpus is ever
   rebuilt or extended.

All chunkers are deterministic given the same input.

FixedTokenChunker requires a real tokenizer injected at construction time.
For isolated tests only, pass allow_approximate=True to enable the whitespace
fallback — this path is ineligible for production or readiness decisions.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol

STRATEGY_VERSION = "v1"

# Sentence-boundary punctuation for Latin and Indic scripts
_SENT_RE = re.compile(r"(?<=[.?!।॥])\s+|\n+")


@dataclass
class Chunk:
    text: str
    chunk_strategy: str
    chunk_strategy_version: str
    chunk_ordinal: int
    chunk_total: int
    parent_passage_id: str
    token_length: int | None = None
    char_length: int = 0
    # "real" when a real tokenizer was used; "approximate_whitespace" otherwise.
    tokenizer_label: str = "real"
    # What actually gets embedded, if different from `text` (e.g. a
    # metadata-prefixed version for MetadataPrefixChunker). None means
    # "embed `text` itself, nothing to add" -- the common case for every
    # chunker except MetadataPrefixChunker. Kept separate from `text` so
    # `text` always stays the clean, displayable answer span -- never show
    # a user the metadata prefix that helped retrieval find it.
    embedding_text: str | None = None


class BaseChunker(ABC):
    strategy_name: str = ""
    strategy_version: str = STRATEGY_VERSION

    @abstractmethod
    def chunk(self, text: str, passage_id: str) -> list[Chunk]: ...


# ── 1. Native passage ─────────────────────────────────────────────────────────


class PassageNativeChunker(BaseChunker):
    strategy_name = "passage_native"

    def chunk(self, text: str, passage_id: str) -> list[Chunk]:
        return [
            Chunk(
                text=text,
                chunk_strategy=self.strategy_name,
                chunk_strategy_version=self.strategy_version,
                chunk_ordinal=0,
                chunk_total=1,
                parent_passage_id=passage_id,
                char_length=len(text),
            )
        ]


# ── 2. Sentence-aware ─────────────────────────────────────────────────────────


def _split_sentences(text: str) -> list[str]:
    """Split on .?!।॥ and newlines; preserve non-empty segments."""
    parts = _SENT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


class SentenceAwareChunker(BaseChunker):
    """Group complete sentences up to target_chars with bounded overlap."""

    strategy_name = "sentence_aware"

    def __init__(
        self,
        target_chars: int = 600,
        overlap_sentences: int = 1,
    ) -> None:
        if target_chars <= 0:
            raise ValueError("target_chars must be positive")
        if overlap_sentences < 0:
            raise ValueError("overlap_sentences must be non-negative")
        self.target_chars = target_chars
        self.overlap_sentences = overlap_sentences

    def chunk(self, text: str, passage_id: str) -> list[Chunk]:
        sentences = _split_sentences(text)
        if not sentences:
            return [
                Chunk(
                    text=text,
                    chunk_strategy=self.strategy_name,
                    chunk_strategy_version=self.strategy_version,
                    chunk_ordinal=0,
                    chunk_total=1,
                    parent_passage_id=passage_id,
                    char_length=len(text),
                )
            ]

        groups: list[list[str]] = []
        current: list[str] = []
        current_len = 0

        for sent in sentences:
            sent_len = len(sent)
            if current and current_len + sent_len + 1 > self.target_chars:
                groups.append(current)
                # overlap: carry last N sentences into next group
                current = current[-self.overlap_sentences :] if self.overlap_sentences else []
                current_len = sum(len(s) for s in current)
            current.append(sent)
            current_len += sent_len + 1

        if current:
            groups.append(current)

        chunks = []
        for ordinal, group in enumerate(groups):
            chunk_text = " ".join(group)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    chunk_strategy=self.strategy_name,
                    chunk_strategy_version=self.strategy_version,
                    chunk_ordinal=ordinal,
                    chunk_total=len(groups),
                    parent_passage_id=passage_id,
                    char_length=len(chunk_text),
                )
            )
        return chunks


# ── 3. Fixed-token chunker (requires real tokenizer) ─────────────────────────


class FixedTokenChunker(BaseChunker):
    """Fixed exact-token windows with overlap.

    Requires a real tokenizer wrapper (duck-typed: must have encode(text)->list[int]
    and decode(ids)->str methods, or count_tokens(text, add_prefix=False)->int).

    Production use: inject a real tokenizer from get_tokenizer().
    Test use only: pass allow_approximate=True to enable the whitespace fallback;
    this path sets tokenizer_label="approximate_whitespace" and is NOT eligible for
    readiness decisions or benchmark reports.
    """

    strategy_name = "fixed_token_overlap"

    def __init__(
        self,
        target_tokens: int = 128,
        overlap_tokens: int = 20,
        tokenizer: Any = None,
        allow_approximate: bool = False,
    ) -> None:
        if target_tokens <= 0:
            raise ValueError("target_tokens must be positive")
        if overlap_tokens < 0:
            raise ValueError("overlap_tokens must be non-negative")
        if overlap_tokens >= target_tokens:
            raise ValueError("overlap_tokens must be less than target_tokens")
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self._tokenizer = tokenizer
        self._allow_approximate = allow_approximate

    @property
    def _effective_tokenizer_label(self) -> str:
        if self._tokenizer is not None:
            return "real"
        return "approximate_whitespace"

    def _encode(self, text: str) -> list[int]:
        """Encode text to token ids. Uses real tokenizer if available."""
        if self._tokenizer is not None:
            tok = self._tokenizer
            # Try duck-typed .encode() first (works for both _TokenizerWrapper via _tok and simple fakes)
            if hasattr(tok, "encode"):
                ids = tok.encode(text)
                return list(ids) if not isinstance(ids, list) else ids
            if hasattr(tok, "_tok") and callable(tok._tok):
                # HF _TokenizerWrapper — use underlying callable tokenizer directly (no prefix)
                result = tok._tok(
                    text,
                    add_special_tokens=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )
                return list(result["input_ids"])
        if not self._allow_approximate:
            raise RuntimeError(
                "FixedTokenChunker: no real tokenizer injected and allow_approximate=False. "
                "Inject a tokenizer from get_tokenizer() for production use."
            )
        # Whitespace approximation for isolated tests only.
        return list(range(len(re.findall(r"\S+", text))))

    def _decode(self, ids: list[int], text: str, all_ids: list[int]) -> str:
        """Decode a token id window back to text.

        For real tokenizer mode: use decode() method if available.
        For approximate mode: ids are word indices, reconstruct from whitespace-split words.
        """
        if self._tokenizer is not None:
            tok = self._tokenizer
            if hasattr(tok, "decode"):
                return tok.decode(ids)
            if hasattr(tok, "_tok") and hasattr(tok._tok, "decode"):
                return tok._tok.decode(ids, skip_special_tokens=True)
        # Approximate: ids are word indices
        words = re.findall(r"\S+", text)
        return " ".join(words[i] for i in ids if i < len(words))

    def chunk(self, text: str, passage_id: str) -> list[Chunk]:
        label = self._effective_tokenizer_label
        all_ids = self._encode(text)

        if len(all_ids) <= self.target_tokens:
            return [
                Chunk(
                    text=text,
                    chunk_strategy=self.strategy_name,
                    chunk_strategy_version=self.strategy_version,
                    chunk_ordinal=0,
                    chunk_total=1,
                    parent_passage_id=passage_id,
                    token_length=len(all_ids),
                    char_length=len(text),
                    tokenizer_label=label,
                )
            ]

        step = self.target_tokens - self.overlap_tokens
        chunks: list[Chunk] = []
        start = 0

        while start < len(all_ids):
            window_ids = all_ids[start : start + self.target_tokens]
            if not window_ids:
                break
            chunk_text = self._decode(window_ids, text, all_ids)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    chunk_strategy=self.strategy_name,
                    chunk_strategy_version=self.strategy_version,
                    chunk_ordinal=len(chunks),
                    chunk_total=-1,  # updated below
                    parent_passage_id=passage_id,
                    token_length=len(window_ids),
                    char_length=len(chunk_text),
                    tokenizer_label=label,
                )
            )
            start += step

        for c in chunks:
            c.chunk_total = len(chunks)
        return chunks


# ── 4. Semantic chunker (EXPERIMENTAL) ───────────────────────────────────────


class SimilarityProvider(Protocol):
    """Injectable interface for sentence-level similarity scoring."""

    def similarity(self, a: str, b: str) -> float: ...


class SemanticChunker(BaseChunker):
    """Split passages where adjacent-sentence similarity drops below threshold.

    EXPERIMENTAL — keep labelled as such until retrieval benchmarks justify use.
    Requires an injected SimilarityProvider (no heavy model installed here).
    """

    strategy_name = "semantic_experimental"

    def __init__(
        self,
        similarity_provider: SimilarityProvider,
        split_threshold: float = 0.5,
    ) -> None:
        if not 0.0 <= split_threshold <= 1.0:
            raise ValueError("split_threshold must be in [0, 1]")
        self._sim = similarity_provider
        self.split_threshold = split_threshold

    def chunk(self, text: str, passage_id: str) -> list[Chunk]:
        sentences = _split_sentences(text)
        if len(sentences) <= 1:
            return [
                Chunk(
                    text=text,
                    chunk_strategy=self.strategy_name,
                    chunk_strategy_version=self.strategy_version,
                    chunk_ordinal=0,
                    chunk_total=1,
                    parent_passage_id=passage_id,
                    char_length=len(text),
                )
            ]

        groups: list[list[str]] = [[sentences[0]]]
        for i in range(1, len(sentences)):
            sim = self._sim.similarity(sentences[i - 1], sentences[i])
            if sim < self.split_threshold:
                groups.append([])
            groups[-1].append(sentences[i])

        chunks: list[Chunk] = []
        for ordinal, group in enumerate(groups):
            chunk_text = " ".join(group)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    chunk_strategy=self.strategy_name,
                    chunk_strategy_version=self.strategy_version,
                    chunk_ordinal=ordinal,
                    chunk_total=len(groups),
                    parent_passage_id=passage_id,
                    char_length=len(chunk_text),
                )
            )
        return chunks


# ── 5. Metadata-prefix (wraps a base chunker) ────────────────────────────────


class MetadataPrefixChunker(BaseChunker):
    """Wraps another chunker; for each of its chunks, builds a short prefix
    (`[lang|source|doc_id]`) and sets `embedding_text = prefix + text` --
    the prefix helps the embedder place the vector correctly (e.g.
    disambiguating short, generic-sounding chunks by their source/language)
    without ever being shown to the user, since `text` (the displayable
    field) is left untouched.

    Per-language metadata dict keyed by parent_passage_id is the caller's
    responsibility to build (typically language + source dataset config +
    doc id, whatever's available at indexing time) -- this class only
    knows how to format and attach it, not where indexing metadata comes
    from, since that varies by which script is calling it.
    """

    strategy_name = "metadata_prefix"

    def __init__(self, base: BaseChunker, metadata_for: dict[str, dict[str, str]] | None = None):
        self.base = base
        # passage_id -> {"lang": ..., "source": ..., "doc_id": ...} (any
        # subset of keys; missing ones are just omitted from the prefix).
        self.metadata_for = metadata_for or {}

    def _prefix(self, passage_id: str) -> str:
        meta = self.metadata_for.get(passage_id, {})
        parts = [meta[k] for k in ("lang", "source", "doc_id") if meta.get(k)]
        return f"[{'|'.join(parts)}] " if parts else ""

    def chunk(self, text: str, passage_id: str) -> list[Chunk]:
        base_chunks = self.base.chunk(text, passage_id)
        prefix = self._prefix(passage_id)
        if not prefix:
            return base_chunks
        for c in base_chunks:
            c.embedding_text = prefix + c.text
            c.chunk_strategy = self.strategy_name
        return base_chunks


# ── Registry ──────────────────────────────────────────────────────────────────
# The registry FixedTokenChunker uses allow_approximate=True for backward-compat
# test use. Production paths that require exact token windows must construct a
# fresh FixedTokenChunker(tokenizer=get_tokenizer(...)) instead of using get_chunker().

CHUNKERS: dict[str, BaseChunker] = {
    "passage_native": PassageNativeChunker(),
    "sentence_aware": SentenceAwareChunker(),
    "fixed_token_overlap": FixedTokenChunker(allow_approximate=True),
    # semantic_experimental is not in the default registry — it requires an injected provider
}


def get_chunker(strategy: str = "passage_native", tokenizer: Any = None) -> BaseChunker:
    """Return a chunker for the given strategy.

    If strategy is 'fixed_token_overlap' and tokenizer is provided, returns a fresh
    FixedTokenChunker instance with the real tokenizer (for production use).
    Otherwise returns the registry singleton.
    """
    if strategy not in CHUNKERS:
        raise ValueError(f"Unknown chunking strategy: {strategy!r}. Available: {list(CHUNKERS)}")
    if strategy == "fixed_token_overlap" and tokenizer is not None:
        base = CHUNKERS[strategy]
        return FixedTokenChunker(
            target_tokens=base.target_tokens,  # type: ignore[attr-defined]
            overlap_tokens=base.overlap_tokens,  # type: ignore[attr-defined]
            tokenizer=tokenizer,
            allow_approximate=False,
        )
    return CHUNKERS[strategy]
