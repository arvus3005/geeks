"""Four chunking strategies for MSMARCO-XI passages.

1. PassageNativeChunker   — preserve passage as-is
2. SentenceAwareChunker   — split on sentence boundaries, group to target size
3. FixedTokenChunker      — fixed approximate-token windows with overlap
4. SemanticChunker        — split on similarity drops (EXPERIMENTAL)

All chunkers are deterministic given the same input.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

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
    # label set to "approximate_whitespace" when real tokenizer unavailable
    tokenizer_label: str = "real"


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


# ── 3. Fixed-token fallback ───────────────────────────────────────────────────


class FixedTokenChunker(BaseChunker):
    """Fixed approximate-token windows with overlap.

    Uses whitespace tokenization when a real tokenizer is unavailable.
    When the whitespace fallback is active, tokenizer_label is set to
    'approximate_whitespace' on every Chunk so downstream code can
    distinguish approximate from real token counts.
    """

    strategy_name = "fixed_token_overlap"

    def __init__(
        self,
        target_tokens: int = 128,
        overlap_tokens: int = 20,
    ) -> None:
        if target_tokens <= 0:
            raise ValueError("target_tokens must be positive")
        if overlap_tokens < 0:
            raise ValueError("overlap_tokens must be non-negative")
        if overlap_tokens >= target_tokens:
            raise ValueError("overlap_tokens must be less than target_tokens")
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self._tokenizer_label = "real"

    def _tokenize(self, text: str) -> list[str]:
        # Whitespace approximation — no heavy model dependency
        tokens = re.findall(r"\S+", text)
        self._tokenizer_label = "approximate_whitespace"
        return tokens

    def _detokenize(self, tokens: list[str]) -> str:
        return " ".join(tokens)

    def chunk(self, text: str, passage_id: str) -> list[Chunk]:
        tokens = self._tokenize(text)
        label = self._tokenizer_label

        if len(tokens) <= self.target_tokens:
            return [
                Chunk(
                    text=text,
                    chunk_strategy=self.strategy_name,
                    chunk_strategy_version=self.strategy_version,
                    chunk_ordinal=0,
                    chunk_total=1,
                    parent_passage_id=passage_id,
                    token_length=len(tokens),
                    char_length=len(text),
                    tokenizer_label=label,
                )
            ]

        step = self.target_tokens - self.overlap_tokens
        chunks: list[Chunk] = []
        start = 0

        while start < len(tokens):
            window = tokens[start : start + self.target_tokens]
            if not window:
                break
            chunk_text = self._detokenize(window)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    chunk_strategy=self.strategy_name,
                    chunk_strategy_version=self.strategy_version,
                    chunk_ordinal=len(chunks),
                    chunk_total=-1,  # updated below
                    parent_passage_id=passage_id,
                    token_length=len(window),
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


# ── Registry ──────────────────────────────────────────────────────────────────

CHUNKERS: dict[str, BaseChunker] = {
    "passage_native": PassageNativeChunker(),
    "sentence_aware": SentenceAwareChunker(),
    "fixed_token_overlap": FixedTokenChunker(),
    # semantic_experimental is not in the default registry — it requires an injected provider
}


def get_chunker(strategy: str = "passage_native") -> BaseChunker:
    if strategy not in CHUNKERS:
        raise ValueError(f"Unknown chunking strategy: {strategy!r}. Available: {list(CHUNKERS)}")
    return CHUNKERS[strategy]
