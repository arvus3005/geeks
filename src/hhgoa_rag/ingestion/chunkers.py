from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class Chunk:
    text: str
    chunk_strategy: str
    chunk_ordinal: int
    parent_passage_id: str
    token_length: int | None = None
    char_length: int = 0


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, text: str, passage_id: str) -> list[Chunk]:
        ...


class PassageNativeChunker(BaseChunker):
    """One chunk per passage — baseline strategy."""

    def chunk(self, text: str, passage_id: str) -> list[Chunk]:
        return [
            Chunk(
                text=text,
                chunk_strategy="passage_native",
                chunk_ordinal=0,
                parent_passage_id=passage_id,
                char_length=len(text),
            )
        ]


class FixedTokenChunker(BaseChunker):
    """Token-aware chunks with overlap."""

    def __init__(self, target_tokens: int = 256, overlap_ratio: float = 0.15):
        self.target_tokens = target_tokens
        self.overlap_ratio = overlap_ratio

    def chunk(self, text: str, passage_id: str) -> list[Chunk]:
        # Simple word-based approximation (4 chars per token)
        words = text.split()
        tokens_per_chunk = self.target_tokens
        overlap_tokens = int(tokens_per_chunk * self.overlap_ratio)
        step = tokens_per_chunk - overlap_tokens

        if len(words) <= tokens_per_chunk:
            return [
                Chunk(
                    text=text,
                    chunk_strategy="fixed_token",
                    chunk_ordinal=0,
                    parent_passage_id=passage_id,
                    char_length=len(text),
                )
            ]

        chunks = []
        i = 0
        ordinal = 0
        while i < len(words):
            chunk_words = words[i : i + tokens_per_chunk]
            chunk_text = " ".join(chunk_words)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    chunk_strategy="fixed_token",
                    chunk_ordinal=ordinal,
                    parent_passage_id=passage_id,
                    char_length=len(chunk_text),
                )
            )
            i += step
            ordinal += 1
        return chunks


CHUNKERS: dict[str, BaseChunker] = {
    "passage_native": PassageNativeChunker(),
    "fixed_token": FixedTokenChunker(),
}


def get_chunker(strategy: str = "passage_native") -> BaseChunker:
    return CHUNKERS[strategy]
