from abc import ABC, abstractmethod
from dataclasses import dataclass

STRATEGY_VERSION = "v1"


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


class BaseChunker(ABC):
    strategy_name: str = ""
    strategy_version: str = STRATEGY_VERSION

    @abstractmethod
    def chunk(self, text: str, passage_id: str) -> list[Chunk]: ...


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


class FixedTokenChunker(BaseChunker):
    strategy_name = "fixed_token_overlap"

    def __init__(
        self,
        target_tokens: int = 256,
        overlap_ratio: float = 0.15,
        tokenizer_name: str = "intfloat/multilingual-e5-small",
    ):
        self.target_tokens = target_tokens
        self.overlap_ratio = overlap_ratio
        self.tokenizer_name = tokenizer_name
        self._tokenizer = None

    def _get_tokenizer(self):
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            except Exception:
                self._tokenizer = None
        return self._tokenizer

    def _tokenize(self, text: str) -> list[str]:
        tok = self._get_tokenizer()
        if tok is not None:
            return tok.tokenize(text)
        import re

        return re.findall(r"\S+", text)

    def chunk(self, text: str, passage_id: str) -> list[Chunk]:
        tokens = self._tokenize(text)

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
                )
            ]

        # Approximate by splitting text proportionally to token count
        chars_per_token = len(text) / max(len(tokens), 1)
        target_chars = int(self.target_tokens * chars_per_token)
        overlap_chars = int(target_chars * self.overlap_ratio)
        step_chars = target_chars - overlap_chars

        chunks = []
        i = 0
        ordinal = 0
        while i < len(text):
            chunk_text = text[i : i + target_chars].strip()
            if not chunk_text:
                break
            chunks.append(
                Chunk(
                    text=chunk_text,
                    chunk_strategy=self.strategy_name,
                    chunk_strategy_version=self.strategy_version,
                    chunk_ordinal=ordinal,
                    chunk_total=-1,  # updated below
                    parent_passage_id=passage_id,
                    char_length=len(chunk_text),
                )
            )
            i += step_chars
            ordinal += 1
            if i >= len(text):
                break

        for c in chunks:
            c.chunk_total = len(chunks)
        return chunks


CHUNKERS: dict[str, BaseChunker] = {
    "passage_native": PassageNativeChunker(),
    "fixed_token_overlap": FixedTokenChunker(),
}


def get_chunker(strategy: str = "passage_native") -> BaseChunker:
    if strategy not in CHUNKERS:
        raise ValueError(f"Unknown chunking strategy: {strategy}. Available: {list(CHUNKERS)}")
    return CHUNKERS[strategy]
