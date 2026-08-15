from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TranscriptResult:
    text: str
    language: str | None
    is_final: bool
    connection_time_ms: float = 0.0
    audio_duration_ms: float = 0.0
    transcript_latency_ms: float = 0.0


class BaseSTTAdapter(ABC):
    @abstractmethod
    async def transcribe(
        self, audio_bytes: bytes, language_hint: str | None = None
    ) -> TranscriptResult: ...
