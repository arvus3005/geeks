"""Local Whisper fallback — DISABLED BY DEFAULT. Dev use only. Not part of submitted path."""
from .base import BaseSTTAdapter, TranscriptResult


class WhisperFallbackAdapter(BaseSTTAdapter):
    """Disabled by default. Set WHISPER_ENABLED=true to enable (dev only)."""

    async def transcribe(
        self, audio_bytes: bytes, language_hint: str | None = None
    ) -> TranscriptResult:
        raise NotImplementedError(
            "Local Whisper fallback is disabled. Enable only for development by setting WHISPER_ENABLED=true."
        )
