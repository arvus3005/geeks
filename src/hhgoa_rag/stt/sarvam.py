import time

import httpx

from .base import BaseSTTAdapter, TranscriptResult


class SarvamSTTAdapter(BaseSTTAdapter):
    """Sarvam Saaras v2 STT adapter."""

    def __init__(self, api_key: str, model: str = "saaras:v2"):
        self.api_key = api_key
        self.model = model

    async def transcribe(
        self, audio_bytes: bytes, language_hint: str | None = None
    ) -> TranscriptResult:
        t0 = time.monotonic()
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.sarvam.ai/speech-to-text",
                headers={"api-subscription-key": self.api_key},
                files={"file": ("audio.wav", audio_bytes, "audio/wav")},
                data={"model": self.model, "language_code": language_hint or "hi-IN"},
            )
            response.raise_for_status()
            data = response.json()
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        return TranscriptResult(
            text=data.get("transcript", ""),
            language=data.get("language_code"),
            is_final=True,
            transcript_latency_ms=elapsed_ms,
        )
