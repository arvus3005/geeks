import time

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from .base import BaseSTTAdapter, TranscriptResult


def _is_retryable(exc: BaseException) -> bool:
    """Network-level failures and 5xx are transient, worth retrying.
    A 4xx (bad request, bad auth key) will fail identically every time --
    retrying it just wastes the STT latency budget on a guaranteed-repeat
    failure, so this checks the real status code rather than catching
    HTTPStatusError broadly (which covers 4xx too)."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return isinstance(exc, httpx.TransportError)


# STT is a real network call to a third-party API, outside our own error
# budget -- a bare try/except (or none at all, as this had before) treats
# a transient network blip or a momentary 5xx the same as a hard failure.
# 3 attempts, exponential backoff (1s/2s/4s) -- matches the pattern task
# spec requirement 5 ("harness... retries... error recovery") asks for
# explicitly, not just a latency-budget nicety.
_sarvam_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    retry=retry_if_exception(_is_retryable),
    reraise=True,
)


class SarvamSTTAdapter(BaseSTTAdapter):
    """Sarvam Saaras v3 STT adapter."""

    def __init__(self, api_key: str, model: str = "saaras:v3"):
        self.api_key = api_key
        self.model = model

    @_sarvam_retry
    async def _post(self, client: httpx.AsyncClient, audio_bytes: bytes, language_hint: str | None):
        response = await client.post(
            "https://api.sarvam.ai/speech-to-text",
            headers={"api-subscription-key": self.api_key},
            files={"file": ("audio.wav", audio_bytes, "audio/wav")},
            data={"model": self.model, "language_code": language_hint or "hi-IN"},
        )
        response.raise_for_status()
        return response

    async def transcribe(
        self, audio_bytes: bytes, language_hint: str | None = None
    ) -> TranscriptResult:
        t0 = time.monotonic()
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await self._post(client, audio_bytes, language_hint)
            data = response.json()
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        return TranscriptResult(
            text=data.get("transcript", ""),
            language=data.get("language_code"),
            is_final=True,
            transcript_latency_ms=elapsed_ms,
        )
