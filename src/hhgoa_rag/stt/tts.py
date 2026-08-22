"""Text-to-Speech (TTS) adapter using Sarvam AI's Bulbul v3 model for Indic languages.

Supports hi, bn, gu, ta, mr, ur, en, te, kn, ml, or, pa (see SARVAM_LANG_MAP
below for the exact wire codes) -- not as/ne/sa, which have no Sarvam TTS
mapping at all.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# Map this project's own internal language codes (language_routing.py's
# SUPPORTED_LANGUAGES/INDEXED_LANGUAGES) to Sarvam's wire codes. Odia is keyed
# on "or" here, not Sarvam's own "od" -- see the 2026-08-22 note in
# src/hhgoa_rag/stt/sarvam.py's SARVAM_STT_LANG_MAP for the real bug this
# fixes: synthesize()'s `language` argument arrives as this project's internal
# code (e.g. `detected_lang`), so keying on Sarvam's own code here meant an
# Odia response silently fell back to English TTS.
SARVAM_LANG_MAP: dict[str, str] = {
    "hi": "hi-IN",
    "bn": "bn-IN",
    "gu": "gu-IN",
    "ta": "ta-IN",
    "mr": "mr-IN",
    "ur": "ur-IN",
    "en": "en-IN",
    "te": "te-IN",
    "kn": "kn-IN",
    "ml": "ml-IN",
    "or": "od-IN",
    "pa": "pa-IN",
}


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return isinstance(exc, httpx.TransportError)


_tts_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    retry=retry_if_exception(_is_retryable),
    reraise=True,
)


@dataclass
class TTSResult:
    audio_base64: str | None
    language_code: str
    latency_ms: float
    error: str | None = None


import threading

_tts_client: httpx.AsyncClient | None = None
_tts_client_lock = threading.Lock()


def _get_tts_client() -> httpx.AsyncClient:
    global _tts_client
    if _tts_client is None or _tts_client.is_closed:
        with _tts_client_lock:
            if _tts_client is None or _tts_client.is_closed:
                _tts_client = httpx.AsyncClient(
                    timeout=httpx.Timeout(20.0, connect=5.0),
                    limits=httpx.Limits(max_keepalive_connections=10, max_connections=20, keepalive_expiry=60.0),
                )
    return _tts_client


class SarvamTTSAdapter:
    """Sarvam Bulbul v3 Text-to-Speech adapter for Indic languages."""

    def __init__(self, api_key: str | None = None, model: str = "bulbul:v3", speaker: str = "priya"):
        self.api_key = api_key
        self.model = model
        self.speaker = speaker

    @_tts_retry
    async def _post(self, client: httpx.AsyncClient, text: str, lang_code: str):
        payload = {
            "inputs": [text[:500]],
            "target_language_code": lang_code,
            "speaker": self.speaker,
            "pace": 1.0,
            "speech_sample_rate": 16000,
            "enable_preprocessing": True,
            "model": self.model,
        }
        if not self.model.startswith("bulbul:v3"):
            payload["pitch"] = 0.0
            payload["loudness"] = 1.0
        response = await client.post(
            "https://api.sarvam.ai/text-to-speech",
            headers={
                "api-subscription-key": self.api_key or "",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        return response

    async def synthesize(self, text: str, language: str = "en") -> TTSResult:
        if not text or not text.strip():
            return TTSResult(audio_base64=None, language_code=language, latency_ms=0.0, error="Empty text")

        if not self.api_key:
            return TTSResult(
                audio_base64=None,
                language_code=language,
                latency_ms=0.0,
                error="SARVAM_API_KEY not configured",
            )

        lang_code = SARVAM_LANG_MAP.get(language, "hi-IN" if language in ("hi", "mr") else "en-IN")
        t0 = time.monotonic()

        try:
            client = _get_tts_client()
            response = await self._post(client, text, lang_code)
            data = response.json()
            audios = data.get("audios", [])
            audio_b64 = audios[0] if audios else None

            elapsed_ms = (time.monotonic() - t0) * 1000.0
            return TTSResult(
                audio_base64=audio_b64,
                language_code=lang_code,
                latency_ms=elapsed_ms,
            )
        except Exception as e:
            logger.warning("Sarvam TTS synthesis failed: %s", e)
            return TTSResult(
                audio_base64=None,
                language_code=lang_code,
                latency_ms=(time.monotonic() - t0) * 1000.0,
                error=str(e),
            )
