import io
import threading
import time
import wave

import httpx
import numpy as np
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

_client: httpx.AsyncClient | None = None
_client_lock = threading.Lock()


def _get_stt_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        with _client_lock:
            if _client is None or _client.is_closed:
                _client = httpx.AsyncClient(
                    timeout=httpx.Timeout(25.0, connect=5.0),
                    limits=httpx.Limits(max_keepalive_connections=10, max_connections=20, keepalive_expiry=60.0),
                )
    return _client


def _detect_audio_format(audio_bytes: bytes) -> tuple[str, str]:
    if audio_bytes.startswith(b"RIFF") and b"WAVE" in audio_bytes[:12]:
        return "audio.wav", "audio/wav"
    if audio_bytes.startswith(b"\x1a\x45\xdf\xa3"):
        return "audio.webm", "audio/webm"
    if audio_bytes.startswith(b"OggS"):
        return "audio.ogg", "audio/ogg"
    if audio_bytes.startswith(b"ID3") or audio_bytes.startswith(b"\xff\xfb"):
        return "audio.mp3", "audio/mp3"
    if audio_bytes.startswith(b"fLaC"):
        return "audio.flac", "audio/flac"
    return "audio.wav", "audio/wav"


def _trim_wav_silence(wav_bytes: bytes, threshold: int = 400, pad_ms: int = 100) -> bytes:
    """Trim leading and trailing silence from 16-bit PCM WAV audio in <0.1ms."""
    if not wav_bytes.startswith(b"RIFF"):
        return wav_bytes
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_in:
            n_channels = wav_in.getnchannels()
            sampwidth = wav_in.getsampwidth()
            framerate = wav_in.getframerate()
            n_frames = wav_in.getnframes()
            if n_channels != 1 or sampwidth != 2 or n_frames == 0:
                return wav_bytes
            raw_frames = wav_in.readframes(n_frames)

        samples = np.frombuffer(raw_frames, dtype=np.int16)
        abs_samples = np.abs(samples)
        non_silent = np.where(abs_samples > threshold)[0]
        if len(non_silent) == 0:
            return wav_bytes

        pad = int(pad_ms * framerate / 1000)
        start = max(0, non_silent[0] - pad)
        end = min(len(samples), non_silent[-1] + pad)
        trimmed_samples = samples[start:end]

        out_buf = io.BytesIO()
        with wave.open(out_buf, "wb") as wav_out:
            wav_out.setnchannels(1)
            wav_out.setsampwidth(2)
            wav_out.setframerate(framerate)
            wav_out.writeframes(trimmed_samples.tobytes())
        return out_buf.getvalue()
    except Exception:
        return wav_bytes


SARVAM_STT_LANG_MAP: dict[str, str] = {
    "hi": "hi-IN",
    "bn": "bn-IN",
    "gu": "gu-IN",
    "mr": "mr-IN",
    "ta": "ta-IN",
    "ur": "ur-IN",
    "en": "en-IN",
    "te": "te-IN",
    "kn": "kn-IN",
    "ml": "ml-IN",
    "pa": "pa-IN",
    # Sarvam's own wire code for Odia is "od-IN", but every other module in
    # this codebase (language_routing.py's SUPPORTED_LANGUAGES/INDEXED_LANGUAGES,
    # dataset/models.py) uses "or" as the internal code -- and language_hint
    # arrives from the same api/routes/*.py field for BOTH retrieval routing
    # and this STT call, so it must be keyed on "or" here too. Found 2026-08-22:
    # the key used to be "od", so an explicit language_hint=or API call fell
    # through this lookup to lang_code="unknown" instead of "od-IN" -- not
    # reachable via the shipped web UI (no Odia option in its dropdown), but a
    # real gap for any direct API caller.
    "or": "od-IN",
}


class SarvamSTTAdapter(BaseSTTAdapter):
    """Sarvam Saaras v3 STT adapter."""

    def __init__(self, api_key: str, model: str = "saaras:v3"):
        self.api_key = api_key
        self.model = model

    @_sarvam_retry
    async def _post(self, client: httpx.AsyncClient, audio_bytes: bytes, language_hint: str | None):
        filename, mime_type = _detect_audio_format(audio_bytes)
        if not language_hint or language_hint in ("auto", "unknown"):
            lang_code = "unknown"
        elif language_hint in SARVAM_STT_LANG_MAP:
            lang_code = SARVAM_STT_LANG_MAP[language_hint]
        elif language_hint.endswith("-IN"):
            lang_code = language_hint
        else:
            lang_code = "unknown"

        response = await client.post(
            "https://api.sarvam.ai/speech-to-text",
            headers={"api-subscription-key": self.api_key},
            files={"file": (filename, audio_bytes, mime_type)},
            data={
                "model": self.model,
                "language_code": lang_code,
                "with_timestamps": "false",
            },
        )
        response.raise_for_status()
        return response

    async def transcribe(
        self, audio_bytes: bytes, language_hint: str | None = None
    ) -> TranscriptResult:
        t0 = time.monotonic()
        client = _get_stt_client()
        optimized_audio = _trim_wav_silence(audio_bytes)
        response = await self._post(client, optimized_audio, language_hint)
        data = response.json()
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        return TranscriptResult(
            text=data.get("transcript", ""),
            language=data.get("language_code"),
            is_final=True,
            transcript_latency_ms=elapsed_ms,
        )
