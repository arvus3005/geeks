import io
import pytest
from httpx import ASGITransport, AsyncClient

from hhgoa_rag.api.app import app
from hhgoa_rag.stt.tts import SarvamTTSAdapter


@pytest.mark.asyncio
async def test_tts_adapter_missing_key_graceful():
    adapter = SarvamTTSAdapter(api_key=None)
    res = await adapter.synthesize("Hello world", language="en")
    assert res.audio_base64 is None
    assert res.error == "SARVAM_API_KEY not configured"


@pytest.mark.asyncio
async def test_tts_endpoint_empty_text():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/v1/voice/tts", json={"text": "", "language": "en"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["audio_base64"] is None
        assert data["error"] == "Empty text"


@pytest.mark.asyncio
async def test_transcribe_endpoint_missing_key():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        fake_audio = io.BytesIO(b"RIFFdummywavdata")
        resp = await client.post(
            "/v1/voice/transcribe",
            files={"file": ("audio.wav", fake_audio, "audio/wav")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["transcript"] == ""
        # If no SARVAM_API_KEY in test env, reports clean error
        assert data["error"] is not None
