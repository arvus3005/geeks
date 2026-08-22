"""Regression test for the bulbul:v3 payload bug -- verified live against the
real Sarvam API on 2026-08-22: sending "pitch"/"loudness" to bulbul:v3 returns
HTTP 400 ("Pitch and loudness parameters are currently not supported for the
Bulbul V3 model"), which silently broke every voice response. Uses
httpx.MockTransport so no real network call or credential is needed.
"""

import httpx
import pytest

from hhgoa_rag.stt.tts import SarvamTTSAdapter


@pytest.mark.asyncio
async def test_bulbul_v3_payload_omits_pitch_and_loudness():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, json={"audios": ["deadbeef"]})

    adapter = SarvamTTSAdapter(api_key="test-key", model="bulbul:v3", speaker="priya")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await adapter._post(client, "hello", "en-IN")

    assert "pitch" not in captured["payload"]
    assert "loudness" not in captured["payload"]
    assert captured["payload"]["model"] == "bulbul:v3"
    assert captured["payload"]["speaker"] == "priya"


@pytest.mark.asyncio
async def test_bulbul_v1_payload_keeps_pitch_and_loudness():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, json={"audios": ["deadbeef"]})

    adapter = SarvamTTSAdapter(api_key="test-key", model="bulbul:v1", speaker="meera")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await adapter._post(client, "hello", "en-IN")

    assert captured["payload"]["pitch"] == 0.0
    assert captured["payload"]["loudness"] == 1.0
