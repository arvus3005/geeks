"""Regression test for the bulbul:v3 payload bug -- verified live against the
real Sarvam API on 2026-08-22: sending "pitch"/"loudness" to bulbul:v3 returns
HTTP 400 ("Pitch and loudness parameters are currently not supported for the
Bulbul V3 model"), which silently broke every voice response. Uses
httpx.MockTransport so no real network call or credential is needed.
"""

import httpx
import pytest

from hhgoa_rag.stt.tts import SARVAM_LANG_MAP, SarvamTTSAdapter


def test_internal_odia_code_resolves_to_sarvam_wire_code():
    # Real 2026-08-22 bug: this map's key used to be "od" (Sarvam's own wire
    # code), but language_routing.py's SUPPORTED_LANGUAGES/INDEXED_LANGUAGES --
    # and everything upstream that produces the `detected_lang` passed into
    # synthesize(text, language=detected_lang) -- use "or" as the internal
    # code for Odia. Looking up "or" missed the map entirely and silently fell
    # back to English TTS for every Odia response. Same class of bug as
    # sarvam.py's STT map, fixed the same way: key on this project's own
    # internal code, not Sarvam's.
    assert SARVAM_LANG_MAP.get("or") == "od-IN"
    assert "od" not in SARVAM_LANG_MAP  # Sarvam's own code is not a valid caller input


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
