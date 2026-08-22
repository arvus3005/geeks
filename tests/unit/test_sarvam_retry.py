"""Real tests for SarvamSTTAdapter's retry behavior -- not assumed, verified.

Uses httpx's MockTransport so no real network call or credential is
needed, while still exercising the real tenacity-decorated code path
(retry/backoff/reraise), not a hand-rolled substitute for it.
"""

import httpx
import pytest

from hhgoa_rag.stt.sarvam import SarvamSTTAdapter


def _adapter() -> SarvamSTTAdapter:
    return SarvamSTTAdapter(api_key="test-key")


def _capturing_handler(captured: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        # multipart/form-data body -- find the language_code PART specifically
        # (by its Content-Disposition name), not just "the first value line" --
        # that grabbed whichever field happened to come first (e.g. "model").
        parts = request.read().decode("utf-8", errors="ignore").split("\r\n\r\n")
        for i, part in enumerate(parts[:-1]):
            if 'name="language_code"' in part:
                captured["language_code"] = parts[i + 1].split("\r\n")[0]
                break
        return httpx.Response(200, json={"transcript": "ok", "language_code": "or-IN"})
    return handler


@pytest.mark.parametrize(
    "language_hint,expected_lang_code",
    [
        ("hi", "hi-IN"),
        ("or", "od-IN"),  # this project's own internal code for Odia everywhere
        # else (language_routing.py's SUPPORTED_LANGUAGES/INDEXED_LANGUAGES) is
        # "or", not Sarvam's own "od" -- language_hint arrives from the same
        # source (api/routes/*.py's language_hint field) for both retrieval
        # routing AND this STT call, so it must resolve through THIS map using
        # "or", not require callers to already know Sarvam's own code.
        ("od", None),  # Sarvam's own code is not a valid INPUT to this adapter
        ("as", None),  # not wired at all -- falls back to auto-detect, not a crash
    ],
)
@pytest.mark.asyncio
async def test_language_hint_resolves_to_sarvam_lang_code(language_hint, expected_lang_code):
    captured: dict = {}
    adapter = _adapter()
    async with httpx.AsyncClient(transport=httpx.MockTransport(_capturing_handler(captured))) as client:
        await adapter._post(client, b"audio", language_hint)
    if expected_lang_code is None:
        assert captured["language_code"] == "unknown"
    else:
        assert captured["language_code"] == expected_lang_code


@pytest.mark.asyncio
async def test_succeeds_first_try_no_retry():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"transcript": "hello", "language_code": "en-IN"})

    adapter = _adapter()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        resp = await adapter._post(client, b"audio", None)
    assert resp.status_code == 200
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_retries_on_500_then_succeeds():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503)
        return httpx.Response(200, json={"transcript": "ok", "language_code": "hi-IN"})

    adapter = _adapter()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        resp = await adapter._post(client, b"audio", None)
    assert resp.status_code == 200
    assert calls["n"] == 3  # failed twice, succeeded on the 3rd (allowed) attempt


@pytest.mark.asyncio
async def test_gives_up_after_3_attempts_on_persistent_500():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500)

    adapter = _adapter()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await adapter._post(client, b"audio", None)
    assert calls["n"] == 3  # stop_after_attempt(3), no more


@pytest.mark.asyncio
async def test_does_not_retry_on_4xx():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401)

    adapter = _adapter()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await adapter._post(client, b"audio", None)
    assert calls["n"] == 1  # a bad API key fails identically every time -- don't waste retries on it
