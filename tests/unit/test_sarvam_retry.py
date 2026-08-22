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
