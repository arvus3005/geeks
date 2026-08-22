"""Minimal reverse-proxy ASGI app — NOT the real RAG pipeline.

Exists only so the old Render free-tier URL (hhgoa-rag-d3fw.onrender.com)
can keep working as a second live link, forwarding every request to the
real backend: the self-hosted local hybrid index served from the indexing
machine via an ngrok tunnel. Render's free tier (512MB RAM) cannot run the
real app — that needs ~6.4GB just for the live-serving shard set, and the
multi-GB index files aren't in git to begin with. This module never
imports hhgoa_rag.retrieval / local_embedder / anything from the real
pipeline, specifically so it stays lightweight enough to run there.

Deployed via render.yaml's dockerCommand override, not the default
Dockerfile CMD (which still runs the real app — this is Render-specific,
not a change to how the app normally starts).

Config: PROXY_UPSTREAM_URL (required) — the real backend's base URL, e.g.
the ngrok reserved domain.
"""

from __future__ import annotations

import os

import httpx
from fastapi import FastAPI, Request, Response

UPSTREAM = os.environ.get("PROXY_UPSTREAM_URL", "").rstrip("/")

app = FastAPI(title="HH Goa RAG API (proxy)")

_client = httpx.AsyncClient(timeout=30.0)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy(request: Request, path: str) -> Response:
    if not UPSTREAM:
        return Response(content=b'{"error":"PROXY_UPSTREAM_URL not configured"}', status_code=503)

    url = f"{UPSTREAM}/{path}"
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
    headers["ngrok-skip-browser-warning"] = "true"

    body = await request.body()
    upstream_resp = await _client.request(
        request.method, url, params=request.query_params, headers=headers, content=body
    )
    return Response(
        content=upstream_resp.content,
        status_code=upstream_resp.status_code,
        headers={
            k: v
            for k, v in upstream_resp.headers.items()
            if k.lower() not in ("content-length", "content-encoding", "transfer-encoding")
        },
    )
