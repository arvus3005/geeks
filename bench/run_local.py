"""Run the real /v1/query pipeline in-process against the local self-hosted
hybrid index and report P50/P70/P100 latency across a real query set
(Requirement 4).

In-process (ASGI transport, no HTTP/network hop) — measures backend pipeline
latency only. For a number that includes real network overhead against a
deployed instance, use bench/run_deployed.py instead.

Usage:
    uv run python -m bench.run_local --n 60
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import uuid
from datetime import UTC, datetime

from bench.queries import DEFAULT_PER_LANGUAGE, build_query_set
from bench.report import build_report, write_report

logger = logging.getLogger(__name__)


async def _run(n_per_language: int) -> None:
    from httpx import ASGITransport, AsyncClient

    from hhgoa_rag.api.app import app
    from hhgoa_rag.api.resources import get_resources

    queries = build_query_set(per_language=n_per_language)
    results: list[dict] = []

    async with app.router.lifespan_context(app):
        resources = get_resources()
        if not resources.ready:
            raise RuntimeError(
                "Local hybrid index not ready after startup "
                f"(detail={resources.readiness_detail}). "
                "Check artifacts/full_local_index has built shards before benchmarking."
            )

        # Warm the local embedder before timing — in a real deployed server this
        # load happens once at process startup (see CLAUDE.md), never per-query.
        # Without this, the first timed query would eat a one-time ~minutes-long
        # model-load cost that has nothing to do with steady-state latency.
        from hhgoa_rag.retrieval.local_embedder import embed_query

        logger.info("Warming up local embedder before timed run…")
        embed_query("warmup")

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://bench.local") as client:
            for q in queries:
                payload = {
                    "question": q["text"],
                    "language_hint": q["language"],
                    "request_id": str(uuid.uuid4()),
                }
                try:
                    resp = await client.post("/v1/query", json=payload)
                    body = resp.json()
                    results.append(
                        {
                            "query": q["text"],
                            "language": q["language"],
                            "decision": body.get("decision", "error"),
                            "reason_code": body.get("reason_code"),
                            "total_backend_ms": body.get("total_backend_ms"),
                            "stages_ms": body.get("timings_ms", {}),
                        }
                    )
                except Exception as exc:  # noqa: BLE001 — record and continue benchmarking
                    logger.error("Query failed: %s", exc)
                    results.append(
                        {
                            "query": q["text"],
                            "language": q["language"],
                            "decision": "error",
                            "total_backend_ms": None,
                            "stages_ms": {},
                            "error": str(exc),
                        }
                    )

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    report = build_report(run_id=run_id, target="local-in-process", results=results)
    json_path, md_path = write_report(report)

    p = report["total_backend_ms_percentiles"]
    print(f"\nWrote {json_path} and {md_path}")
    if p:
        print(
            f"P50={p['p50']:.1f}ms  P70={p['p70']:.1f}ms  "
            f"P95={p['p95']:.1f}ms  P99={p['p99']:.1f}ms  P100={p['p100']:.1f}ms  "
            f"(n={p['count']})"
        )
    else:
        print("No successful queries — check readiness/credentials.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n",
        type=int,
        default=DEFAULT_PER_LANGUAGE,
        help="Queries per language to run (hi/bn native + en translation each contribute)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(_run(args.n))


if __name__ == "__main__":
    main()
