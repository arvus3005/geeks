"""Run the real /v1/query pipeline over real HTTP against a deployed instance
and report P50/P70/P100 latency (Requirement 4) — includes real network overhead,
unlike bench/run_local.py's in-process measurement.

Usage:
    uv run python -m bench.run_deployed --url https://your-deployment --n 60
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


async def _run(base_url: str, n_per_language: int) -> None:
    from httpx import AsyncClient

    queries = build_query_set(per_language=n_per_language)
    results: list[dict] = []

    async with AsyncClient(base_url=base_url, timeout=30.0) as client:
        health = await client.get("/health/ready")
        if health.status_code != 200:
            raise RuntimeError(
                f"Deployment at {base_url} is not ready (health/ready returned "
                f"{health.status_code}): {health.text}"
            )

        for q in queries:
            payload = {
                "question": q["text"],
                "language_hint": q["language"],
                "request_id": str(uuid.uuid4()),
            }
            import time

            t0 = time.monotonic()
            try:
                resp = await client.post("/v1/query", json=payload)
                wall_ms = (time.monotonic() - t0) * 1000.0
                body = resp.json()
                stages = dict(body.get("timings_ms", {}))
                stages["wall_clock_incl_network"] = wall_ms
                results.append(
                    {
                        "query": q["text"],
                        "language": q["language"],
                        "decision": body.get("decision", "error"),
                        "reason_code": body.get("reason_code"),
                        "total_backend_ms": wall_ms,
                        "stages_ms": stages,
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
    report = build_report(run_id=run_id, target=f"deployed:{base_url}", results=results)
    json_path, md_path = write_report(report)

    p = report["total_backend_ms_percentiles"]
    print(f"\nWrote {json_path} and {md_path}")
    if p:
        print(
            f"P50={p['p50']:.1f}ms  P70={p['p70']:.1f}ms  "
            f"P95={p['p95']:.1f}ms  P99={p['p99']:.1f}ms  P100={p['p100']:.1f}ms  "
            f"(n={p['count']}, wall-clock incl. network)"
        )
    else:
        print("No successful queries — check the deployment URL and readiness.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Base URL of the deployed instance")
    parser.add_argument("--n", type=int, default=DEFAULT_PER_LANGUAGE, help="Queries per language")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(_run(args.url.rstrip("/"), args.n))


if __name__ == "__main__":
    main()
