"""Render a latency benchmark run (list of per-query results) into JSON + Markdown."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bench.percentiles import compute_percentiles


def build_report(
    run_id: str,
    target: str,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """results: list of {query, language, decision, total_backend_ms, stages_ms, error}"""
    ok_results = [r for r in results if r.get("total_backend_ms") is not None]
    totals = [r["total_backend_ms"] for r in ok_results]
    decisions: dict[str, int] = {}
    for r in results:
        decisions[r.get("decision", "error")] = decisions.get(r.get("decision", "error"), 0) + 1

    percentiles = compute_percentiles(totals) if totals else None

    stage_names = sorted({k for r in ok_results for k in r.get("stages_ms", {})})
    stage_percentiles = {}
    for name in stage_names:
        vals = [r["stages_ms"][name] for r in ok_results if name in r["stages_ms"]]
        if vals:
            stage_percentiles[name] = compute_percentiles(vals)

    return {
        "run_id": run_id,
        "target": target,
        "generated_at": datetime.now(UTC).isoformat(),
        "n_queries": len(results),
        "n_ok": len(ok_results),
        "n_failed": len(results) - len(ok_results),
        "decisions": decisions,
        "total_backend_ms_percentiles": percentiles,
        "stage_percentiles_ms": stage_percentiles,
        "results": results,
    }


def render_markdown(report: dict[str, Any]) -> str:
    p = report["total_backend_ms_percentiles"]
    lines = [
        f"# Latency Benchmark — {report['run_id']}",
        "",
        f"- **Target**: {report['target']}",
        f"- **Generated**: {report['generated_at']}",
        f"- **Queries**: {report['n_queries']} ({report['n_ok']} ok, {report['n_failed']} failed)",
        f"- **Decisions**: {report['decisions']}",
        "",
        "## End-to-end backend latency (ms)",
        "",
    ]
    if p is None:
        lines.append("_No successful queries — percentiles unavailable._")
    else:
        lines += [
            "| P50 | P70 | P95 | P99 | P100 | Mean |",
            "|---|---|---|---|---|---|",
            f"| {p['p50']:.1f} | {p['p70']:.1f} | {p['p95']:.1f} | {p['p99']:.1f} "
            f"| {p['p100']:.1f} | {p['mean']:.1f} |",
        ]

    if report["stage_percentiles_ms"]:
        lines += ["", "## Per-stage P50 / P100 (ms)", "", "| Stage | P50 | P100 |", "|---|---|---|"]
        for name, sp in report["stage_percentiles_ms"].items():
            lines.append(f"| {name} | {sp['p50']:.1f} | {sp['p100']:.1f} |")

    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any], reports_dir: Path = Path("artifacts/reports")) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / f"latency_benchmark_{report['run_id']}.json"
    md_path = reports_dir / f"latency_benchmark_{report['run_id']}.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    md_path.write_text(render_markdown(report))
    return json_path, md_path
