#!/usr/bin/env python3
"""Resumable one-command canary indexer for MSMARCO-XI → Pinecone.

Default mode: offline dry-run (validates everything, schedules no writes).

Live mode requires ALL of:
  --execute
  CONFIRM_PINECONE_WRITE=1 environment variable
  PINECONE_API_KEY environment variable

Usage:
  # Dry run (safe, no credentials needed):
  uv run python scripts/index_canary.py --manifest artifacts/prepared/<id>_manifest.json

  # Live canary write:
  CONFIRM_PINECONE_WRITE=1 PINECONE_API_KEY=<key> \\
    uv run python scripts/index_canary.py \\
      --manifest artifacts/prepared/<id>_manifest.json \\
      --execute --resume --concurrency 4

  # Resume after interruption:
  CONFIRM_PINECONE_WRITE=1 PINECONE_API_KEY=<key> \\
    uv run python scripts/index_canary.py \\
      --manifest artifacts/prepared/<id>_manifest.json \\
      --execute --resume

  # Background (nohup):
  nohup CONFIRM_PINECONE_WRITE=1 PINECONE_API_KEY=<key> \\
    uv run python scripts/index_canary.py \\
      --manifest artifacts/prepared/<id>_manifest.json \\
      --execute --resume > logs/index_canary.log 2>&1 &

Hard limits:
  - Only works with a valid 300-record canary manifest bound to pilot_v1.
  - Batch size capped at canonical maximum of 96.
  - Token rate: 225,000 passage tokens/minute ceiling (configurable downward).
  - Pinecone hard limit: 250,000 passage tokens/minute.
  - Retries: transient network errors, HTTP 408, 429, 5xx only.
  - Permanent 4xx (auth/schema/validation) are NOT retried.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import logging
import os
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from hhgoa_rag.pinecone_contract import (
    INDEX_NAME as CANONICAL_INDEX_NAME,
)

# ── Canonical constants ───────────────────────────────────────────────────────
from hhgoa_rag.pinecone_contract import (
    MANIFEST_SCHEMA_VERSION,
    canonical_contract,
)
from hhgoa_rag.pinecone_contract import (
    MAX_BATCH_SIZE as CANONICAL_MAX_BATCH_SIZE,
)
from hhgoa_rag.pinecone_contract import (
    NAMESPACE as CANONICAL_NAMESPACE,
)
from hhgoa_rag.pinecone_contract import (
    contract_fingerprint as canonical_fingerprint,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("index_canary")

# ── Hard limits ───────────────────────────────────────────────────────────────
CANARY_EXPECTED_TOTAL = 300
CANARY_EXPECTED_PER_LANG = 100
CANARY_EXPECTED_LANGUAGES = {"en", "hi", "bn"}

# Token rate: Pinecone Starter = 250k passage tokens/minute.
# We default to 225k/min (10% headroom) to avoid hitting the hard limit.
DEFAULT_TOKEN_RATE_LIMIT = 225_000  # passage tokens/minute

# Maximum concurrency guard — defensible maximum for Starter plan.
MAX_CONCURRENCY = 8
DEFAULT_CONCURRENCY = 4

# Retry parameters
MAX_RETRIES = 5
RETRY_BASE_DELAY = 1.0
RETRY_MAX_DELAY = 60.0

# Freshness polling
FRESHNESS_POLL_MAX_WAIT = 180  # seconds
FRESHNESS_POLL_BASE = 5  # seconds

CHECKPOINT_SCHEMA_VERSION = "1"

REQUIRED_MANIFEST_FIELDS = {
    "manifest_schema_version",
    "manifest_id",
    "manifest_checksum",
    "contract_version",
    "contract_fingerprint",
    "index_contract",
    "index_name",
    "index_namespace",
    "dataset_revision",
    "total_records",
    "total_tokens",
    "prepared_record_path",
    "prepared_record_checksum",
    "ready_for_write",
    "readiness_failures",
    "forbidden_field_audit",
    "actual_per_language_records",
}

FORBIDDEN_FIELDS = {"query", "Answer", "Eng_Query", "Eng_Answer", "query_type", "is_selected"}

_REGENERATION_COMMAND = (
    "uv run python scripts/prepare_canary.py "
    "--dataset-revision bf5cdc1f26e581e519018e434db14edd1b77602b "
    "--tokenizer-revision 3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3 "
    "--seed 42"
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _is_transient_error(exc: Exception) -> bool:
    """True for errors that are safe to retry (network, 408, 429, 5xx)."""
    msg = str(exc).lower()
    transient_patterns = [
        "timeout",
        "connection",
        "429",
        "too many requests",
        "503",
        "502",
        "500",
        "504",
        "408",
        "request timeout",
        "service unavailable",
        "bad gateway",
        "internal server error",
    ]
    return any(p in msg for p in transient_patterns)


def _retry_after(exc: Exception) -> float | None:
    """Parse Retry-After header value from exception if available."""
    msg = str(exc)
    import re

    m = re.search(r"retry.after[:\s]+(\d+)", msg, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def _batch_digest(manifest_checksum: str, batch_index: int, record_ids: list[str]) -> str:
    """Deterministic digest for a batch — based on manifest + ordered record IDs."""
    content = f"{manifest_checksum}|{batch_index}|" + "|".join(sorted(record_ids))
    return hashlib.sha256(content.encode()).hexdigest()[:16]


# ── Manifest loading ──────────────────────────────────────────────────────────


def _load_manifest(manifest_path: Path) -> dict:
    """Load and strictly validate the canary manifest. Raises on any failure."""
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}\n" f"Regenerate with:\n  {_REGENERATION_COMMAND}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    missing = REQUIRED_MANIFEST_FIELDS - set(manifest.keys())
    if missing:
        raise ValueError(f"Manifest missing required fields: {sorted(missing)}")

    # Schema version
    if manifest.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Manifest schema version {manifest.get('manifest_schema_version')!r} "
            f"is not the required {MANIFEST_SCHEMA_VERSION!r}. "
            "Regenerate the manifest."
        )

    # Manifest checksum
    m_for_ck = {k: v for k, v in manifest.items() if k != "manifest_checksum"}
    computed = hashlib.sha256(
        json.dumps(m_for_ck, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    if computed != manifest["manifest_checksum"]:
        raise ValueError(
            f"Manifest checksum mismatch: stored={manifest['manifest_checksum']}, "
            f"computed={computed}. Manifest may have been tampered with."
        )

    # Contract version
    if manifest["contract_version"] not in ("1",):
        raise ValueError(f"Unknown contract_version {manifest['contract_version']!r}.")

    # Contract fingerprint
    expected_fp = canonical_fingerprint()
    if manifest["contract_fingerprint"] != expected_fp:
        raise ValueError(
            f"Manifest contract_fingerprint {manifest['contract_fingerprint']!r} "
            f"does not match canonical {expected_fp!r}. Refusing to ingest."
        )

    # Index name
    if manifest["index_name"] != CANONICAL_INDEX_NAME:
        raise ValueError(
            f"Manifest index_name {manifest['index_name']!r} != "
            f"canonical {CANONICAL_INDEX_NAME!r}."
        )

    # Namespace
    if manifest["index_namespace"] != CANONICAL_NAMESPACE:
        raise ValueError(
            f"Manifest index_namespace {manifest['index_namespace']!r} != "
            f"canonical {CANONICAL_NAMESPACE!r}."
        )

    # Embedded index contract
    m_contract = manifest["index_contract"]
    expected_contract = canonical_contract()
    missing_keys = set(expected_contract) - set(m_contract)
    extra_keys = set(m_contract) - set(expected_contract)
    if missing_keys:
        raise ValueError(f"Manifest index_contract missing keys: {sorted(missing_keys)}.")
    if extra_keys:
        raise ValueError(f"Manifest index_contract has unexpected keys: {sorted(extra_keys)}.")
    mismatches = [
        f"  {k}: expected {v!r}, got {m_contract.get(k)!r}"
        for k, v in expected_contract.items()
        if m_contract.get(k) != v
    ]
    if mismatches:
        raise ValueError(
            "Manifest index_contract differs from canonical:\n"
            + "\n".join(mismatches)
            + "\nRefusing to ingest."
        )

    # Forbidden field audit
    if manifest.get("forbidden_field_audit", "").startswith("FAIL"):
        raise ValueError(
            f"Manifest forbidden field audit failed: {manifest['forbidden_field_audit']}."
        )

    # Readiness
    if not manifest.get("ready_for_write"):
        failures = manifest.get("readiness_failures", ["unknown"])
        raise ValueError(
            f"Manifest is not ready_for_write. Failures: {failures}\n"
            f"Regenerate with:\n  {_REGENERATION_COMMAND}"
        )

    # Record counts
    total = manifest.get("total_records", 0)
    if total != CANARY_EXPECTED_TOTAL:
        raise ValueError(
            f"Manifest declares {total} total records; expected exactly {CANARY_EXPECTED_TOTAL} "
            f"for the canary. Refusing to proceed."
        )
    per_lang = manifest.get("actual_per_language_records", {})
    for lang in CANARY_EXPECTED_LANGUAGES:
        if per_lang.get(lang, 0) != CANARY_EXPECTED_PER_LANG:
            raise ValueError(
                f"Expected {CANARY_EXPECTED_PER_LANG} {lang} records; "
                f"got {per_lang.get(lang, 0)}."
            )

    return manifest


def _resolve_record_path(manifest: dict, manifest_path: Path) -> Path:
    stored = manifest["prepared_record_path"]
    candidate = Path(stored)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    sibling = manifest_path.resolve().parent / candidate.name
    if sibling.exists():
        return sibling
    full = manifest.get("prepared_record_path_full")
    if full and Path(full).exists():
        return Path(full)
    return sibling


def _verify_and_load_records(manifest: dict, manifest_path: Path) -> list[dict]:
    """Verify the JSONL data file and load all records."""
    record_path = _resolve_record_path(manifest, manifest_path)
    if not record_path.exists():
        raise FileNotFoundError(
            f"Prepared data file not found: {record_path}\n"
            f"Regenerate with:\n  {_REGENERATION_COMMAND}"
        )

    # Checksum
    computed = _sha256_file(record_path)
    stored = manifest.get("prepared_record_checksum", "")
    if stored and computed != stored:
        raise ValueError(
            f"Data file checksum mismatch: stored={stored}, computed={computed}. "
            f"File: {record_path}"
        )

    records = []
    with open(record_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            bad = FORBIDDEN_FIELDS & set(rec.keys())
            if bad:
                raise ValueError(f"Forbidden fields in record line {lineno}: {sorted(bad)}.")
            if not rec.get("id"):
                raise ValueError(f"Record line {lineno} has empty id.")
            if not rec.get("chunk_text"):
                raise ValueError(f"Record line {lineno} has empty chunk_text.")
            records.append(rec)

    if len(records) != CANARY_EXPECTED_TOTAL:
        raise ValueError(
            f"Record count mismatch: expected {CANARY_EXPECTED_TOTAL}, got {len(records)}."
        )
    return records


def _build_batches(records: list[dict], batch_size: int) -> list[list[dict]]:
    return [records[i : i + batch_size] for i in range(0, len(records), batch_size)]


# ── Checkpoint ────────────────────────────────────────────────────────────────


def _checkpoint_path(manifest_id: str, checkpoint_dir: Path, run_id: str) -> Path:
    return checkpoint_dir / f"canary_{manifest_id}_{run_id}.json"


def _save_checkpoint(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _load_checkpoint(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Could not read checkpoint %s: %s", path, e)
        return None


def _validate_checkpoint_compat(
    ckpt: dict,
    manifest_id: str,
    manifest_checksum: str,
    contract_fp: str,
    index_name: str,
    namespace: str,
    batch_size: int,
    batch_digests: list[str],
) -> None:
    """Raise if the checkpoint is incompatible with the current run configuration."""
    errors = []
    if ckpt.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        errors.append(
            f"checkpoint_schema_version {ckpt.get('checkpoint_schema_version')!r} "
            f"!= {CHECKPOINT_SCHEMA_VERSION!r}"
        )
    if ckpt.get("manifest_id") != manifest_id:
        errors.append(f"manifest_id {ckpt.get('manifest_id')!r} != {manifest_id!r}")
    if ckpt.get("manifest_checksum") != manifest_checksum:
        errors.append("manifest_checksum mismatch")
    if ckpt.get("contract_fingerprint") != contract_fp:
        errors.append("contract_fingerprint mismatch")
    if ckpt.get("index_name") != index_name:
        errors.append(f"index_name {ckpt.get('index_name')!r} != {index_name!r}")
    if ckpt.get("namespace") != namespace:
        errors.append(f"namespace {ckpt.get('namespace')!r} != {namespace!r}")
    if ckpt.get("batch_size") != batch_size:
        errors.append(f"batch_size {ckpt.get('batch_size')} != {batch_size}")
    # Batch digest list must match (same data, same ordering)
    if ckpt.get("batch_digests") != batch_digests:
        errors.append("batch_digests list mismatch — data or ordering changed")
    if errors:
        raise ValueError(
            "Checkpoint is incompatible with current run:\n"
            + "\n".join(f"  • {e}" for e in errors)
            + "\nDelete the checkpoint or fix the configuration."
        )


# ── Report generation ─────────────────────────────────────────────────────────


def _write_reports(
    report_dir: Path,
    run_id: str,
    data: dict,
) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"canary_index_execution_{run_id}.json"
    md_path = report_dir / f"canary_index_execution_{run_id}.md"

    # Write JSON
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    # Write Markdown
    lines = [
        f"# Canary Index Execution Report — {run_id}",
        "",
        f"**Status**: {data.get('status', 'unknown')}",
        f"**Git commit**: {data.get('git_commit', 'unknown')}",
        f"**Manifest ID**: {data.get('manifest_id', 'unknown')}",
        f"**Manifest checksum**: {data.get('manifest_checksum', 'unknown')}",
        f"**Contract fingerprint**: {data.get('contract_fingerprint', 'unknown')}",
        f"**Index**: {data.get('index_name', 'unknown')} / {data.get('namespace', 'unknown')}",
        "",
        "## Timing",
        f"- Start: {data.get('start_time', 'unknown')}",
        f"- End: {data.get('end_time', 'unknown')}",
        f"- Duration: {data.get('duration_seconds', 'unknown')}s",
        "",
        "## Throughput",
        f"- Records: {data.get('total_records', 0)}",
        f"- Tokens: {data.get('total_tokens', 0):,}",
        f"- Records/second: {data.get('records_per_second', 'unknown')}",
        f"- Tokens/minute: {data.get('tokens_per_minute', 'unknown')}",
        "",
        "## Batches",
        f"- Batch size: {data.get('batch_size', 'unknown')}",
        f"- Concurrency: {data.get('concurrency', 'unknown')}",
        f"- Total batches: {data.get('total_batches', 0)}",
        f"- Completed: {data.get('completed_batches', 0)}",
        f"- Skipped (resumed): {data.get('skipped_batches', 0)}",
        f"- Failed: {data.get('failed_batches', 0)}",
        f"- Total attempts: {data.get('total_attempts', 0)}",
        f"- Retries: {data.get('total_retries', 0)}",
        f"- Throttle waits (s): {data.get('total_throttle_wait_seconds', 0):.1f}",
        "",
        "## Validation",
        f"- Resume used: {data.get('resume_used', False)}",
        f"- Remote index validation: {data.get('remote_validation', 'not run')}",
        f"- Freshness reconciliation: {data.get('freshness_reconciliation', 'not run')}",
        "",
    ]
    if data.get("failure_category"):
        lines += [
            "## Failure",
            f"- Category: {data['failure_category']}",
            f"- Message: {data.get('failure_message', '')}",
            f"- Safe next action: {data.get('safe_next_action', 'Check logs and retry.')}",
            "",
        ]
    md_path.write_text("\n".join(lines), encoding="utf-8")

    return json_path, md_path


def _git_commit() -> str:
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


# ── Main logic ────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Resumable canary indexer for MSMARCO-XI → Pinecone",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--manifest", type=Path, required=True, help="Path to the _manifest.json")
    p.add_argument(
        "--execute",
        action="store_true",
        help="Perform live Pinecone writes (also requires CONFIRM_PINECONE_WRITE=1 and PINECONE_API_KEY)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing checkpoint. Incompatible checkpoints are rejected.",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Parallel upsert workers (default {DEFAULT_CONCURRENCY}, max {MAX_CONCURRENCY})",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=CANONICAL_MAX_BATCH_SIZE,
        help=f"Records per batch (max {CANONICAL_MAX_BATCH_SIZE})",
    )
    p.add_argument(
        "--token-rate-limit",
        type=int,
        default=DEFAULT_TOKEN_RATE_LIMIT,
        help=f"Max passage tokens/minute (default {DEFAULT_TOKEN_RATE_LIMIT}, hard limit 250000)",
    )
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("artifacts/checkpoints"),
        help="Directory for checkpoint files",
    )
    p.add_argument(
        "--report-dir",
        type=Path,
        default=Path("artifacts/reports"),
        help="Directory for execution reports",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    # Validate constraints
    if args.batch_size > CANONICAL_MAX_BATCH_SIZE:
        logger.error(
            "--batch-size %d exceeds canonical maximum %d.",
            args.batch_size,
            CANONICAL_MAX_BATCH_SIZE,
        )
        sys.exit(2)
    if args.concurrency > MAX_CONCURRENCY:
        logger.error("--concurrency %d exceeds maximum %d.", args.concurrency, MAX_CONCURRENCY)
        sys.exit(2)
    if args.token_rate_limit > 250_000:
        logger.error("--token-rate-limit exceeds Pinecone hard limit of 250,000 tokens/minute.")
        sys.exit(2)

    # --execute without confirmation must exit 2
    if args.execute and os.environ.get("CONFIRM_PINECONE_WRITE") != "1":
        logger.error(
            "--execute requires CONFIRM_PINECONE_WRITE=1. " "This prevents accidental live writes."
        )
        sys.exit(2)

    live_mode = args.execute and os.environ.get("CONFIRM_PINECONE_WRITE") == "1"

    # In dry-run mode: must never import Pinecone, must never read the API key.
    if not live_mode:
        logger.info("DRY-RUN MODE — no Pinecone client will be constructed.")

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    start_time = datetime.now(UTC).isoformat()
    git_commit = _git_commit()

    report_data: dict = {
        "run_id": run_id,
        "start_time": start_time,
        "git_commit": git_commit,
        "status": "started",
        "live_mode": live_mode,
    }

    json_report: Path | None = None
    md_report: Path | None = None

    try:
        _run(args, live_mode, run_id, start_time, git_commit, report_data)
    except SystemExit:
        raise
    except Exception as exc:
        logger.error("Fatal error: %s", exc, exc_info=True)
        report_data.update(
            {
                "status": "failed",
                "end_time": datetime.now(UTC).isoformat(),
                "failure_category": type(exc).__name__,
                "failure_message": str(exc),
                "safe_next_action": (
                    "Check the error above. If transient, retry with --resume. "
                    "If permanent, check credentials and index state."
                ),
            }
        )
        sys.exit(1)
    finally:
        report_data.setdefault("end_time", datetime.now(UTC).isoformat())
        start_dt = datetime.fromisoformat(start_time)
        end_dt = datetime.fromisoformat(report_data["end_time"])
        report_data["duration_seconds"] = round((end_dt - start_dt).total_seconds(), 2)
        try:
            json_report, md_report = _write_reports(args.report_dir, run_id, report_data)
            logger.info("Reports written: %s  %s", json_report, md_report)
        except Exception as re:
            logger.warning("Could not write reports: %s", re)


def _run(
    args: argparse.Namespace,
    live_mode: bool,
    run_id: str,
    start_time: str,
    git_commit: str,
    report_data: dict,
) -> None:
    manifest_path = args.manifest
    batch_size = args.batch_size
    concurrency = args.concurrency
    token_rate_limit = args.token_rate_limit
    checkpoint_dir = args.checkpoint_dir

    # ── Step 1: Load and validate manifest ───────────────────────────────────
    logger.info("Loading manifest: %s", manifest_path)
    manifest = _load_manifest(manifest_path)
    manifest_id = manifest["manifest_id"]
    manifest_checksum = manifest["manifest_checksum"]
    contract_fp = manifest["contract_fingerprint"]

    report_data.update(
        {
            "manifest_id": manifest_id,
            "manifest_checksum": manifest_checksum,
            "contract_fingerprint": contract_fp,
            "index_name": CANONICAL_INDEX_NAME,
            "namespace": CANONICAL_NAMESPACE,
            "batch_size": batch_size,
            "concurrency": concurrency,
            "total_tokens": manifest.get("total_tokens", 0),
            "resume_used": args.resume,
        }
    )

    # ── Step 2: Verify and load records ──────────────────────────────────────
    records = _verify_and_load_records(manifest, manifest_path)
    total_records = len(records)
    total_tokens = sum(r.get("token_length", 0) for r in records)
    report_data["total_records"] = total_records
    report_data["total_tokens"] = total_tokens

    # ── Step 3: Build batches ─────────────────────────────────────────────────
    batches = _build_batches(records, batch_size)
    total_batches = len(batches)
    batch_digests = [
        _batch_digest(manifest_checksum, i, [r["id"] for r in b]) for i, b in enumerate(batches)
    ]
    report_data["total_batches"] = total_batches

    # ── Step 4: Handle checkpoint / resume ────────────────────────────────────
    ckpt_path = _checkpoint_path(manifest_id, checkpoint_dir, run_id)
    completed_digests: set[str] = set()
    skipped_batches = 0

    if args.resume:
        # Look for any existing checkpoint for this manifest
        existing_ckpts = sorted(
            checkpoint_dir.glob(f"canary_{manifest_id}_*.json") if checkpoint_dir.exists() else []
        )
        if existing_ckpts:
            ckpt_path = existing_ckpts[-1]  # most recent
            ckpt = _load_checkpoint(ckpt_path)
            if ckpt is not None:
                _validate_checkpoint_compat(
                    ckpt,
                    manifest_id,
                    manifest_checksum,
                    contract_fp,
                    CANONICAL_INDEX_NAME,
                    CANONICAL_NAMESPACE,
                    batch_size,
                    batch_digests,
                )
                completed_digests = set(ckpt.get("completed_batch_digests", []))
                logger.info(
                    "Resuming from checkpoint %s — %d/%d batches already completed.",
                    ckpt_path,
                    len(completed_digests),
                    total_batches,
                )
                # Run ID carries over for checkpoint continuity
                run_id_for_ckpt = ckpt.get("run_id", run_id)
            else:
                run_id_for_ckpt = run_id
        else:
            logger.info("--resume set but no existing checkpoint found; starting fresh.")
            run_id_for_ckpt = run_id
    else:
        run_id_for_ckpt = run_id

    if not live_mode:
        # Dry-run: print plan and exit.
        pending = sum(1 for d in batch_digests if d not in completed_digests)
        print("\n" + "=" * 60)
        print("CANARY INDEX DRY-RUN PLAN (no Pinecone calls)")
        print("=" * 60)
        print(f"  Manifest ID         : {manifest_id}")
        print(f"  Index               : {CANONICAL_INDEX_NAME} / {CANONICAL_NAMESPACE}")
        print(f"  Records             : {total_records}")
        print(f"  Total tokens        : {total_tokens:,}")
        print(f"  Batch size          : {batch_size}")
        print(f"  Concurrency         : {concurrency}")
        print(f"  Total batches       : {total_batches}")
        print(f"  Already completed   : {len(completed_digests)}")
        print(f"  Pending             : {pending}")
        print(f"  Token rate ceiling  : {token_rate_limit:,} tokens/min")
        print("=" * 60)
        print("\nDRY-RUN complete — no records were written.\n")
        report_data.update(
            {
                "status": "dry_run_complete",
                "total_batches": total_batches,
                "completed_batches": len(completed_digests),
                "skipped_batches": len(completed_digests),
                "pending_batches": pending,
                "failed_batches": 0,
            }
        )
        return

    # ── Step 5: Live mode — validate remote index ─────────────────────────────
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        logger.error("PINECONE_API_KEY is not set.")
        sys.exit(1)

    # Import Pinecone only in live mode.
    from pinecone import Pinecone

    from hhgoa_rag.pinecone_lifecycle import validate_index
    from hhgoa_rag.pinecone_store import PineconeStore

    logger.info("Connecting to Pinecone …")
    pc = Pinecone(api_key=api_key)

    logger.info("Validating remote index '%s' …", CANONICAL_INDEX_NAME)
    validation_errors = validate_index(pc, CANONICAL_INDEX_NAME)
    if validation_errors:
        logger.error(
            "Remote index validation failed:\n%s",
            "\n".join(f"  • {e}" for e in validation_errors),
        )
        report_data["remote_validation"] = f"FAILED: {validation_errors}"
        sys.exit(1)
    logger.info("Remote index validation PASSED.")
    report_data["remote_validation"] = "PASSED"

    index = pc.Index(CANONICAL_INDEX_NAME)
    from hhgoa_rag.pinecone_contract import MODEL

    store = PineconeStore(index, embed_model=MODEL)

    # ── Step 6: Token-paced parallel upserts ─────────────────────────────────
    # Initialize checkpoint for this run.
    ckpt_data: dict = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "run_id": run_id_for_ckpt,
        "manifest_id": manifest_id,
        "manifest_checksum": manifest_checksum,
        "contract_fingerprint": contract_fp,
        "index_name": CANONICAL_INDEX_NAME,
        "namespace": CANONICAL_NAMESPACE,
        "batch_size": batch_size,
        "batch_digests": batch_digests,
        "completed_batch_digests": list(completed_digests),
        "total_batches": total_batches,
        "attempts": 0,
        "retries": 0,
        "started_at": start_time,
        "updated_at": start_time,
    }
    _save_checkpoint(ckpt_path, ckpt_data)

    completed_batches = 0
    failed_batches = 0
    total_attempts = 0
    total_retries = 0
    total_throttle_wait = 0.0
    tokens_submitted = 0
    window_start = time.monotonic()
    window_tokens = 0

    def _upsert_batch(
        batch_idx: int, batch: list[dict], digest: str
    ) -> tuple[int, int, int, float]:
        """Submit one batch with retry. Returns (submitted, attempts, retries, throttle_wait)."""
        nonlocal window_start, window_tokens
        attempts = 0
        retries = 0
        throttle_wait = 0.0
        last_exc: Exception | None = None

        batch_tokens = sum(r.get("token_length", 0) for r in batch)

        for attempt in range(1, MAX_RETRIES + 1):
            # Token-rate pacing: check if we need to slow down.
            elapsed = time.monotonic() - window_start
            if elapsed >= 60.0:
                window_start = time.monotonic()
                window_tokens = 0
                elapsed = 0.0
            projected = window_tokens + batch_tokens
            if projected > token_rate_limit and elapsed < 60.0:
                wait = 60.0 - elapsed + 0.5
                logger.info(
                    "Token rate limit: projected %d > %d tokens/min; waiting %.1fs",
                    projected,
                    token_rate_limit,
                    wait,
                )
                time.sleep(wait)
                throttle_wait += wait
                window_start = time.monotonic()
                window_tokens = 0

            try:
                attempts += 1
                logger.info(
                    "Upserting batch %d/%d (%d records, %d tokens) attempt %d/%d …",
                    batch_idx + 1,
                    total_batches,
                    len(batch),
                    batch_tokens,
                    attempt,
                    MAX_RETRIES,
                )
                submitted = store.upsert_records(
                    batch, namespace=CANONICAL_NAMESPACE, context="pilot"
                )
                window_tokens += batch_tokens
                return submitted, attempts, retries, throttle_wait
            except Exception as exc:
                last_exc = exc
                if not _is_transient_error(exc):
                    logger.error("Permanent error on batch %d: %s", batch_idx + 1, exc)
                    raise

                retries += 1
                if attempt >= MAX_RETRIES:
                    break

                # Respect Retry-After if present
                retry_after = _retry_after(exc)
                if retry_after is not None:
                    delay = min(retry_after, RETRY_MAX_DELAY)
                else:
                    delay = min(RETRY_BASE_DELAY * (2 ** (attempt - 1)), RETRY_MAX_DELAY)
                # Add jitter
                import random

                delay = delay * (0.8 + 0.4 * random.random())
                logger.warning(
                    "Transient error on batch %d attempt %d/%d: %s — retrying in %.1fs",
                    batch_idx + 1,
                    attempt,
                    MAX_RETRIES,
                    exc,
                    delay,
                )
                time.sleep(delay)
                throttle_wait += delay

        if last_exc is not None:
            raise last_exc
        return 0, attempts, retries, throttle_wait

    pending_indices = [i for i, d in enumerate(batch_digests) if d not in completed_digests]
    skipped_batches = total_batches - len(pending_indices)

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures: dict[concurrent.futures.Future, tuple[int, str]] = {}

        for batch_idx in pending_indices:
            batch = batches[batch_idx]
            digest = batch_digests[batch_idx]
            future = executor.submit(_upsert_batch, batch_idx, batch, digest)
            futures[future] = (batch_idx, digest)

        for future in concurrent.futures.as_completed(futures):
            batch_idx, digest = futures[future]
            try:
                submitted, attempts, retries, tw = future.result()
                completed_batches += 1
                total_attempts += attempts
                total_retries += retries
                total_throttle_wait += tw
                tokens_submitted += sum(r.get("token_length", 0) for r in batches[batch_idx])
                completed_digests.add(digest)

                # Atomic checkpoint after each acknowledged batch.
                ckpt_data["completed_batch_digests"] = list(completed_digests)
                ckpt_data["attempts"] = total_attempts
                ckpt_data["retries"] = total_retries
                ckpt_data["updated_at"] = datetime.now(UTC).isoformat()
                _save_checkpoint(ckpt_path, ckpt_data)
                logger.info(
                    "Batch %d/%d committed. Checkpoint saved.",
                    completed_batches + skipped_batches,
                    total_batches,
                )
            except Exception as exc:
                failed_batches += 1
                logger.error("Batch %d failed permanently: %s", batch_idx + 1, exc)

    if failed_batches > 0:
        logger.error("%d batch(es) failed. Run with --resume to retry.", failed_batches)
        report_data.update(
            {
                "status": "failed",
                "failure_category": "UpsertError",
                "failure_message": f"{failed_batches} batch(es) failed permanently.",
                "safe_next_action": "Run with --resume to retry failed batches.",
            }
        )
        sys.exit(1)

    # ── Step 7: Freshness reconciliation ─────────────────────────────────────
    logger.info("Waiting for index freshness (max %ds) …", FRESHNESS_POLL_MAX_WAIT)
    deadline = time.monotonic() + FRESHNESS_POLL_MAX_WAIT
    wait = FRESHNESS_POLL_BASE
    final_count: int | None = None
    reconciled = False

    while time.monotonic() < deadline:
        try:
            stats = index.describe_index_stats()
            ns_stats = (stats.namespaces or {}).get(CANONICAL_NAMESPACE)
            count = ns_stats.vector_count if ns_stats else 0
            logger.info("Namespace '%s' vector count: %d", CANONICAL_NAMESPACE, count)
            if count >= CANARY_EXPECTED_TOTAL:
                final_count = count
                reconciled = True
                break
        except Exception as e:
            logger.warning("Freshness poll error: %s", e)
        time.sleep(min(wait, deadline - time.monotonic()))
        wait = min(wait * 2, 30)

    end_time = datetime.now(UTC).isoformat()
    duration = (
        datetime.fromisoformat(end_time) - datetime.fromisoformat(start_time)
    ).total_seconds()

    if not reconciled:
        logger.warning(
            "Freshness reconciliation timeout after %ds. Count=%s (expected %d).",
            FRESHNESS_POLL_MAX_WAIT,
            final_count,
            CANARY_EXPECTED_TOTAL,
        )
        freshness_result = f"TIMEOUT — count={final_count}, expected={CANARY_EXPECTED_TOTAL}"
    else:
        freshness_result = f"PASS — count={final_count}"
        logger.info("Freshness reconciliation PASSED: count=%d", final_count)

    rps = round(total_records / max(duration, 0.001), 2) if duration > 0 else 0
    tpm = round(tokens_submitted / max(duration / 60, 0.001), 0) if duration > 0 else 0

    report_data.update(
        {
            "status": "success" if reconciled else "partial_success",
            "end_time": end_time,
            "total_batches": total_batches,
            "completed_batches": completed_batches,
            "skipped_batches": skipped_batches,
            "failed_batches": failed_batches,
            "total_attempts": total_attempts,
            "total_retries": total_retries,
            "total_throttle_wait_seconds": round(total_throttle_wait, 2),
            "records_per_second": rps,
            "tokens_per_minute": tpm,
            "freshness_reconciliation": freshness_result,
        }
    )

    if not reconciled:
        sys.exit(1)

    logger.info(
        "Canary indexing complete: %d records, %d tokens in %.1fs (%.1f r/s, %d t/min)",
        total_records,
        total_tokens,
        duration,
        rps,
        tpm,
    )


if __name__ == "__main__":
    main()
