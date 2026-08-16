#!/usr/bin/env python3
"""Resumable one-command canary/pilot indexer for MSMARCO-XI → Pinecone.

Default mode: offline dry-run (validates everything, schedules no writes).

Live mode requires ALL of:
  --execute
  CONFIRM_PINECONE_WRITE=1 environment variable
  PINECONE_API_KEY environment variable

Scopes:
  --scope canary-300   (default) 300 records: 100 EN / 100 HI / 100 BN
  --scope pilot-10000  10,000 records: 3334 EN / 3333 HI / 3333 BN

Usage:
  # Dry run (safe, no credentials needed):
  uv run python scripts/index_canary.py --manifest artifacts/prepared/<id>_manifest.json

  # Live canary write:
  CONFIRM_PINECONE_WRITE=1 PINECONE_API_KEY=<key> \\
    uv run python scripts/index_canary.py \\
      --manifest artifacts/prepared/<id>_manifest.json \\
      --execute --resume --concurrency 4

  # Live pilot-10000 write:
  CONFIRM_PINECONE_WRITE=1 PINECONE_API_KEY=<key> \\
    uv run python scripts/index_canary.py \\
      --scope pilot-10000 \\
      --manifest artifacts/prepared/<id>_manifest.json \\
      --execute --resume --concurrency 4

  # Resume after interruption:
  CONFIRM_PINECONE_WRITE=1 PINECONE_API_KEY=<key> \\
    uv run python scripts/index_canary.py \\
      --manifest artifacts/prepared/<id>_manifest.json \\
      --execute --resume

  # Background (nohup) — use `env` so the variables reach the detached process:
  nohup env \\
    CONFIRM_PINECONE_WRITE=1 \\
    PINECONE_API_KEY="$PINECONE_API_KEY" \\
    uv run python scripts/index_canary.py \\
    --manifest <manifest-path> \\
    --execute \\
    > artifacts/logs/index_canary.log 2>&1 &

Hard limits:
  - Scope is fixed: canary-300 or pilot-10000. No arbitrary totals.
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
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from hhgoa_rag.pinecone_contract import (
    DATASET_REPO as CANONICAL_DATASET_REPO,
)
from hhgoa_rag.pinecone_contract import (
    DATASET_REVISION as CANONICAL_DATASET_REVISION,
)
from hhgoa_rag.pinecone_contract import (
    INDEX_NAME as CANONICAL_INDEX_NAME,
)
from hhgoa_rag.pinecone_contract import (
    MANIFEST_SCHEMA_VERSION,
    canonical_contract,
)
from hhgoa_rag.pinecone_contract import (
    MAX_BATCH_SIZE as CANONICAL_MAX_BATCH_SIZE,
)
from hhgoa_rag.pinecone_contract import (
    MAX_INPUT_TOKENS as CANONICAL_MAX_INPUT_TOKENS,
)
from hhgoa_rag.pinecone_contract import (
    NAMESPACE as CANONICAL_NAMESPACE,
)
from hhgoa_rag.pinecone_contract import (
    TOKENIZER_REPO as CANONICAL_TOKENIZER_REPO,
)
from hhgoa_rag.pinecone_contract import (
    TOKENIZER_REVISION as CANONICAL_TOKENIZER_REVISION,
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

# ── Scope configuration ───────────────────────────────────────────────────────

SCOPE_CANARY_300 = "canary-300"
SCOPE_PILOT_10000 = "pilot-10000"
VALID_SCOPES = (SCOPE_CANARY_300, SCOPE_PILOT_10000)

# Fixed scope expectations — no arbitrary totals permitted.
_SCOPE_EXPECTED: dict[str, dict] = {
    SCOPE_CANARY_300: {
        "total": 300,
        "per_lang": {"en": 100, "hi": 100, "bn": 100},
    },
    SCOPE_PILOT_10000: {
        "total": 10_000,
        "per_lang": {"en": 3334, "hi": 3333, "bn": 3333},
    },
}

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
    # Provenance fields
    "dataset_revision",
    "dataset_repo",
    "tokenizer_repo",
    "tokenizer_revision",
    "tokenizer_fingerprint",
    "model_input_limit",
    # Data
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


class CanaryError(Exception):
    """Raised for expected fatal errors so main() can set status:failed before writing reports."""

    def __init__(self, message: str, category: str, safe_next_action: str = "") -> None:
        super().__init__(message)
        self.category = category
        self.safe_next_action = safe_next_action or (
            "Check the error above. If transient, retry with --resume. "
            "If permanent, check credentials and index state."
        )


_REGENERATION_COMMAND = (
    "uv run python scripts/prepare_canary.py "
    "--dataset-revision bf5cdc1f26e581e519018e434db14edd1b77602b "
    "--tokenizer-revision 3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3 "
    "--seed 42"
)


# ── Token-rate limiter ────────────────────────────────────────────────────────


class _TokenRateLimiter:
    """Thread-safe rolling sliding-window token-rate limiter.

    Maintains a timestamped queue of reservations within the rolling interval
    `[now - window_seconds, now]`. Any reservation that would cause the sum
    of reserved tokens in any rolling `window_seconds` interval to exceed
    `tokens_per_window` is delayed until older reservations slide out of the window.

    Check-and-reserve decisions are executed atomically under the lock.
    Sleeps are performed outside the lock so other threads can proceed.

    Parameters
    ----------
    tokens_per_window:
        Maximum tokens that may be reserved within any rolling window (must be > 0).
    window_seconds:
        Duration of the rolling window in seconds. Defaults to 60.0.
    clock:
        Injectable monotonic clock callable (default: ``time.monotonic``).
        Tests may pass a deterministic fake clock.
    sleeper:
        Injectable sleep callable that accepts a ``float`` number of seconds
        (default: ``time.sleep``). Tests may pass a recording fake sleeper.
    """

    def __init__(
        self,
        tokens_per_window: int,
        window_seconds: float = 60.0,
        clock: Callable[[], float] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        if tokens_per_window <= 0:
            raise ValueError(f"tokens_per_window must be positive, got {tokens_per_window}")
        if window_seconds <= 0:
            raise ValueError(f"window_seconds must be positive, got {window_seconds}")
        self._tokens_per_window = tokens_per_window
        self._window_seconds = window_seconds
        self._clock: Callable[[], float] = clock if clock is not None else time.monotonic
        self._sleeper: Callable[[float], None] = sleeper if sleeper is not None else time.sleep
        self._lock = threading.Lock()
        self._reservations: deque[tuple[float, int]] = deque()
        self._current_tokens: int = 0

    def _prune(self, now: float) -> None:
        cutoff = now - self._window_seconds
        while self._reservations and self._reservations[0][0] <= cutoff:
            _, tokens = self._reservations.popleft()
            self._current_tokens -= tokens

    def acquire(self, token_count: int) -> float:
        """Block until ``token_count`` tokens can be reserved in the current rolling window.

        Parameters
        ----------
        token_count:
            Number of tokens to reserve. Must be > 0 and <= tokens_per_window.

        Returns
        -------
        float
            Cumulative seconds spent waiting by this call (0.0 when no wait was needed).

        Raises
        ------
        ValueError
            If ``token_count`` is <= 0 or exceeds ``tokens_per_window``.
        """
        if token_count <= 0:
            raise ValueError(f"token_count must be positive, got {token_count}")
        if token_count > self._tokens_per_window:
            raise ValueError(
                f"token_count {token_count} exceeds tokens_per_window {self._tokens_per_window}"
            )

        total_waited = 0.0
        while True:
            wait = 0.0
            with self._lock:
                now = self._clock()
                self._prune(now)

                if self._current_tokens + token_count <= self._tokens_per_window:
                    self._reservations.append((now, token_count))
                    self._current_tokens += token_count
                    return total_waited

                # Calculate required sleep until enough tokens slide out of the window
                needed_freed = (self._current_tokens + token_count) - self._tokens_per_window
                freed = 0
                earliest_expiring_ts = now
                for ts, tok in self._reservations:
                    freed += tok
                    earliest_expiring_ts = ts
                    if freed >= needed_freed:
                        break

                wait = max((earliest_expiring_ts + self._window_seconds) - now, 0.001)

            # Sleep outside the lock so other threads can proceed
            self._sleeper(wait)
            total_waited += wait


# ── Helpers ───────────────────────────────────────────────────────────────────


def _get_ns_vector_count(stats: object, namespace: str = CANONICAL_NAMESPACE) -> int | None:
    """Extract namespace vector count from SDK object or dict-shaped fixture.

    Returns:
        int: Non-negative vector count if verified.
        None: If stats or namespace vector count is missing, malformed, or unverifiable.
    """
    if stats is None:
        return None
    namespaces = getattr(stats, "namespaces", None)
    if namespaces is None and isinstance(stats, dict):
        namespaces = stats.get("namespaces")
    if namespaces is None or not isinstance(namespaces, dict):
        return None

    if namespace not in namespaces:
        # In Pinecone describe_index_stats, an existing empty namespace is omitted
        # from the namespaces dictionary when it has 0 vectors.
        return 0

    ns_info = namespaces[namespace]
    if ns_info is None:
        return None

    vc = getattr(ns_info, "vector_count", None)
    if vc is None and isinstance(ns_info, dict):
        vc = ns_info.get("vector_count")

    # Strict type acceptance: only a genuine, non-negative Python ``int`` is a
    # verifiable count. ``bool`` (subclass of int), floats (300.0, 300.9),
    # numeric strings ("300"), None, and negatives are unverifiable and must
    # NOT be coerced — coercion would silently accept ``300.9 -> 300`` or
    # ``"300" -> 300`` and let a malformed value pass preflight / polling /
    # reconciliation gates. Return None (the "unverifiable" result) instead.
    if isinstance(vc, bool) or not isinstance(vc, int):
        return None
    return vc if vc >= 0 else None


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


def _load_manifest(manifest_path: Path, scope: str = SCOPE_CANARY_300) -> dict:
    """Load and strictly validate the canary/pilot manifest. Raises on any failure."""
    if scope not in VALID_SCOPES:
        raise ValueError(f"Unknown scope {scope!r}. Must be one of {VALID_SCOPES}.")
    _expected_total = _SCOPE_EXPECTED[scope]["total"]
    _expected_per_lang: dict[str, int] = _SCOPE_EXPECTED[scope]["per_lang"]
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

    # Record counts — validated against scope expectations
    total = manifest.get("total_records", 0)
    if total != _expected_total:
        raise ValueError(
            f"Manifest declares {total} total records; expected exactly {_expected_total} "
            f"for scope '{scope}'. Refusing to proceed."
        )
    per_lang = manifest.get("actual_per_language_records", {})
    for lang, expected_count in _expected_per_lang.items():
        if per_lang.get(lang, 0) != expected_count:
            raise ValueError(
                f"Expected {expected_count} {lang} records for scope '{scope}'; "
                f"got {per_lang.get(lang, 0)}."
            )

    # Provenance fields — must match the canonical contract
    if manifest.get("dataset_repo") != CANONICAL_DATASET_REPO:
        raise ValueError(
            f"Manifest dataset_repo {manifest.get('dataset_repo')!r} != "
            f"canonical {CANONICAL_DATASET_REPO!r}."
        )
    if manifest.get("dataset_revision") != CANONICAL_DATASET_REVISION:
        raise ValueError(
            f"Manifest dataset_revision {manifest.get('dataset_revision')!r} != "
            f"canonical {CANONICAL_DATASET_REVISION!r}."
        )
    if manifest.get("tokenizer_repo") != CANONICAL_TOKENIZER_REPO:
        raise ValueError(
            f"Manifest tokenizer_repo {manifest.get('tokenizer_repo')!r} != "
            f"canonical {CANONICAL_TOKENIZER_REPO!r}."
        )
    if manifest.get("tokenizer_revision") != CANONICAL_TOKENIZER_REVISION:
        raise ValueError(
            f"Manifest tokenizer_revision {manifest.get('tokenizer_revision')!r} != "
            f"canonical {CANONICAL_TOKENIZER_REVISION!r}."
        )
    if manifest.get("model_input_limit") != CANONICAL_MAX_INPUT_TOKENS:
        raise ValueError(
            f"Manifest model_input_limit {manifest.get('model_input_limit')!r} != "
            f"canonical {CANONICAL_MAX_INPUT_TOKENS!r}."
        )
    if not manifest.get("tokenizer_fingerprint"):
        raise ValueError("Manifest missing non-empty tokenizer_fingerprint.")

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


def _verify_and_load_records(
    manifest: dict, manifest_path: Path, scope: str = SCOPE_CANARY_300
) -> list[dict]:
    """Verify the JSONL data file and load all records.

    Calls validate_record() from the authoritative schema on EVERY record.
    This must complete before any Pinecone client is constructed.
    """
    from hhgoa_rag.ingestion.schema import SchemaViolationError, validate_record

    _expected_total = _SCOPE_EXPECTED[scope]["total"]
    _expected_per_lang: dict[str, int] = _SCOPE_EXPECTED[scope]["per_lang"]

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
    seen_ids: set[str] = set()
    with open(record_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # Authoritative schema validation — runs BEFORE any provider client creation.
            try:
                validate_record(rec)
            except SchemaViolationError as e:
                raise ValueError(f"Record line {lineno} schema violation: {e}") from e
            # Duplicate ID check
            rec_id = rec["id"]
            if rec_id in seen_ids:
                raise ValueError(f"Duplicate record id {rec_id!r} at line {lineno}.")
            seen_ids.add(rec_id)
            records.append(rec)

    if len(records) != _expected_total:
        raise ValueError(
            f"Record count mismatch: expected {_expected_total} for scope '{scope}', "
            f"got {len(records)}."
        )

    # Verify actual language counts match manifest's declared counts.
    actual_lang_counts: dict[str, int] = {}
    for rec in records:
        lang = rec.get("language", "")
        actual_lang_counts[lang] = actual_lang_counts.get(lang, 0) + 1
    declared_lang_counts = manifest.get("actual_per_language_records", {})
    for lang, expected_count in _expected_per_lang.items():
        actual = actual_lang_counts.get(lang, 0)
        declared = declared_lang_counts.get(lang, 0)
        if actual != declared:
            raise ValueError(
                f"Actual JSONL language count for '{lang}' is {actual} "
                f"but manifest declares {declared}."
            )
        if actual != expected_count:
            raise ValueError(
                f"Actual JSONL language count for '{lang}' is {actual} "
                f"but scope '{scope}' expects {expected_count}."
            )

    # Verify token total matches manifest.
    actual_token_total = sum(r.get("token_length", 0) for r in records)
    manifest_token_total = manifest.get("total_tokens", -1)
    if actual_token_total != manifest_token_total:
        raise ValueError(
            f"JSONL token total {actual_token_total} does not match manifest "
            f"total_tokens {manifest_token_total}."
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


class _CorruptCheckpointError(Exception):
    """Raised when a checkpoint file exists but cannot be read or parsed."""


def _load_checkpoint(path: Path) -> dict | None:
    """Load checkpoint file.

    Returns:
        None if path does not exist (fresh start is OK).
        dict if the file is valid JSON.

    Raises:
        _CorruptCheckpointError if the file exists but is unreadable or malformed.
        Never silently treats a corrupt checkpoint as absent.
    """
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Checkpoint is not a JSON object (got {type(data).__name__})")
        return data
    except Exception as e:
        raise _CorruptCheckpointError(
            f"Checkpoint file {path} exists but cannot be read or parsed: {e}. "
            "Delete or repair the checkpoint file before retrying."
        ) from e


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
        f"- Count reconciliation: {data.get('count_reconciliation', 'not run')}",
        f"- Exact-ID reconciliation: {data.get('exact_id_reconciliation', 'not run')}",
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
    p.add_argument(
        "--scope",
        choices=list(VALID_SCOPES),
        default=SCOPE_CANARY_300,
        help=f"Fixed indexing scope (default: {SCOPE_CANARY_300}). "
        f"'{SCOPE_CANARY_300}' = 300 records 100/100/100. "
        f"'{SCOPE_PILOT_10000}' = 10,000 records 3334/3333/3333.",
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

    # Validate zero/negative values — exit 2
    if args.batch_size <= 0:
        logger.error("--batch-size must be a positive integer, got %d.", args.batch_size)
        sys.exit(2)
    if args.concurrency <= 0:
        logger.error("--concurrency must be a positive integer, got %d.", args.concurrency)
        sys.exit(2)
    if args.token_rate_limit <= 0:
        logger.error("--token-rate-limit must be positive, got %d.", args.token_rate_limit)
        sys.exit(2)

    # Validate maximums
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

    _exit_code = 0
    try:
        _run(args, live_mode, run_id, start_time, git_commit, report_data)
    except SystemExit as se:
        # sys.exit() from inside _run — convert to CanaryError so reports are written correctly.
        _exit_code = se.code if isinstance(se.code, int) else 1
        if report_data.get("status") in ("started", None):
            report_data.update(
                {
                    "status": "failed",
                    "end_time": datetime.now(UTC).isoformat(),
                    "failure_category": "FatalExit",
                    "failure_message": f"Process exited with code {_exit_code} — see logs above.",
                    "safe_next_action": (
                        "Check the error above. If transient, retry with --resume. "
                        "If permanent, check credentials and index state."
                    ),
                }
            )
    except CanaryError as exc:
        logger.error("Fatal error [%s]: %s", exc.category, exc, exc_info=False)
        report_data.update(
            {
                "status": "failed",
                "end_time": datetime.now(UTC).isoformat(),
                "failure_category": exc.category,
                "failure_message": str(exc),
                "safe_next_action": exc.safe_next_action,
            }
        )
        _exit_code = 1
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
        _exit_code = 1
    finally:
        # Ensure status never stays "started" — always has a terminal value.
        if report_data.get("status") == "started":
            report_data["status"] = "failed"
            report_data.setdefault(
                "failure_message", "Run terminated without setting final status."
            )
        report_data.setdefault("end_time", datetime.now(UTC).isoformat())
        start_dt = datetime.fromisoformat(start_time)
        end_dt = datetime.fromisoformat(report_data["end_time"])
        report_data["duration_seconds"] = round((end_dt - start_dt).total_seconds(), 2)
        try:
            json_report, md_report = _write_reports(args.report_dir, run_id, report_data)
            logger.info("Reports written: %s  %s", json_report, md_report)
        except Exception as re:
            logger.warning("Could not write reports: %s", re)
    if _exit_code:
        sys.exit(_exit_code)


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
    scope = args.scope
    expected_total = _SCOPE_EXPECTED[scope]["total"]

    # ── Step 1: Load and validate manifest ───────────────────────────────────
    logger.info("Loading manifest: %s", manifest_path)
    manifest = _load_manifest(manifest_path, scope=scope)
    manifest_id = manifest["manifest_id"]
    manifest_checksum = manifest["manifest_checksum"]
    contract_fp = manifest["contract_fingerprint"]

    report_data.update(
        {
            "scope": scope,
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
    records = _verify_and_load_records(manifest, manifest_path, scope=scope)
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
            # _load_checkpoint raises _CorruptCheckpointError on malformed files — fail closed.
            try:
                ckpt = _load_checkpoint(ckpt_path)
            except _CorruptCheckpointError as e:
                raise CanaryError(
                    str(e),
                    category="CorruptCheckpoint",
                    safe_next_action=(
                        f"Delete or repair {ckpt_path} before retrying. "
                        "Starting fresh without --resume is also safe."
                    ),
                ) from e
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
    # Whitespace-aware credential validation. Absent, empty, or whitespace-only
    # values fail closed BEFORE the Pinecone SDK is imported. Never log the value.
    api_key = os.environ.get("PINECONE_API_KEY", "").strip()
    if not api_key:
        raise CanaryError(
            "PINECONE_API_KEY is not set. Export it before running with --execute.",
            category="MissingAPIKey",
            safe_next_action="Export PINECONE_API_KEY and retry.",
        )

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
        raise CanaryError(
            f"Remote index validation failed: {validation_errors}",
            category="RemoteValidationFailure",
            safe_next_action="Verify the Pinecone index matches the canonical contract and retry.",
        )
    logger.info("Remote index validation PASSED.")
    report_data["remote_validation"] = "PASSED"

    index = pc.Index(CANONICAL_INDEX_NAME)
    from hhgoa_rag.pinecone_contract import MODEL

    store = PineconeStore(index, embed_model=MODEL)

    # ── Step 5.5: Pre-write namespace preflight & resume ownership verification (fail-closed) ──
    logger.info("Performing pre-write preflight check on namespace '%s' …", CANONICAL_NAMESPACE)
    preflight_stats: object = None
    try:
        preflight_stats = index.describe_index_stats()
    except Exception as exc:
        logger.error("Pre-write namespace preflight provider call failed: %s", exc)
        report_data["preflight_status"] = f"FAILED: Provider error {exc}"
        raise CanaryError(
            f"Pre-write namespace preflight failed due to provider error: {exc}",
            category="PreflightProviderFailure",
            safe_next_action="Check network connectivity, API credentials, and Pinecone service status before retrying.",
        ) from exc

    preflight_count = _get_ns_vector_count(preflight_stats, CANONICAL_NAMESPACE)
    if preflight_count is None:
        msg = (
            f"Namespace '{CANONICAL_NAMESPACE}' preflight unverifiable: "
            "Pinecone index statistics are missing, malformed, ambiguous, or cannot be parsed. "
            "Cannot verify whether the target namespace is empty or valid."
        )
        logger.error(msg)
        report_data["preflight_status"] = "FAILED: Preflight unverifiable"
        raise CanaryError(
            msg,
            category="PreflightUnverifiable",
            safe_next_action="Verify Pinecone index statistics and connectivity before retrying.",
        )

    is_resume_run = bool(args.resume and completed_digests)
    expected_id_set_all = {r["id"] for r in records}

    if scope == SCOPE_PILOT_10000:
        # ── Pilot-10000 preflight: ownership verification ────────────────────
        # All current namespace IDs MUST belong to the pilot-10000 expected set.
        # Re-upserting existing canary vectors is acceptable; any unrelated ID fails.
        if preflight_count > expected_total:
            msg = (
                f"Namespace '{CANONICAL_NAMESPACE}' contains {preflight_count} vectors, "
                f"which exceeds the pilot-10000 expected total of {expected_total}. "
                "Namespace is contaminated or stale."
            )
            logger.error(msg)
            report_data["preflight_status"] = (
                f"FAILED: Contaminated ({preflight_count} vectors > {expected_total})"
            )
            raise CanaryError(
                msg,
                category="NamespaceContaminatedPreflight",
                safe_next_action="Verify index contents. Stale vectors must be manually cleared by the operator.",
            )

        # Enumerate current IDs and verify all ⊆ pilot expected set.
        try:
            current_namespace_ids = store.list_vector_ids(namespace=CANONICAL_NAMESPACE)
        except Exception as exc:
            logger.error("Pilot preflight ID enumeration failed: %s", exc)
            report_data["preflight_status"] = f"FAILED: ID enumeration error {exc}"
            raise CanaryError(
                f"Pilot preflight ID enumeration failed: {exc}",
                category="PreflightProviderFailure",
                safe_next_action="Verify Pinecone ID enumeration support and connectivity before running.",
            ) from exc

        current_id_set = set(current_namespace_ids)
        enumerated_count = len(current_namespace_ids)

        # Detect duplicate IDs from enumeration
        if enumerated_count != len(current_id_set):
            msg = (
                f"Pilot preflight: enumeration returned {enumerated_count} IDs but only "
                f"{len(current_id_set)} are unique — possible pagination ambiguity or duplicate IDs."
            )
            logger.error(msg)
            report_data["preflight_status"] = "FAILED: Duplicate IDs in enumeration"
            raise CanaryError(msg, category="PreflightEnumerationAmbiguous")

        # Verify stats count matches enumeration count
        if preflight_count != enumerated_count:
            msg = (
                f"Pilot preflight: stats reports {preflight_count} vectors but enumeration "
                f"returned {enumerated_count} IDs — possible pagination ambiguity."
            )
            logger.error(msg)
            report_data["preflight_status"] = "FAILED: Stats/enumeration count mismatch"
            raise CanaryError(msg, category="PreflightEnumerationAmbiguous")

        # Every current ID must belong to the pilot expected set.
        unrelated_ids = current_id_set - expected_id_set_all
        if unrelated_ids:
            msg = (
                f"Pilot preflight FAILED: namespace '{CANONICAL_NAMESPACE}' contains "
                f"{len(unrelated_ids)} ID(s) that are NOT in the pilot-10000 expected set. "
                "Refusing to write. Do not delete or clear automatically."
            )
            logger.error(msg)
            report_data["preflight_status"] = (
                f"FAILED: {len(unrelated_ids)} unrelated ID(s) in namespace"
            )
            raise CanaryError(
                msg,
                category="PilotOwnershipFailure",
                safe_next_action=(
                    "Verify namespace contents. All existing IDs must belong to the "
                    "pilot-10000 manifest before proceeding."
                ),
            )

        logger.info(
            "Pilot preflight PASSED: namespace '%s' has %d vectors, all belong to the "
            "pilot-10000 expected set (expected_total=%d, is_resume=%s).",
            CANONICAL_NAMESPACE,
            preflight_count,
            expected_total,
            is_resume_run,
        )
        report_data["preflight_status"] = (
            f"PASSED (pilot-10000: {preflight_count} existing vectors all ⊆ expected set)"
        )

    else:
        # ── Canary-300 preflight (original logic, unchanged) ─────────────────
        if not is_resume_run:
            # Fresh canary run: namespace MUST be verifiably empty (0 records).
            if preflight_count > 0:
                msg = (
                    f"Namespace '{CANONICAL_NAMESPACE}' is not empty (contains {preflight_count} vectors) "
                    f"for a fresh canary run (expected 0). Refusing to write to prevent contamination. "
                    "The namespace must be manually verified and cleared by the operator before running a fresh canary."
                )
                logger.error(msg)
                report_data["preflight_status"] = (
                    f"FAILED: Contaminated ({preflight_count} vectors)"
                )
                raise CanaryError(
                    msg,
                    category="NamespaceContaminatedPreflight",
                    safe_next_action="Verify index contents. If stale, manually clear vectors in the pilot namespace. Never automatically clear.",
                )
            # Defense-in-depth: check ID enumeration on fresh run to ensure 0 IDs
            try:
                actual_fresh_ids = store.list_vector_ids(namespace=CANONICAL_NAMESPACE)
            except Exception as exc:
                logger.error("Pre-write namespace ID enumeration check failed: %s", exc)
                report_data["preflight_status"] = f"FAILED: ID enumeration error {exc}"
                raise CanaryError(
                    f"Pre-write namespace preflight ID enumeration failed: {exc}",
                    category="PreflightProviderFailure",
                    safe_next_action="Verify Pinecone ID enumeration support and connectivity before running.",
                ) from exc

            if len(actual_fresh_ids) > 0:
                msg = (
                    f"Namespace '{CANONICAL_NAMESPACE}' contains {len(actual_fresh_ids)} vector IDs "
                    f"for a fresh canary run (expected 0). Refusing to write to prevent contamination."
                )
                logger.error(msg)
                report_data["preflight_status"] = (
                    f"FAILED: Contaminated ({len(actual_fresh_ids)} IDs)"
                )
                raise CanaryError(
                    msg,
                    category="NamespaceContaminatedPreflight",
                    safe_next_action="Verify index contents. If stale, manually clear vectors in the pilot namespace. Never automatically clear.",
                )

        else:
            # Resume run: verify exact resume ownership using deterministic vector IDs
            expected_completed_ids = {
                r["id"]
                for i, b in enumerate(batches)
                if batch_digests[i] in completed_digests
                for r in b
            }
            completed_record_count = len(expected_completed_ids)

            if preflight_count > CANARY_EXPECTED_TOTAL:
                msg = (
                    f"Namespace '{CANONICAL_NAMESPACE}' contains {preflight_count} vectors, "
                    f"which exceeds the total expected canary size of {CANARY_EXPECTED_TOTAL}. "
                    "Namespace is contaminated or stale."
                )
                logger.error(msg)
                report_data["preflight_status"] = (
                    f"FAILED: Contaminated ({preflight_count} vectors > {CANARY_EXPECTED_TOTAL})"
                )
                raise CanaryError(
                    msg,
                    category="NamespaceContaminatedPreflight",
                    safe_next_action="Verify index contents. Stale vectors must be manually cleared by the operator before resuming.",
                )

            # Enumerate vector IDs in namespace
            try:
                actual_namespace_ids = store.list_vector_ids(namespace=CANONICAL_NAMESPACE)
            except Exception as exc:
                logger.error("Pre-write resume ownership ID enumeration failed: %s", exc)
                report_data["preflight_status"] = f"FAILED: Resume ownership unverifiable ({exc})"
                raise CanaryError(
                    f"Pre-write resume ownership verification failed due to provider error: {exc}",
                    category="ResumeOwnershipUnverifiable",
                    safe_next_action="Verify Pinecone ID enumeration support, network connectivity, and credentials before retrying with --resume.",
                ) from exc

            actual_id_set = set(actual_namespace_ids)

            # Verify exact ownership: actual namespace IDs must match expected completed IDs precisely
            if actual_id_set != expected_completed_ids or preflight_count != completed_record_count:
                missing_ids = expected_completed_ids - actual_id_set
                unrelated_ids = actual_id_set - expected_completed_ids
                msg = (
                    f"Resume ownership verification failed for namespace '{CANONICAL_NAMESPACE}': "
                    f"expected exactly {len(expected_completed_ids)} completed IDs from checkpoint, "
                    f"found {len(actual_id_set)} actual IDs (stats vector count: {preflight_count}). "
                    f"Missing expected IDs: {len(missing_ids)}, Unrelated/extra IDs: {len(unrelated_ids)}."
                )
                logger.error(msg)
                report_data["preflight_status"] = (
                    f"FAILED: Resume ownership mismatch (missing {len(missing_ids)}, unexpected {len(unrelated_ids)})"
                )
                raise CanaryError(
                    msg,
                    category="ResumeOwnershipMismatch",
                    safe_next_action="Verify index contents against checkpoint before retrying with --resume. Do not automatically clear.",
                )

        logger.info(
            "Pre-write namespace preflight PASSED (namespace '%s' has %d vectors, is_resume=%s).",
            CANONICAL_NAMESPACE,
            preflight_count,
            is_resume_run,
        )
    report_data["preflight_status"] = "PASSED"

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

    # Thread-safe token rate limiter — uses the tested _TokenRateLimiter class.
    _rate_limiter = _TokenRateLimiter(tokens_per_window=token_rate_limit)

    def _upsert_batch(
        batch_idx: int, batch: list[dict], digest: str
    ) -> tuple[int, int, int, float]:
        """Submit one batch with retry. Returns (submitted, attempts, retries, throttle_wait)."""
        attempts = 0
        retries = 0
        throttle_wait = 0.0
        last_exc: Exception | None = None

        batch_tokens = sum(r.get("token_length", 0) for r in batch)

        for attempt in range(1, MAX_RETRIES + 1):
            # Thread-safe token-rate pacing via _TokenRateLimiter.
            # Retried attempts conservatively re-reserve their tokens.
            waited = _rate_limiter.acquire(max(batch_tokens, 1))
            if waited > 0:
                logger.info(
                    "Token rate limit: batch %d waited %.1fs for capacity "
                    "(%d tokens/window limit: %d)",
                    batch_idx + 1,
                    waited,
                    batch_tokens,
                    token_rate_limit,
                )
            throttle_wait += waited

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
        raise CanaryError(
            f"{failed_batches} batch(es) failed permanently.",
            category="UpsertError",
            safe_next_action="Run with --resume to retry failed batches.",
        )

    # ── Step 7: Post-write reconciliation — exact COUNT and exact ID SET ──────
    # Count equality alone is insufficient. We require BOTH:
    #   1. Statistics count == exactly expected_total, AND
    #   2. The enumerated namespace ID set == the manifest-derived expected ID set.
    # Any enumeration failure/ambiguity fails closed. No further upserts occur.
    if len(expected_id_set_all) != expected_total:
        raise CanaryError(
            f"Expected manifest-derived ID set has {len(expected_id_set_all)} unique IDs, "
            f"not {expected_total} for scope '{scope}'. Refusing to reconcile.",
            category="ExpectedIdSetInvalid",
            safe_next_action="Regenerate the manifest — record IDs are not unique.",
        )

    logger.info("Waiting for index freshness (max %ds) …", FRESHNESS_POLL_MAX_WAIT)
    deadline = time.monotonic() + FRESHNESS_POLL_MAX_WAIT
    wait = FRESHNESS_POLL_BASE
    final_count: int | None = None
    count_reconciled = False
    contaminated = False

    while time.monotonic() < deadline:
        try:
            stats = index.describe_index_stats()
            count = _get_ns_vector_count(stats)
            if count is None:
                logger.warning("Could not extract namespace count from stats response.")
            else:
                logger.info("Namespace '%s' vector count: %d", CANONICAL_NAMESPACE, count)
                final_count = count
                if count == expected_total:
                    count_reconciled = True
                    break
                elif count > expected_total:
                    contaminated = True
                    logger.error(
                        "Namespace '%s' has %d vectors — EXCEEDS expected %d. "
                        "Namespace may be contaminated or stale. "
                        "This must be cleared by the live indexing operator before retrying.",
                        CANONICAL_NAMESPACE,
                        count,
                        expected_total,
                    )
                    break
        except Exception as e:
            logger.warning("Freshness poll error: %s", e)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(wait, remaining))
        wait = min(wait * 2, 30)

    # Store the COUNT result separately from the exact-ID result.
    if contaminated:
        count_result = (
            f"CONTAMINATED — namespace has {final_count} vectors, expected {expected_total}."
        )
    elif not count_reconciled:
        count_result = f"TIMEOUT — count={final_count}, expected={expected_total}"
    else:
        count_result = f"PASS — count={final_count}"
    report_data["count_reconciliation"] = count_result

    def _finalize(
        status: str,
        exact_id_result: str,
        failure: CanaryError | None,
    ) -> None:
        end_time = datetime.now(UTC).isoformat()
        duration = (
            datetime.fromisoformat(end_time) - datetime.fromisoformat(start_time)
        ).total_seconds()
        rps = round(total_records / max(duration, 0.001), 2) if duration > 0 else 0
        tpm = round(tokens_submitted / max(duration / 60, 0.001), 0) if duration > 0 else 0
        report_data.update(
            {
                "status": status,
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
                "count_reconciliation": count_result,
                "exact_id_reconciliation": exact_id_result,
                # Retained for backward-compatible report readers.
                "freshness_reconciliation": count_result,
            }
        )
        if failure is not None:
            raise failure
        logger.info(
            "Indexing complete: %d records, %d tokens in %.1fs (%.1f r/s, %d t/min). "
            "Exact-ID verification: %s",
            total_records,
            total_tokens,
            duration,
            rps,
            tpm,
            exact_id_result,
        )

    # Fail closed BEFORE ID enumeration if count not reached.
    if not count_reconciled:
        category = "ContaminatedNamespace" if contaminated else "ReconciliationTimeout"
        safe_action = (
            "The namespace must be cleared by the live indexing operator before retrying."
            if contaminated
            else "Wait for index propagation and retry with --resume."
        )
        logger.error("Count reconciliation failed: %s", count_result)
        _finalize(
            "failed",
            "NOT RUN — count precondition not met",
            CanaryError(
                f"Count reconciliation did not reach exactly {expected_total} vectors: "
                f"{count_result}",
                category=category,
                safe_next_action=safe_action,
            ),
        )
        return

    # Count == expected_total. Now enumerate IDs and require EXACT set equality.
    logger.info(
        "Count reconciled at %d. Enumerating vector IDs for exact-ID reconciliation …", final_count
    )
    try:
        enumerated = store.list_vector_ids(namespace=CANONICAL_NAMESPACE)
    except Exception as exc:
        logger.error("Post-write ID enumeration failed: %s", exc)
        _finalize(
            "failed",
            f"UNVERIFIABLE — enumeration error: {type(exc).__name__}",
            CanaryError(
                f"Post-write reconciliation could not enumerate vector IDs: {exc}",
                category="PostWriteReconciliationUnverifiable",
                safe_next_action=(
                    "Verify Pinecone ID enumeration support and connectivity. "
                    "Do not assume success — the exact ID set is unverified."
                ),
            ),
        )
        return

    # Duplicates are already rejected inside list_vector_ids (fail-closed), but
    # guard again defensively: a set smaller than the list would signal a bug.
    enumerated_set = set(enumerated)
    if len(enumerated) != len(enumerated_set):
        _finalize(
            "failed",
            "UNVERIFIABLE — duplicate IDs enumerated",
            CanaryError(
                "Post-write ID enumeration returned duplicate IDs.",
                category="PostWriteReconciliationUnverifiable",
                safe_next_action="ID enumeration is unreliable; do not assume success.",
            ),
        )
        return

    if len(enumerated_set) != expected_total:
        _finalize(
            "failed",
            f"FAIL — enumerated {len(enumerated_set)} unique IDs, expected {expected_total}",
            CanaryError(
                f"Post-write ID enumeration returned {len(enumerated_set)} unique IDs, "
                f"expected exactly {expected_total} for scope '{scope}'.",
                category="PostWriteOwnershipMismatch",
                safe_next_action=(
                    "Namespace contents do not match the manifest. Verify and clear "
                    "manually before retrying. Do not automatically clear."
                ),
            ),
        )
        return

    missing = expected_id_set_all - enumerated_set
    unexpected = enumerated_set - expected_id_set_all
    if missing or unexpected:
        # Report only COUNTS of discrepancies — never dump the ID lists.
        _finalize(
            "failed",
            f"FAIL — missing {len(missing)}, unexpected {len(unexpected)}",
            CanaryError(
                f"Post-write ID set does not equal the manifest-derived expected set: "
                f"missing {len(missing)} expected IDs, {len(unexpected)} unexpected IDs.",
                category="PostWriteOwnershipMismatch",
                safe_next_action=(
                    "Namespace contents do not match the manifest. Verify and clear "
                    "manually before retrying. Do not automatically clear."
                ),
            ),
        )
        return

    # All conditions satisfied: count == expected_total AND exact ID-set equality.
    logger.info(
        "Post-write EXACT-ID reconciliation PASSED: %d IDs match manifest exactly.",
        expected_total,
    )
    _finalize(
        "success",
        f"PASS — {expected_total} IDs match manifest exactly",
        None,
    )


if __name__ == "__main__":
    main()
