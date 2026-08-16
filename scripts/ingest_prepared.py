#!/usr/bin/env python3
"""Prepared-data-only OFFLINE VALIDATION / DRY-RUN for a canary/pilot manifest.

Reads the JSONL produced by prepare_canary.py, validates the manifest and every
record, builds batches, and prints a plan. It performs NO live writes.

The legacy live-ingestion path (`--execute`) is DISABLED: it lacks the canonical
empty-namespace preflight, exact resume ownership verification, atomic per-batch
checkpointing, and post-write exact-ID reconciliation. Live indexing is performed
exclusively by scripts/index_canary.py.

Behaviour:
  - Default / --dry-run: validate manifest and data file, build batches, print
    plan. NEVER imports or instantiates Pinecone. NEVER reads PINECONE_API_KEY.
  - --execute: exits non-zero (2) BEFORE constructing any Pinecone client or
    reading credentials, even when CONFIRM_PINECONE_WRITE=1 and
    PINECONE_API_KEY are present. Redirects the operator to index_canary.py.

Usage:
  # Offline validation / dry-run (no credentials needed):
  uv run python scripts/ingest_prepared.py \\
      --manifest artifacts/prepared/<id>_manifest.json --dry-run

  # Live ingestion (use the approved canary indexer):
  export PINECONE_API_KEY=<your-key>
  CONFIRM_PINECONE_WRITE=1 \\
      uv run python scripts/index_canary.py \\
      --manifest artifacts/prepared/<id>_manifest.json \\
      --execute --resume --concurrency 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

# Import canonical contract — no hand-written contract dicts here.
from hhgoa_rag.pinecone_contract import (
    CLOUD,
    DIMENSION,
    FIELD_MAP,
    MANIFEST_SCHEMA_VERSION,
    MAX_BATCH_SIZE,
    METRIC,
    MODEL,
    READ_PARAMETERS,
    REGION,
    TEXT_FIELD,
    WRITE_PARAMETERS,
    canonical_contract,
)
from hhgoa_rag.pinecone_contract import (
    INDEX_NAME as CANONICAL_INDEX_NAME,
)
from hhgoa_rag.pinecone_contract import (
    NAMESPACE as SAFE_NAMESPACE,
)
from hhgoa_rag.pinecone_contract import (
    contract_fingerprint as _canonical_fingerprint,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Batch limits for Pinecone Starter
MAX_RECORDS_PER_BATCH = MAX_BATCH_SIZE
MAX_BYTES_PER_BATCH = 1_800_000  # 1.8 MB serialized

# Expected per-language counts for this pilot
EXPECTED_PER_LANG = 100
EXPECTED_TOTAL = 300
EXPECTED_LANGUAGES = {"en", "hi", "bn"}

# Bounded retry parameters
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0  # seconds

# Required manifest fields — all must be present; absent fields fail closed.
REQUIRED_MANIFEST_FIELDS = {
    "manifest_schema_version",
    "manifest_id",
    "manifest_checksum",
    "mode",
    "contract_version",
    "contract_fingerprint",
    "index_contract",
    "index_name",
    "index_namespace",
    "dataset_repo",
    "dataset_revision",
    "total_records",
    "total_tokens",
    "prepared_record_path",
    "prepared_record_checksum",
    "ready_for_write",
    "readiness_failures",
    "forbidden_field_audit",
    "tokenizer_fingerprint",
    "actual_per_language_records",
}

FORBIDDEN_FIELDS = {"query", "Answer", "Eng_Query", "Eng_Answer", "query_type", "is_selected"}

INDEX_CONTRACT = {
    "model": MODEL,
    "dimension": DIMENSION,
    "metric": METRIC,
    "field_map": FIELD_MAP,
    "write_parameters": WRITE_PARAMETERS,
    "read_parameters": READ_PARAMETERS,
    "cloud": CLOUD,
    "region": REGION,
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ingest prepared JSONL into Pinecone (manifest-driven)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to the _manifest.json produced by prepare_canary.py",
    )
    p.add_argument(
        "--namespace",
        default=None,
        help="Pinecone namespace to write into (default: pilot_<manifest_id>)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate manifest/data, build batches, print plan. "
            "Never imports Pinecone, never reads PINECONE_API_KEY."
        ),
    )
    p.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Perform real Pinecone writes. "
            "Also requires CONFIRM_PINECONE_WRITE=1 and PINECONE_API_KEY."
        ),
    )
    p.add_argument(
        "--pinecone-index",
        default=None,
        help="Pinecone index name (overrides PINECONE_INDEX env var)",
    )
    p.add_argument(
        "--embed-model",
        default=MODEL,
        help="Pinecone integrated embed model name",
    )
    return p


def _is_live_mode(args: argparse.Namespace) -> bool:
    """True only if --execute AND CONFIRM_PINECONE_WRITE=1 are both set."""
    import os

    return args.execute and os.environ.get("CONFIRM_PINECONE_WRITE") == "1"


def _load_manifest(manifest_path: Path) -> dict:
    """Load and validate the manifest file. Raises on any validation failure."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    missing = REQUIRED_MANIFEST_FIELDS - set(manifest.keys())
    if missing:
        raise ValueError(f"Manifest missing required fields: {sorted(missing)}")

    # Manifest schema version must be exactly the canonical version.
    mschema = manifest.get("manifest_schema_version")
    if mschema != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Manifest schema version {mschema!r} is not the required "
            f"{MANIFEST_SCHEMA_VERSION!r}. Legacy manifests are rejected."
        )

    stored_checksum = manifest.get("manifest_checksum")
    if stored_checksum:
        manifest_for_checksum = {k: v for k, v in manifest.items() if k != "manifest_checksum"}
        computed = hashlib.sha256(
            json.dumps(manifest_for_checksum, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        if computed != stored_checksum:
            raise ValueError(
                f"Manifest checksum mismatch: stored={stored_checksum}, computed={computed}. "
                "Manifest may have been tampered with."
            )

    if manifest.get("forbidden_field_audit", "").startswith("FAIL"):
        raise ValueError(
            f"Manifest forbidden field audit failed: {manifest['forbidden_field_audit']}. "
            "Refusing to ingest."
        )

    # Contract fingerprint validation: mandatory, must match canonical.
    manifest_contract_fp = manifest["contract_fingerprint"]
    expected_fp = _canonical_fingerprint()
    if manifest_contract_fp != expected_fp:
        raise ValueError(
            f"Manifest contract_fingerprint {manifest_contract_fp!r} does not match "
            f"canonical fingerprint {expected_fp!r}. Manifest may bind a different "
            "index contract. Refusing to ingest."
        )

    # Contract version validation: mandatory.
    manifest_contract_version = manifest["contract_version"]
    if manifest_contract_version not in ("1",):
        raise ValueError(
            f"Unknown contract_version {manifest_contract_version!r} in manifest. "
            "Refusing to ingest."
        )

    # Index name validation: mandatory, must match canonical.
    manifest_index_name = manifest["index_name"]
    if manifest_index_name != CANONICAL_INDEX_NAME:
        raise ValueError(
            f"Manifest declares index_name {manifest_index_name!r} but canonical "
            f"index name is {CANONICAL_INDEX_NAME!r}. Refusing to ingest."
        )

    # Namespace validation: mandatory, must match canonical.
    manifest_namespace = manifest["index_namespace"]
    if manifest_namespace != SAFE_NAMESPACE:
        raise ValueError(
            f"Manifest declares namespace {manifest_namespace!r} but canonical "
            f"namespace is {SAFE_NAMESPACE!r}. Refusing to ingest."
        )

    # Embedded index contract validation: mandatory, must exactly match canonical.
    manifest_contract = manifest["index_contract"]
    expected_contract = canonical_contract()
    # Check for missing keys in manifest contract
    missing_contract_keys = set(expected_contract.keys()) - set(manifest_contract.keys())
    if missing_contract_keys:
        raise ValueError(
            f"Manifest index_contract is missing keys: {sorted(missing_contract_keys)}. "
            "Refusing to ingest."
        )
    # Check for unexpected keys in manifest contract
    extra_contract_keys = set(manifest_contract.keys()) - set(expected_contract.keys())
    if extra_contract_keys:
        raise ValueError(
            f"Manifest index_contract has unexpected keys: {sorted(extra_contract_keys)}. "
            "Refusing to ingest."
        )
    # Check for value mismatches
    mismatches = []
    for key, expected_val in expected_contract.items():
        actual_val = manifest_contract.get(key)
        if actual_val != expected_val:
            mismatches.append(f"  {key}: expected {expected_val!r}, got {actual_val!r}")
    if mismatches:
        raise ValueError(
            "Manifest index_contract differs from canonical:\n"
            + "\n".join(mismatches)
            + "\nRefusing to ingest."
        )

    return manifest


def _resolve_record_path(manifest: dict, manifest_path: Path) -> Path:
    """Resolve the prepared JSONL path portably.

    The manifest stores a bare filename ('prepared_record_path'); we resolve it
    relative to the manifest's own directory so that ingestion works regardless
    of the current working directory (repo root, /tmp, anywhere).
    """
    stored = manifest["prepared_record_path"]
    candidate = Path(stored)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    # Resolve relative to the manifest directory (portable).
    sibling = manifest_path.resolve().parent / candidate.name
    if sibling.exists():
        return sibling
    # Fall back to any recorded full path for legacy manifests.
    full = manifest.get("prepared_record_path_full")
    if full and Path(full).exists():
        return Path(full)
    return sibling


def _verify_data_file(manifest: dict, manifest_path: Path) -> Path:
    """Verify the prepared JSONL file exists and matches its checksum."""
    record_path = _resolve_record_path(manifest, manifest_path)
    if not record_path.exists():
        raise FileNotFoundError(
            f"Prepared data file not found: {record_path}. " "Run prepare_canary.py first."
        )

    stored_checksum = manifest.get("prepared_record_checksum")
    if stored_checksum:
        h = hashlib.sha256()
        with open(record_path, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                h.update(block)
        computed = h.hexdigest()
        if computed != stored_checksum:
            raise ValueError(
                f"Data file checksum mismatch: stored={stored_checksum}, computed={computed}. "
                f"File: {record_path}"
            )

    return record_path


def _load_and_validate_records(record_path: Path, manifest: dict) -> list[dict]:
    """Load, validate, and return all records from the JSONL file."""
    from hhgoa_rag.ingestion.schema import SchemaViolationError, validate_record
    from hhgoa_rag.ingestion.tokenizer import MODEL_INPUT_LIMIT

    records = []
    with open(record_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {lineno} of {record_path}: {e}") from e

            # Forbidden field check
            bad = FORBIDDEN_FIELDS & set(rec.keys())
            if bad:
                raise ValueError(
                    f"Forbidden fields found in record on line {lineno}: {sorted(bad)}. "
                    "Refusing to ingest."
                )

            if not rec.get("id"):
                raise ValueError(f"Record on line {lineno} has empty or missing 'id'.")

            if not rec.get(TEXT_FIELD):
                raise ValueError(f"Record on line {lineno} has empty or missing {TEXT_FIELD!r}.")

            # Token limit check
            token_length = rec.get("token_length", 0)
            if not isinstance(token_length, int) or token_length <= 0:
                raise ValueError(
                    f"Record on line {lineno} has invalid token_length: {token_length!r}"
                )
            if token_length > MODEL_INPUT_LIMIT:
                raise ValueError(
                    f"Record on line {lineno} id={rec['id']!r} exceeds model token limit: "
                    f"{token_length} > {MODEL_INPUT_LIMIT}"
                )

            # Individual record byte size check
            rec_bytes = len(json.dumps(rec, ensure_ascii=False).encode("utf-8"))
            if rec_bytes > MAX_BYTES_PER_BATCH:
                raise ValueError(
                    f"Record on line {lineno} id={rec['id']!r} serialized size {rec_bytes} "
                    f"exceeds individual limit {MAX_BYTES_PER_BATCH}. Refusing to ingest."
                )

            # Authoritative schema validation
            try:
                validate_record(rec)
            except SchemaViolationError as e:
                raise ValueError(f"Schema violation on line {lineno}: {e}") from e

            records.append(rec)

    # Total record count validation against manifest
    manifest_total = manifest["total_records"]
    if len(records) != manifest_total:
        raise ValueError(
            f"Record count mismatch: manifest says {manifest_total}, " f"file has {len(records)}."
        )

    # Per-language count validation against manifest's declared counts
    lang_counts: dict[str, int] = {}
    for rec in records:
        lang = rec.get("language", "unknown")
        lang_counts[lang] = lang_counts.get(lang, 0) + 1

    manifest_lang_counts = manifest.get("actual_per_language_records", {})
    for lang, expected_count in manifest_lang_counts.items():
        actual = lang_counts.get(lang, 0)
        if actual != expected_count:
            raise ValueError(
                f"Per-language count mismatch for '{lang}': "
                f"manifest says {expected_count}, file has {actual}."
            )

    # When this is a full pilot (300 records), enforce exact 100/100/100
    if manifest_total == EXPECTED_TOTAL:
        for lang in EXPECTED_LANGUAGES:
            actual = lang_counts.get(lang, 0)
            if actual != EXPECTED_PER_LANG:
                raise ValueError(
                    f"Per-language count mismatch for '{lang}': expected {EXPECTED_PER_LANG}, "
                    f"got {actual}."
                )

    # Total token validation
    file_total_tokens = sum(r.get("token_length", 0) for r in records)
    manifest_total_tokens = manifest.get("total_tokens", 0)
    if manifest_total_tokens > 0 and file_total_tokens != manifest_total_tokens:
        raise ValueError(
            f"Total token count mismatch: manifest={manifest_total_tokens}, "
            f"file={file_total_tokens}."
        )

    return records


def _load_records(record_path: Path) -> list[dict]:
    """Basic record loader without manifest cross-validation.

    Kept for backward compatibility with tests that call this standalone.
    For the live/dry-run path use _load_and_validate_records.
    """
    records = []
    with open(record_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {lineno} of {record_path}: {e}") from e

            bad = FORBIDDEN_FIELDS & set(rec.keys())
            if bad:
                raise ValueError(
                    f"Forbidden fields found in record on line {lineno}: {sorted(bad)}. "
                    "Refusing to ingest."
                )

            if not rec.get("id"):
                raise ValueError(f"Record on line {lineno} has empty or missing 'id'.")

            records.append(rec)
    return records


def _build_batches(records: list[dict]) -> list[list[dict]]:
    """Linear greedy packer: max 96 records, max 1,800,000 serialized bytes per batch."""
    batches: list[list[dict]] = []
    current_batch: list[dict] = []
    current_bytes = 0

    for rec in records:
        rec_bytes = len(json.dumps(rec, ensure_ascii=False).encode("utf-8"))
        if current_batch and (
            len(current_batch) >= MAX_RECORDS_PER_BATCH
            or current_bytes + rec_bytes > MAX_BYTES_PER_BATCH
        ):
            batches.append(current_batch)
            current_batch = []
            current_bytes = 0
        current_batch.append(rec)
        current_bytes += rec_bytes

    if current_batch:
        batches.append(current_batch)

    return batches


def _print_plan(
    manifest: dict, records: list[dict], batches: list[list[dict]], namespace: str
) -> None:
    """Print ingestion plan for dry-run mode."""
    print("\n" + "=" * 60)
    print("INGESTION PLAN (dry-run — no Pinecone calls)")
    print("=" * 60)
    print(f"  Manifest ID       : {manifest['manifest_id']}")
    print(f"  Dataset revision  : {manifest['dataset_revision']}")
    print(f"  Total records     : {len(records)}")
    total_tokens = manifest.get("total_tokens", "N/A")
    if isinstance(total_tokens, int):
        print(f"  Total tokens      : {total_tokens:,}")
    else:
        print(f"  Total tokens      : {total_tokens}")
    print(f"  Target namespace  : {namespace}")
    print(f"  Batches           : {len(batches)}")
    print(f"  Max batch records : {MAX_RECORDS_PER_BATCH}")
    print(f"  Max batch bytes   : {MAX_BYTES_PER_BATCH:,}")
    print()

    lang_counts: dict[str, int] = {}
    for rec in records:
        lang = rec.get("language", "unknown")
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
    print("  Per-language counts:")
    for lang, count in sorted(lang_counts.items()):
        print(f"    {lang}: {count}")
    print()

    print("  Batch sizes (records / bytes):")
    max_batch_bytes = 0
    for i, batch in enumerate(batches):
        batch_bytes = sum(len(json.dumps(r, ensure_ascii=False).encode()) for r in batch)
        max_batch_bytes = max(max_batch_bytes, batch_bytes)
        print(f"    Batch {i+1:03d}: {len(batch)} records, {batch_bytes:,} bytes")

    print()
    print(f"  Max actual batch bytes : {max_batch_bytes:,}")
    print("  Ready for write   :", manifest.get("ready_for_write", False))
    failures = manifest.get("readiness_failures", [])
    if failures:
        print("  Readiness failures:")
        for f_msg in failures:
            print(f"    - {f_msg}")
    print("=" * 60)
    print()


def main() -> None:
    args = _build_parser().parse_args()

    manifest_path: Path = args.manifest

    # Fail closed on contradictory flags.
    if args.dry_run and args.execute:
        logger.error("--dry-run and --execute are mutually exclusive. Aborting.")
        sys.exit(2)

    # ── Legacy live-ingestion path is permanently DISABLED ────────────────────
    # This script is retained ONLY for offline manifest validation and dry-run
    # planning. Live writes bypass the canonical empty-namespace preflight, exact
    # resume ownership, atomic checkpointing, and post-write exact-ID
    # reconciliation. The refusal occurs BEFORE any Pinecone client construction
    # or credential read, even when CONFIRM_PINECONE_WRITE=1 and
    # PINECONE_API_KEY are present.
    if args.execute:
        logger.error(
            "Live ingestion via ingest_prepared.py --execute is DISABLED.\n"
            "This legacy path lacks the canonical empty-namespace preflight, exact\n"
            "resume ownership verification, atomic per-batch checkpointing, and\n"
            "post-write exact-ID reconciliation.\n"
            "\n"
            "Use the approved canary indexer instead:\n"
            "  export PINECONE_API_KEY=<your-key>\n"
            "  CONFIRM_PINECONE_WRITE=1 \\\n"
            "    uv run python scripts/index_canary.py \\\n"
            "    --manifest %s \\\n"
            "    --execute --resume --concurrency 4\n"
            "\n"
            "ingest_prepared.py remains available for offline validation:\n"
            "  uv run python scripts/ingest_prepared.py --manifest %s --dry-run",
            manifest_path,
            manifest_path,
        )
        sys.exit(2)

    # Past the --execute refusal above, this script is ALWAYS an offline dry-run.
    # Pinecone is never imported and PINECONE_API_KEY is never read below.

    # Step 1: Load and validate manifest (always, including dry-run)
    logger.info("Loading manifest: %s", manifest_path)
    manifest = _load_manifest(manifest_path)
    logger.info(
        "Manifest loaded: id=%s revision=%s records=%d",
        manifest["manifest_id"],
        manifest["dataset_revision"],
        manifest["total_records"],
    )

    # Step 2: Verify and load data file (always, including dry-run)
    record_path = _verify_data_file(manifest, manifest_path)
    logger.info("Data file verified: %s", record_path)

    records = _load_and_validate_records(record_path, manifest)
    logger.info("Loaded and validated %d records from data file", len(records))

    # Step 3: Determine target namespace (default to the immutable safe namespace)
    namespace = args.namespace or SAFE_NAMESPACE

    # Step 4: Build batches
    batches = _build_batches(records)
    logger.info("Built %d batches from %d records", len(batches), len(records))

    # Offline dry-run is the only supported mode for this script.
    _print_plan(manifest, records, batches, namespace)
    print("DRY RUN complete — no records were written.")
    print(
        "NOTE: Live ingestion is performed exclusively by scripts/index_canary.py "
        "(ingest_prepared.py --execute is disabled)."
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
