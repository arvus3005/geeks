#!/usr/bin/env python3
"""Prepared-data-only ingestion from a canary/pilot manifest.

Reads the JSONL produced by prepare_canary.py and ingests it into Pinecone.

Dry-run mode (--dry-run):
  - Validates manifest and data file
  - Builds batches and calculates budget
  - Prints a full plan
  - NEVER imports or instantiates Pinecone
  - NEVER reads PINECONE_API_KEY

Live mode (no --dry-run):
  - Requires all manifest/data validations to pass before constructing Pinecone client
  - Credentials must come from PINECONE_API_KEY environment variable only
  - Refuses unsafe collection names (smoke/pilot safety guards apply)

Usage:
  # Dry run (no credentials needed):
  uv run python scripts/ingest_prepared.py --manifest artifacts/prepared/<id>_manifest.json --dry-run

  # Live ingest:
  PINECONE_API_KEY=... uv run python scripts/ingest_prepared.py \\
      --manifest artifacts/prepared/<id>_manifest.json \\
      --namespace pilot_canary_001
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Batch limits for Pinecone Starter
MAX_RECORDS_PER_BATCH = 96
MAX_BYTES_PER_BATCH = 1_800_000  # 1.8 MB serialized

# Required manifest fields
REQUIRED_MANIFEST_FIELDS = {
    "manifest_schema_version",
    "manifest_id",
    "mode",
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
}

FORBIDDEN_FIELDS = {"query", "Answer", "Eng_Query", "Eng_Answer", "query_type", "is_selected"}


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
        "--pinecone-index",
        default=None,
        help="Pinecone index name (overrides PINECONE_INDEX env var)",
    )
    p.add_argument(
        "--embed-model",
        default="multilingual-e5-large",
        help="Pinecone integrated embed model name",
    )
    return p


def _load_manifest(manifest_path: Path) -> dict:
    """Load and validate the manifest file. Raises on any validation failure."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Check required fields
    missing = REQUIRED_MANIFEST_FIELDS - set(manifest.keys())
    if missing:
        raise ValueError(f"Manifest missing required fields: {sorted(missing)}")

    # Verify manifest checksum
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

    # Check forbidden field audit
    if manifest.get("forbidden_field_audit", "").startswith("FAIL"):
        raise ValueError(
            f"Manifest forbidden field audit failed: {manifest['forbidden_field_audit']}. "
            "Refusing to ingest."
        )

    return manifest


def _verify_data_file(manifest: dict) -> Path:
    """Verify the prepared JSONL file exists and matches its checksum."""
    record_path = Path(manifest["prepared_record_path"])
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


def _load_records(record_path: Path) -> list[dict]:
    """Load and validate all records from the JSONL file."""
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

            # Check for forbidden fields
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
        # Check if adding this record would exceed limits
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
    print(
        f"  Total tokens      : {manifest.get('total_tokens', 'N/A'):,}"
        if isinstance(manifest.get("total_tokens"), int)
        else f"  Total tokens      : {manifest.get('total_tokens', 'N/A')}"
    )
    print(f"  Target namespace  : {namespace}")
    print(f"  Batches           : {len(batches)}")
    print(f"  Max batch records : {MAX_RECORDS_PER_BATCH}")
    print(f"  Max batch bytes   : {MAX_BYTES_PER_BATCH:,}")
    print()

    # Per-language breakdown
    lang_counts: dict[str, int] = {}
    for rec in records:
        lang = rec.get("language", "unknown")
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
    print("  Per-language counts:")
    for lang, count in sorted(lang_counts.items()):
        print(f"    {lang}: {count}")
    print()

    # Batch sizes
    print("  Batch sizes (records / bytes):")
    for i, batch in enumerate(batches):
        batch_bytes = sum(len(json.dumps(r, ensure_ascii=False).encode()) for r in batch)
        print(f"    Batch {i+1:03d}: {len(batch)} records, {batch_bytes:,} bytes")

    print()
    print("  Ready for write   :", manifest.get("ready_for_write", False))
    failures = manifest.get("readiness_failures", [])
    if failures:
        print("  Readiness failures:")
        for f in failures:
            print(f"    - {f}")
    print("=" * 60)
    print()


def main() -> None:
    args = _build_parser().parse_args()

    manifest_path: Path = args.manifest

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
    record_path = _verify_data_file(manifest)
    logger.info("Data file verified: %s", record_path)

    records = _load_records(record_path)
    logger.info("Loaded %d records from data file", len(records))

    if len(records) != manifest["total_records"]:
        raise ValueError(
            f"Record count mismatch: manifest says {manifest['total_records']}, "
            f"file has {len(records)}."
        )

    # Step 3: Determine target namespace
    namespace = args.namespace or f"pilot_{manifest['manifest_id']}"

    # Step 4: Build batches
    batches = _build_batches(records)
    logger.info("Built %d batches from %d records", len(batches), len(records))

    if args.dry_run:
        # Dry-run: print plan and exit. Never touch Pinecone.
        _print_plan(manifest, records, batches, namespace)
        print("DRY RUN complete — no records were written.")
        sys.exit(0)

    # Step 5: Live mode — all validations must pass before constructing Pinecone client
    import os

    if not manifest.get("ready_for_write"):
        failures = manifest.get("readiness_failures", ["unknown"])
        logger.error("Manifest is not ready_for_write: %s", failures)
        sys.exit(1)

    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        logger.error("PINECONE_API_KEY environment variable is not set. Cannot proceed.")
        sys.exit(1)

    import os as _os

    index_name = args.pinecone_index or _os.environ.get("PINECONE_INDEX", "msmarco-xi")

    # Only import Pinecone after all validations pass
    from pinecone import Pinecone

    from hhgoa_rag.pinecone_store import PineconeStore, is_safe_namespace

    if not is_safe_namespace(namespace):
        logger.error(
            "Namespace '%s' is not a safe namespace for prepared ingestion. "
            "Only smoke/* and pilot_* namespaces are permitted. "
            "Use the full ingestion pipeline for the full namespace.",
            namespace,
        )
        sys.exit(1)

    logger.info("Connecting to Pinecone index: %s", index_name)
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    store = PineconeStore(index, embed_model=args.embed_model)

    total_submitted = 0
    for i, batch in enumerate(batches):
        logger.info(
            "Upserting batch %d/%d (%d records) into namespace '%s' …",
            i + 1,
            len(batches),
            len(batch),
            namespace,
        )
        submitted = store.upsert_records(batch, namespace=namespace, context="pilot")
        total_submitted += submitted
        logger.info("Batch %d/%d complete: %d records submitted", i + 1, len(batches), submitted)

    logger.info(
        "Ingestion complete: %d/%d records submitted to namespace '%s'",
        total_submitted,
        len(records),
        namespace,
    )
    if total_submitted != len(records):
        logger.error("Submitted %d but expected %d!", total_submitted, len(records))
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
