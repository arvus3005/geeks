#!/usr/bin/env python3
"""Offline canary preparation — no Pinecone imports, no provider calls.

Generates the exact immutable records that live ingestion will consume.
Live ingestion MUST read from the prepared JSONL; it must not independently
stream a new dataset subset.

Default: 300 records — 100 English, 100 Hindi, 100 Bengali.
Source configs: hi (Hindi) and bn (Bengali).
English passages are extracted from both configs and deduplicated globally.

Output (both Git-ignored):
  artifacts/prepared/<manifest_id>_records.jsonl
  artifacts/prepared/<manifest_id>_manifest.json

Usage:
  uv run python scripts/prepare_canary.py \\
      --dataset-revision <commit-sha> \\
      --seed 42

  uv run python scripts/prepare_canary.py --help
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATASET_REPO = "ai4bharat/MSMARCO-XI"
CHUNK_STRATEGY = "passage_native"
CHUNK_STRATEGY_VERSION = "v1"

# NOTE: The ai4bharat/MSMARCO-XI dataset currently exposes a single 'default'
# config rather than per-language configs.  The config name must be confirmed
# against the pinned dataset revision before running the real canary.
# Pass --hf-config to override the default.
DEFAULT_HF_CONFIG = "default"

# Forbidden fields that must never appear in prepared records
FORBIDDEN_FIELDS = {"query", "Answer", "Eng_Query", "Eng_Answer", "query_type", "is_selected"}

# Default quotas for the initial English/Hindi/Bengali canary
DEFAULT_QUOTAS = {"en": 100, "hi": 100, "bn": 100}

# Source configs for the canary (these yield both native lang + English passages)
CANARY_SOURCE_CONFIGS = ["hi", "bn"]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Prepare offline canary records for MSMARCO-XI → Pinecone",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--dataset-revision",
        default=None,
        help="Pinned HuggingFace dataset commit hash (required for ready_for_write=true)",
    )
    p.add_argument("--split", default="train", choices=["train", "validation"])
    p.add_argument("--seed", type=int, default=42, help="Sampling seed for reproducibility")
    p.add_argument("--en-quota", type=int, default=100)
    p.add_argument("--hi-quota", type=int, default=100)
    p.add_argument("--bn-quota", type=int, default=100)
    p.add_argument(
        "--max-rows-per-config",
        type=int,
        default=5000,
        help="Max source rows to scan per config (cap for streaming datasets)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/prepared"),
        help="Directory for JSONL and manifest output",
    )
    p.add_argument("--tokenizer-revision", default=None, help="Pin HF tokenizer revision")
    p.add_argument(
        "--hf-config",
        default=DEFAULT_HF_CONFIG,
        help=(
            "HuggingFace dataset config name. "
            "ai4bharat/MSMARCO-XI currently uses 'default'; "
            "confirm against the pinned revision before running."
        ),
    )
    p.add_argument("--output-json", action="store_true", help="Print manifest JSON to stdout")
    return p


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _make_point_id(
    dataset_revision: str,
    language: str,
    content_hash: str,
    chunk_strategy_version: str,
    chunk_ordinal: int,
) -> str:
    namespace = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
    key = f"{dataset_revision}|{language}|{content_hash}|{chunk_strategy_version}|{chunk_ordinal}"
    return str(uuid.uuid5(namespace, key))


def _stream_config(
    config_lang: str,
    split: str,
    dataset_revision: str | None,
    max_rows: int,
    hf_config: str = DEFAULT_HF_CONFIG,
) -> list[dict]:
    """Stream up to max_rows records from the HF dataset config.  Returns raw HF records.

    NOTE: ai4bharat/MSMARCO-XI currently exposes a single 'default' config.
    The hf_config parameter must be verified against the pinned revision.
    Pass --hf-config to override.
    """
    from datasets import load_dataset

    logger.info(
        "Streaming hf_config=%s lang_filter=%s split=%s (max_rows=%d) …",
        hf_config,
        config_lang,
        split,
        max_rows,
    )
    ds = load_dataset(
        DATASET_REPO,
        hf_config,
        split=split,
        streaming=True,
        revision=dataset_revision,
    )
    rows = []
    for i, row in enumerate(ds):
        if i >= max_rows:
            break
        rows.append(row)
    logger.info("Collected %d rows for %s/%s", len(rows), config_lang, split)
    return rows


def _extract_passages(
    rows: list[dict],
    config_lang: str,
    split: str,
    dataset_revision: str,
) -> tuple[list[dict], list[dict]]:
    """Extract (native_passages, english_passages) from raw HF rows.

    Returns lists of dicts with provenance fields.  Forbidden fields are stripped.
    """
    from hhgoa_rag.dataset.parser import parse_record

    native: list[dict] = []
    english: list[dict] = []

    for row_idx, row in enumerate(rows):
        # Strip forbidden fields before any processing
        safe_row = {k: v for k, v in row.items() if k not in FORBIDDEN_FIELDS}
        occurrences, _ = parse_record(
            safe_row,
            config_language=config_lang,
            split=split,
            source_shard="0",
            source_row=row_idx,
            dataset_revision=dataset_revision,
        )
        for occ in occurrences:
            # Verify no forbidden fields leaked through
            record_info = {
                "config_language": occ.config_language,
                "language": occ.passage_language,
                "split": occ.split,
                "source_row": occ.source_row,
                "passage_position": occ.passage_position,
                "normalized_text": occ.normalized_text,
                "content_hash": _content_hash(occ.normalized_text),
                "dataset_revision": dataset_revision,
                "is_original_english": occ.is_original_english,
            }
            if occ.is_original_english:
                english.append(record_info)
            else:
                native.append(record_info)

    return native, english


def _reservoir_sample(items: list, k: int, seed: int) -> list:
    """Deterministic reservoir sampling."""
    rng = random.Random(seed)
    if len(items) <= k:
        result = list(items)
        rng.shuffle(result)
        return result
    return rng.sample(items, k)


def _build_prepared_record(
    rec: dict,
    dataset_revision: str,
    tok_wrapper: object,
    chunk_strategy_version: str,
    manifest_id: str,
) -> dict | None:
    """Build a fully-tokenized prepared record.  Returns None if text exceeds model limit."""
    from hhgoa_rag.ingestion.tokenizer import MODEL_INPUT_LIMIT

    text = rec["normalized_text"]
    token_length = tok_wrapper.count_tokens(text, add_prefix=True)  # type: ignore[attr-defined]

    if token_length > MODEL_INPUT_LIMIT:
        logger.warning(
            "Passage exceeds model limit (%d > %d tokens), skipping: %s…",
            token_length,
            MODEL_INPUT_LIMIT,
            text[:80],
        )
        return None

    point_id = _make_point_id(
        dataset_revision=dataset_revision,
        language=rec["language"],
        content_hash=rec["content_hash"],
        chunk_strategy_version=f"{CHUNK_STRATEGY}_{chunk_strategy_version}",
        chunk_ordinal=0,
    )

    return {
        "id": point_id,
        "chunk_text": text,
        "language": rec["language"],
        "config_language": rec["config_language"],
        "dataset_repo": DATASET_REPO,
        "dataset_revision": dataset_revision,
        "split": rec["split"],
        "physical_shard": "0",
        "local_source_row": rec["source_row"],
        "passage_position": rec["passage_position"],
        "parent_passage_id": rec["content_hash"],
        "content_hash": rec["content_hash"],
        "chunk_strategy": CHUNK_STRATEGY,
        "chunk_strategy_version": chunk_strategy_version,
        "chunk_ordinal": 0,
        "chunk_total": 1,
        "token_length": token_length,
        "tokenizer_fingerprint": tok_wrapper.fingerprint,  # type: ignore[attr-defined]
        "manifest_id": manifest_id,
    }


def main() -> None:
    args = _build_parser().parse_args()

    quotas = {"en": args.en_quota, "hi": args.hi_quota, "bn": args.bn_quota}

    dataset_revision = args.dataset_revision

    manifest_id = f"canary-{args.seed}-{hashlib.sha256((str(quotas) + str(args.seed) + str(dataset_revision)).encode()).hexdigest()[:8]}"

    logger.info("Manifest ID: %s", manifest_id)
    logger.info("Quotas: %s", quotas)
    logger.info("Dataset revision: %s", dataset_revision or "UNPINNED (not ready for write)")

    # Load tokenizer — fail closed if unavailable
    logger.info("Loading tokenizer …")
    from hhgoa_rag.ingestion.tokenizer import MODEL_INPUT_LIMIT, TOKENIZER_REPO, get_tokenizer

    tok = get_tokenizer(revision=args.tokenizer_revision)
    logger.info("Tokenizer loaded: %s (fingerprint=%s)", TOKENIZER_REPO, tok.fingerprint)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Collect raw passages from source configs
    all_native_hi: list[dict] = []
    all_native_bn: list[dict] = []
    all_english: list[dict] = []
    seen_en_hashes: set[str] = set()

    eff_revision = dataset_revision or "main"

    for config_lang in CANARY_SOURCE_CONFIGS:
        rows = _stream_config(
            config_lang,
            args.split,
            dataset_revision,
            args.max_rows_per_config,
            hf_config=args.hf_config,
        )
        native, english = _extract_passages(rows, config_lang, args.split, eff_revision)

        if config_lang == "hi":
            all_native_hi.extend(native)
        elif config_lang == "bn":
            all_native_bn.extend(native)

        # Global English dedup across configs
        for rec in english:
            if rec["content_hash"] not in seen_en_hashes:
                seen_en_hashes.add(rec["content_hash"])
                all_english.append(rec)

    logger.info(
        "Raw candidates: en=%d, hi=%d, bn=%d",
        len(all_english),
        len(all_native_hi),
        len(all_native_bn),
    )

    # Deduplicate within each language
    def _dedup(items: list[dict]) -> list[dict]:
        seen: set[str] = set()
        out: list[dict] = []
        for item in items:
            if item["content_hash"] not in seen:
                seen.add(item["content_hash"])
                out.append(item)
        return out

    all_native_hi = _dedup(all_native_hi)
    all_native_bn = _dedup(all_native_bn)

    # Deterministic sampling
    sampled_en = _reservoir_sample(all_english, quotas["en"], seed=args.seed)
    sampled_hi = _reservoir_sample(all_native_hi, quotas["hi"], seed=args.seed + 1)
    sampled_bn = _reservoir_sample(all_native_bn, quotas["bn"], seed=args.seed + 2)

    # Build prepared records with real tokenization
    prepared: list[dict] = []
    dedup_counts: dict[str, int] = {}
    rejection_counts: dict[str, int] = {}

    def process_lang(samples: list[dict], lang: str) -> list[dict]:
        out = []
        rejections = 0
        for rec in samples:
            built = _build_prepared_record(
                rec, eff_revision, tok, CHUNK_STRATEGY_VERSION, manifest_id
            )
            if built is None:
                rejections += 1
            else:
                out.append(built)
        if rejections:
            rejection_counts[lang] = rejections
        return out

    en_records = process_lang(sampled_en, "en")
    hi_records = process_lang(sampled_hi, "hi")
    bn_records = process_lang(sampled_bn, "bn")

    prepared = en_records + hi_records + bn_records

    # Compute actual per-language stats
    actual_counts: dict[str, int] = {}
    actual_tokens: dict[str, int] = {}
    for rec in prepared:
        lang = rec["language"]
        actual_counts[lang] = actual_counts.get(lang, 0) + 1
        actual_tokens[lang] = actual_tokens.get(lang, 0) + rec["token_length"]

    total_records = len(prepared)
    total_tokens = sum(r["token_length"] for r in prepared)

    # Forbidden field audit
    forbidden_found: list[str] = []
    for rec in prepared:
        bad = FORBIDDEN_FIELDS & set(rec.keys())
        if bad:
            forbidden_found.extend(sorted(bad))

    # Compute deterministic data checksum (excludes timestamps)
    data_for_checksum = json.dumps(
        [
            {k: v for k, v in r.items() if k != "manifest_id"}
            for r in sorted(prepared, key=lambda x: x["id"])
        ],
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    prepared_data_checksum = hashlib.sha256(data_for_checksum).hexdigest()

    # Write JSONL
    jsonl_path = args.output_dir / f"{manifest_id}_records.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in prepared:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    jsonl_checksum = _sha256_file(jsonl_path)
    jsonl_bytes = jsonl_path.stat().st_size

    # Projected indexed bytes (conservative estimate)
    projected_indexed_bytes = total_records * 1500

    # Starter budget projections
    starter_budget_ok = (
        total_records <= 10_000
        and total_tokens <= 4_000_000
        and projected_indexed_bytes <= int(1.5 * 1024 * 1024 * 1024)
    )

    # Determine ready_for_write
    readiness_failures: list[str] = []
    if not dataset_revision:
        readiness_failures.append("dataset_revision is unpinned")
    if actual_counts.get("en", 0) < quotas["en"]:
        readiness_failures.append(
            f"English quota not met: {actual_counts.get('en', 0)} < {quotas['en']}"
        )
    if actual_counts.get("hi", 0) < quotas["hi"]:
        readiness_failures.append(
            f"Hindi quota not met: {actual_counts.get('hi', 0)} < {quotas['hi']}"
        )
    if actual_counts.get("bn", 0) < quotas["bn"]:
        readiness_failures.append(
            f"Bengali quota not met: {actual_counts.get('bn', 0)} < {quotas['bn']}"
        )
    if forbidden_found:
        readiness_failures.append(f"Forbidden fields present: {sorted(set(forbidden_found))}")
    if not starter_budget_ok:
        readiness_failures.append("Starter budget projections exceeded")
    if rejection_counts:
        readiness_failures.append(f"Some records rejected (over token limit): {rejection_counts}")

    ready_for_write = len(readiness_failures) == 0

    created_at = datetime.now(UTC).isoformat()

    manifest: dict = {
        "manifest_schema_version": "1",
        "manifest_id": manifest_id,
        "mode": "canary",
        "dataset_repo": DATASET_REPO,
        "dataset_revision": dataset_revision,
        "source_configs": CANARY_SOURCE_CONFIGS,
        "hf_config": args.hf_config,
        "split": args.split,
        "sampling_algorithm": "reservoir",
        "sampling_seed": args.seed,
        "requested_quotas": quotas,
        "actual_per_language_records": actual_counts,
        "actual_per_language_tokens": actual_tokens,
        "total_records": total_records,
        "total_tokens": total_tokens,
        "prepared_data_bytes": jsonl_bytes,
        "projected_indexed_bytes": projected_indexed_bytes,
        "tokenizer_repo": TOKENIZER_REPO,
        "tokenizer_revision": args.tokenizer_revision or "HEAD",
        "tokenizer_fingerprint": tok.fingerprint,
        "model_input_limit": MODEL_INPUT_LIMIT,
        "chunk_strategy": CHUNK_STRATEGY,
        "chunk_strategy_version": CHUNK_STRATEGY_VERSION,
        "deduplication_counts": dedup_counts,
        "rejection_counts": rejection_counts,
        "forbidden_field_audit": "PASS"
        if not forbidden_found
        else f"FAIL: {sorted(set(forbidden_found))}",
        "prepared_record_path": str(jsonl_path.resolve()),
        "prepared_record_checksum": jsonl_checksum,
        "prepared_data_checksum": prepared_data_checksum,
        "starter_budget_projections": {
            "records_ok": total_records <= 10_000,
            "tokens_ok": total_tokens <= 4_000_000,
            "storage_ok": projected_indexed_bytes <= int(1.5 * 1024 * 1024 * 1024),
            "total_ok": starter_budget_ok,
        },
        "created_at": created_at,
        "ready_for_write": ready_for_write,
        "readiness_failures": readiness_failures,
    }

    # Manifest checksum (over content excluding the checksum field itself)
    manifest_for_checksum = {k: v for k, v in manifest.items() if k != "manifest_checksum"}
    manifest_checksum = hashlib.sha256(
        json.dumps(manifest_for_checksum, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    manifest["manifest_checksum"] = manifest_checksum

    manifest_path = args.output_dir / f"{manifest_id}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    if args.output_json:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
    else:
        print("\nCanary preparation complete")
        print(f"  Manifest ID  : {manifest_id}")
        print(
            f"  Records      : {total_records} (en={actual_counts.get('en',0)}, hi={actual_counts.get('hi',0)}, bn={actual_counts.get('bn',0)})"
        )
        print(f"  Total tokens : {total_tokens:,}")
        print(f"  JSONL path   : {jsonl_path}")
        print(f"  Manifest     : {manifest_path}")
        print(
            f"  Ready        : {'YES' if ready_for_write else 'NO — ' + '; '.join(readiness_failures)}"
        )

    sys.exit(0 if ready_for_write else 1)


if __name__ == "__main__":
    main()
