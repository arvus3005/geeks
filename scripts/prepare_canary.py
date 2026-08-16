#!/usr/bin/env python3
"""Offline canary preparation — no Pinecone imports, no provider calls.

Generates the exact immutable records that live ingestion will consume.
Live ingestion MUST read from the prepared JSONL; it must not independently
stream a new dataset subset.

Default: 300 records — 100 English, 100 Hindi, 100 Bengali.
Source Parquet files:
  Hindi train   : train/hintrain.parquet
  Hindi val     : validation/hinval.parquet
  Bengali train : train/bentrain.parquet
  Bengali val   : validation/benval.parquet
  Repo          : https://huggingface.co/datasets/ai4bharat/MSMARCO-XI

Dataset revision is resolved to a full immutable commit SHA before any data is
read, ensuring byte-for-byte reproducibility. Pass --dataset-revision to pin.

Selection is deterministic: passages are assigned a stable key derived from
(seed, language, content_hash, physical_source, local_source_row,
passage_position, chunk_ordinal), sorted by that key, and the first N are
selected.  Reservoir sampling is NOT used.

Output (both Git-ignored):
  artifacts/prepared/<manifest_id>_records.jsonl
  artifacts/prepared/<manifest_id>_manifest.json

Usage:
  uv run python scripts/prepare_canary.py \\
      --dataset-revision <full-commit-sha> \\
      --seed 42

  uv run python scripts/prepare_canary.py --help
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATASET_REPO = "ai4bharat/MSMARCO-XI"
CHUNK_STRATEGY = "passage_native"
CHUNK_STRATEGY_VERSION = "v1"

# Forbidden fields that must never appear in prepared records
FORBIDDEN_FIELDS = {"query", "Answer", "Eng_Query", "Eng_Answer", "query_type", "is_selected"}

# Default quotas for the initial English/Hindi/Bengali canary
DEFAULT_QUOTAS = {"en": 100, "hi": 100, "bn": 100}

# Source configs for the canary (these yield both native lang + English passages)
CANARY_SOURCE_CONFIGS = ["hi", "bn"]

# Parquet file paths within the HF dataset repo, keyed by (config_lang, split)
_PARQUET_FILES: dict[tuple[str, str], str] = {
    ("hi", "train"): "train/hintrain.parquet",
    ("hi", "validation"): "validation/hinval.parquet",
    ("bn", "train"): "train/bentrain.parquet",
    ("bn", "validation"): "validation/benval.parquet",
}

# Expected language values in translated passages for each config
_EXPECTED_NATIVE_LANG: dict[str, str] = {"hi": "hi", "bn": "bn"}

# MSMARCO-XI dataset schema: fields present in every record
_EXPECTED_SCHEMA_FIELDS = {"passages"}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Prepare offline canary records for MSMARCO-XI → Pinecone",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--dataset-revision",
        default=None,
        help=(
            "Pinned HuggingFace dataset commit hash (full SHA). "
            "Required for ready_for_write=true. "
            "When omitted, the current HEAD is resolved and logged."
        ),
    )
    p.add_argument("--split", default="train", choices=["train", "validation"])
    p.add_argument("--seed", type=int, default=42, help="Deterministic selection seed")
    p.add_argument("--en-quota", type=int, default=100)
    p.add_argument("--hi-quota", type=int, default=100)
    p.add_argument("--bn-quota", type=int, default=100)
    p.add_argument(
        "--max-rows-per-config",
        type=int,
        default=5000,
        help="Max source rows to scan per Parquet file (cap for streaming)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/prepared"),
        help="Directory for JSONL and manifest output",
    )
    p.add_argument(
        "--tokenizer-revision",
        default=None,
        help="Pin HF tokenizer commit SHA (full SHA required for ready_for_write=true)",
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


def _resolve_dataset_revision(pinned_revision: str | None) -> str:
    """Resolve dataset revision to a full immutable commit SHA.

    If pinned_revision is given (full SHA), verify it looks like a SHA and return it.
    Otherwise resolve HEAD via huggingface_hub and return the resolved SHA.
    Never returns 'HEAD', 'main', or other symbolic refs.
    """
    try:
        from huggingface_hub import list_repo_commits
    except ImportError as e:
        raise RuntimeError(
            "huggingface_hub is required to resolve dataset revision. "
            "Install with: uv add huggingface-hub"
        ) from e

    if pinned_revision is not None:
        if len(pinned_revision) < 7:
            raise ValueError(
                f"dataset_revision looks too short to be a commit SHA: {pinned_revision!r}. "
                "Pass the full 40-character hex SHA."
            )
        logger.info("Using pinned dataset revision: %s", pinned_revision)
        return pinned_revision

    # Resolve HEAD
    logger.info("Resolving HEAD for %s …", DATASET_REPO)
    try:
        commits = list(list_repo_commits(DATASET_REPO, repo_type="dataset"))
        if not commits:
            raise RuntimeError(f"No commits found for {DATASET_REPO}")
        resolved = commits[0].commit_id
    except Exception as e:
        raise RuntimeError(
            f"Cannot resolve HEAD for {DATASET_REPO}. " "Pass --dataset-revision with a pinned SHA."
        ) from e

    logger.info("Resolved HEAD → %s", resolved)
    return resolved


def _parquet_url(dataset_revision: str, parquet_path: str) -> str:
    """Build a revision-pinned HF Parquet URL."""
    return (
        f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/{dataset_revision}/{parquet_path}"
    )


def _preflight_schema(row: dict, config_lang: str, parquet_path: str) -> None:
    """Validate that the first row from a Parquet file has the expected schema."""
    missing = _EXPECTED_SCHEMA_FIELDS - set(row.keys())
    if missing:
        raise RuntimeError(
            f"Schema preflight failed for {parquet_path}: missing fields {missing}. "
            f"Available fields: {list(row.keys())}"
        )
    # Validate passages structure
    passages = row.get("passages")
    if not isinstance(passages, dict):
        raise RuntimeError(
            f"Schema preflight failed for {parquet_path}: "
            f"'passages' field is {type(passages).__name__}, expected dict."
        )
    for key in ("English_passages", "Translated_passages"):
        if key not in passages:
            raise RuntimeError(
                f"Schema preflight failed for {parquet_path}: "
                f"passages missing '{key}' key. Found: {list(passages.keys())}"
            )
    logger.info("Schema preflight PASSED for %s (config_lang=%s)", parquet_path, config_lang)


def _stream_parquet(
    config_lang: str,
    split: str,
    dataset_revision: str,
    max_rows: int,
) -> tuple[list[dict], str]:
    """Stream up to max_rows records from the revision-pinned Parquet file.

    Returns (rows, physical_parquet_path).
    Raises RuntimeError if the config_lang+split combination is unsupported.
    """
    from datasets import load_dataset

    parquet_path = _PARQUET_FILES.get((config_lang, split))
    if parquet_path is None:
        supported = sorted(_PARQUET_FILES.keys())
        raise RuntimeError(
            f"No Parquet file configured for config_lang={config_lang!r}, split={split!r}. "
            f"Supported combinations: {supported}"
        )

    url = _parquet_url(dataset_revision, parquet_path)
    logger.info(
        "Streaming Parquet: config_lang=%s split=%s revision=%s path=%s (max_rows=%d)",
        config_lang,
        split,
        dataset_revision,
        parquet_path,
        max_rows,
    )

    try:
        ds = load_dataset(
            "parquet",
            data_files=url,
            split="train",  # 'parquet' loader always uses split='train'
            streaming=True,
        )
    except Exception as e:
        raise RuntimeError(
            f"Cannot stream Parquet file {url}. "
            f"Check HuggingFace access and dataset revision. Original error: {e}"
        ) from e

    rows = []
    schema_checked = False
    for i, row in enumerate(ds):
        if i >= max_rows:
            break
        if not schema_checked:
            _preflight_schema(row, config_lang, parquet_path)
            schema_checked = True
        rows.append(row)

    logger.info("Collected %d rows from %s (config_lang=%s)", len(rows), parquet_path, config_lang)
    return rows, parquet_path


def _validate_language_metadata(
    rows: list[dict],
    config_lang: str,
    parquet_path: str,
) -> None:
    """Spot-check that translated passages look like the expected language.

    This is a heuristic guard — not a full language detection pass.
    Fails closed if every translated passage is empty or missing.
    """
    non_empty = 0
    for row in rows[:20]:  # sample first 20 rows
        passages = row.get("passages", {})
        trans = passages.get("Translated_passages", [])
        if isinstance(trans, list):
            for t in trans:
                if isinstance(t, str) and len(t.strip()) > 5:
                    non_empty += 1
                    break

    if non_empty == 0 and len(rows) > 0:
        raise RuntimeError(
            f"Language metadata validation failed for {parquet_path}: "
            f"no non-empty translated passages found in first {min(20, len(rows))} rows. "
            f"Expected config_lang={config_lang!r}. Parquet file may be wrong."
        )
    logger.info(
        "Language metadata validation PASSED: %d/%d sampled rows have non-empty translated passages",
        non_empty,
        min(20, len(rows)),
    )


def _extract_passages(
    rows: list[dict],
    config_lang: str,
    split: str,
    dataset_revision: str,
    parquet_path: str,
) -> tuple[list[dict], list[dict]]:
    """Extract (native_passages, english_passages) from raw Parquet rows.

    Records physical_source (parquet_path) and local_source_row for stable IDs.
    Forbidden fields are stripped. Never prints passage content.
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
            record_info = {
                "config_language": occ.config_language,
                "language": occ.passage_language,
                "split": occ.split,
                "physical_source": parquet_path,
                "local_source_row": occ.source_row,
                "passage_position": occ.passage_position,
                "chunk_ordinal": 0,
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


def _stable_selection_key(seed: int, lang: str, rec: dict) -> str:
    """Compute stable, deterministic selection key for a passage record.

    Key components:
      seed, language, content_hash, physical_source, local_source_row,
      passage_position, chunk_ordinal
    """
    components = "|".join(
        [
            str(seed),
            lang,
            rec["content_hash"],
            rec["physical_source"],
            str(rec["local_source_row"]),
            str(rec["passage_position"]),
            str(rec["chunk_ordinal"]),
        ]
    )
    return hashlib.sha256(components.encode("utf-8")).hexdigest()


def _deterministic_select(items: list[dict], k: int, seed: int, lang: str) -> list[dict]:
    """Select exactly k items deterministically.

    Assigns each item a stable SHA-256 key, sorts by key, returns first k.
    This is NOT reservoir sampling — it is purely deterministic given the same inputs.
    """
    if not items:
        return []
    keyed = [(item, _stable_selection_key(seed, lang, item)) for item in items]
    keyed.sort(key=lambda t: t[1])
    return [item for item, _ in keyed[:k]]


def _check_sources_differ(
    hi_path: str,
    bn_path: str,
) -> None:
    """Fail closed if Hindi and Bengali resolve to the same physical Parquet file."""
    if hi_path == bn_path:
        raise RuntimeError(
            f"Hindi and Bengali Parquet files resolved to the same physical source: {hi_path!r}. "
            "This indicates a misconfiguration. Aborting to prevent corpus contamination."
        )
    logger.info("Source identity check PASSED: hi=%s, bn=%s (different files)", hi_path, bn_path)


def _build_prepared_record(
    rec: dict,
    dataset_revision: str,
    tok_wrapper: object,
    chunk_strategy_version: str,
    manifest_id: str,
) -> dict | None:
    """Build a fully-tokenized prepared record.

    Returns None if text exceeds model limit after exhausting token-safe splits.
    Never prints passage content.
    """
    from hhgoa_rag.ingestion.tokenizer import MODEL_INPUT_LIMIT

    text = rec["normalized_text"]
    token_length = tok_wrapper.count_tokens(text, add_prefix=True)  # type: ignore[attr-defined]

    if token_length > MODEL_INPUT_LIMIT:
        logger.warning(
            "Passage exceeds model limit (%d > %d tokens) at physical_source=%s row=%d pos=%d — skipping",
            token_length,
            MODEL_INPUT_LIMIT,
            rec["physical_source"],
            rec["local_source_row"],
            rec["passage_position"],
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
        "physical_source": rec["physical_source"],
        "physical_shard": "0",
        "local_source_row": rec["local_source_row"],
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

    # Step 1: Resolve dataset revision to a full immutable SHA
    dataset_revision = _resolve_dataset_revision(args.dataset_revision)
    is_pinned = args.dataset_revision is not None

    # Fingerprint: seed + quotas + dataset_revision (canonical configuration)
    fingerprint_src = json.dumps(
        {
            "quotas": quotas,
            "seed": args.seed,
            "dataset_revision": dataset_revision,
            "split": args.split,
        },
        sort_keys=True,
    )
    manifest_id = (
        f"canary-{args.seed}-" f"{hashlib.sha256(fingerprint_src.encode()).hexdigest()[:8]}"
    )

    logger.info("Manifest ID: %s", manifest_id)
    logger.info("Quotas: %s", quotas)
    logger.info(
        "Dataset revision: %s (%s)", dataset_revision, "pinned" if is_pinned else "resolved HEAD"
    )

    # Step 2: Load tokenizer — fail closed if unavailable
    logger.info("Loading tokenizer …")
    from hhgoa_rag.ingestion.tokenizer import MODEL_INPUT_LIMIT, TOKENIZER_REPO, get_tokenizer

    tok = get_tokenizer(revision=args.tokenizer_revision)
    logger.info("Tokenizer loaded: %s (fingerprint=%s)", TOKENIZER_REPO, tok.fingerprint)

    # Refuse HEAD/main/unknown in tokenizer revision for ready_for_write
    tok_revision_str = args.tokenizer_revision or "HEAD"
    tokenizer_revision_pinned = (
        args.tokenizer_revision is not None
        and args.tokenizer_revision not in ("HEAD", "main", "unknown")
        and len(args.tokenizer_revision) >= 7
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Step 3: Stream Parquet files for each source config
    all_native_hi: list[dict] = []
    all_native_bn: list[dict] = []
    all_english: list[dict] = []
    seen_en_hashes: set[str] = set()

    hi_parquet_path: str = ""
    bn_parquet_path: str = ""

    for config_lang in CANARY_SOURCE_CONFIGS:
        rows, parquet_path = _stream_parquet(
            config_lang,
            args.split,
            dataset_revision,
            args.max_rows_per_config,
        )

        # Validate language metadata
        _validate_language_metadata(rows, config_lang, parquet_path)

        native, english = _extract_passages(
            rows, config_lang, args.split, dataset_revision, parquet_path
        )

        if config_lang == "hi":
            all_native_hi.extend(native)
            hi_parquet_path = parquet_path
        elif config_lang == "bn":
            all_native_bn.extend(native)
            bn_parquet_path = parquet_path

        # Global English chunk deduplication across both physical files
        for rec in english:
            if rec["content_hash"] not in seen_en_hashes:
                seen_en_hashes.add(rec["content_hash"])
                all_english.append(rec)

    # Step 4: Fail closed if Hindi and Bengali resolve to the same source
    _check_sources_differ(hi_parquet_path, bn_parquet_path)

    logger.info(
        "Raw candidates: en=%d, hi=%d, bn=%d",
        len(all_english),
        len(all_native_hi),
        len(all_native_bn),
    )

    # Step 5: Within-language deduplication
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

    # Step 6: Deterministic selection (sort by stable key, take first N)
    sampled_en = _deterministic_select(all_english, quotas["en"], seed=args.seed, lang="en")
    sampled_hi = _deterministic_select(all_native_hi, quotas["hi"], seed=args.seed, lang="hi")
    sampled_bn = _deterministic_select(all_native_bn, quotas["bn"], seed=args.seed, lang="bn")

    # Step 7: Build prepared records with real tokenization
    prepared: list[dict] = []
    rejection_counts: dict[str, int] = {}

    def process_lang(samples: list[dict], lang: str) -> list[dict]:
        out = []
        rejections = 0
        for rec in samples:
            built = _build_prepared_record(
                rec, dataset_revision, tok, CHUNK_STRATEGY_VERSION, manifest_id
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

    # Step 8: Compute per-language stats
    actual_counts: dict[str, int] = {}
    actual_tokens: dict[str, int] = {}
    for rec in prepared:
        lang = rec["language"]
        actual_counts[lang] = actual_counts.get(lang, 0) + 1
        actual_tokens[lang] = actual_tokens.get(lang, 0) + rec["token_length"]

    total_records = len(prepared)
    total_tokens = sum(r["token_length"] for r in prepared)

    # Step 9: Forbidden field audit
    forbidden_found: list[str] = []
    for rec in prepared:
        bad = FORBIDDEN_FIELDS & set(rec.keys())
        if bad:
            forbidden_found.extend(sorted(bad))

    # Step 10: Compute deterministic data checksum (excludes timestamps)
    data_for_checksum = json.dumps(
        [
            {k: v for k, v in r.items() if k != "manifest_id"}
            for r in sorted(prepared, key=lambda x: x["id"])
        ],
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    prepared_data_checksum = hashlib.sha256(data_for_checksum).hexdigest()

    # Step 11: Atomic write — temp file + rename
    jsonl_path = args.output_dir / f"{manifest_id}_records.jsonl"
    jsonl_tmp = jsonl_path.with_suffix(".jsonl.tmp")
    try:
        with open(jsonl_tmp, "w", encoding="utf-8") as f:
            for rec in prepared:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        jsonl_tmp.rename(jsonl_path)
    except Exception:
        jsonl_tmp.unlink(missing_ok=True)
        raise

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

    # Step 12: Determine ready_for_write
    readiness_failures: list[str] = []
    if not is_pinned:
        readiness_failures.append(
            "dataset_revision was not pinned by caller (resolved HEAD — not reproducible)"
        )
    if not tokenizer_revision_pinned:
        readiness_failures.append(
            f"tokenizer_revision is not a pinned SHA: {tok_revision_str!r} "
            "(pass --tokenizer-revision <full-sha>)"
        )
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
        "manifest_schema_version": "2",
        "manifest_id": manifest_id,
        "mode": "canary",
        "dataset_repo": DATASET_REPO,
        "dataset_revision": dataset_revision,
        "dataset_revision_pinned": is_pinned,
        "source_configs": CANARY_SOURCE_CONFIGS,
        "parquet_sources": {
            "hi": hi_parquet_path,
            "bn": bn_parquet_path,
        },
        "split": args.split,
        "selection_algorithm": "stable_key_sort",
        "sampling_seed": args.seed,
        "requested_quotas": quotas,
        "actual_per_language_records": actual_counts,
        "actual_per_language_tokens": actual_tokens,
        "total_records": total_records,
        "total_tokens": total_tokens,
        "prepared_data_bytes": jsonl_bytes,
        "projected_indexed_bytes": projected_indexed_bytes,
        "tokenizer_repo": TOKENIZER_REPO,
        "tokenizer_revision": tok_revision_str,
        "tokenizer_revision_pinned": tokenizer_revision_pinned,
        "tokenizer_fingerprint": tok.fingerprint,
        "model_input_limit": MODEL_INPUT_LIMIT,
        "chunk_strategy": CHUNK_STRATEGY,
        "chunk_strategy_version": CHUNK_STRATEGY_VERSION,
        "rejection_counts": rejection_counts,
        "forbidden_field_audit": (
            "PASS" if not forbidden_found else f"FAIL: {sorted(set(forbidden_found))}"
        ),
        "prepared_record_path": str(jsonl_path),
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

    # Atomic manifest write
    manifest_path = args.output_dir / f"{manifest_id}_manifest.json"
    manifest_tmp = manifest_path.with_suffix(".json.tmp")
    try:
        manifest_tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        manifest_tmp.rename(manifest_path)
    except Exception:
        manifest_tmp.unlink(missing_ok=True)
        raise

    if args.output_json:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
    else:
        print("\nCanary preparation complete")
        print(f"  Manifest ID  : {manifest_id}")
        print(
            f"  Records      : {total_records} "
            f"(en={actual_counts.get('en', 0)}, "
            f"hi={actual_counts.get('hi', 0)}, "
            f"bn={actual_counts.get('bn', 0)})"
        )
        print(f"  Total tokens : {total_tokens:,}")
        print(f"  JSONL path   : {jsonl_path}")
        print(f"  Manifest     : {manifest_path}")
        print(
            f"  Ready        : "
            f"{'YES' if ready_for_write else 'NO — ' + '; '.join(readiness_failures)}"
        )

    sys.exit(0 if ready_for_write else 1)


if __name__ == "__main__":
    main()
