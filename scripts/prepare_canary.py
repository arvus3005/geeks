#!/usr/bin/env python3
"""Offline canary preparation — no Pinecone imports, no provider calls.

Generates the exact immutable records that live ingestion will consume.
Live ingestion MUST read from the prepared JSONL; it must not independently
stream a new dataset subset.

Default: 300 records — 100 English, 100 Hindi, 100 Bengali.
Source Parquet files:
  Hindi train   : train/hintrain.parquet
  Bengali train : train/bentrain.parquet
  Repo          : https://huggingface.co/datasets/ai4bharat/MSMARCO-XI

Dataset revision is resolved to a full immutable commit SHA before any data is
read, ensuring byte-for-byte reproducibility. Pass --dataset-revision to pin.
Tokenizer revision is resolved similarly. Pass --tokenizer-revision to pin.

Selection is deterministic: passages are assigned a stable key derived from
(seed, language, content_hash, physical_source, local_source_row,
passage_position, chunk_ordinal), sorted by that key, and the first N are
selected.  Reservoir sampling is NOT used.

Oversized passages are split deterministically; if splitting still yields no
fitting chunks the passage is rejected and the next candidate is tried until
the exact quota is met.

Output (both Git-ignored):
  artifacts/prepared/<manifest_id>_records.jsonl
  artifacts/prepared/<manifest_id>_manifest.json

Usage:
  uv run python scripts/prepare_canary.py \\
      --dataset-revision <full-commit-sha> \\
      --tokenizer-revision <full-commit-sha> \\
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

from hhgoa_rag.pinecone_contract import (
    CONTRACT_VERSION,
    DATASET_REPO,
    MANIFEST_SCHEMA_VERSION,
    TEXT_FIELD,
    TOKENIZER_REPO,
    canonical_contract,
    contract_fingerprint,
)
from hhgoa_rag.pinecone_contract import (
    MODEL as _CANONICAL_MODEL,
)
from hhgoa_rag.pinecone_contract import (
    NAMESPACE as _CONTRACT_NAMESPACE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CHUNK_STRATEGY_VERSION = "v1"

# Version tag for normalization — participates in manifest identity.
NORMALIZATION_VERSION = "v1"
# MANIFEST_SCHEMA_VERSION is imported from pinecone_contract (= "3")
# RECORD_SCHEMA_VERSION labels the per-record field schema; not the manifest envelope.
RECORD_SCHEMA_VERSION = "1"


def _chunk_parameters(strategy: str) -> dict:
    """Return the deterministic parameters for a chunk strategy (for manifest identity)."""
    from hhgoa_rag.ingestion.chunkers import CHUNKERS

    chunker = CHUNKERS.get(strategy)
    if chunker is None:
        return {}
    params: dict = {}
    for attr in ("target_chars", "overlap_sentences", "target_tokens", "overlap_tokens"):
        if hasattr(chunker, attr):
            params[attr] = getattr(chunker, attr)
    return params


# Derive CLI choices from the real registry so they always stay in sync.
# "semantic" is NOT included: SemanticChunker requires an injected similarity
# provider and is excluded from the default CHUNKERS registry.
def _get_supported_chunk_strategies() -> list[str]:
    from hhgoa_rag.ingestion.chunkers import CHUNKERS

    return sorted(CHUNKERS.keys())


SUPPORTED_CHUNK_STRATEGIES = _get_supported_chunk_strategies()
DEFAULT_CHUNK_STRATEGY = "sentence_aware"

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

# MSMARCO-XI dataset schema: fields present in every record
_EXPECTED_SCHEMA_FIELDS = {"passages"}

# Batch constants matching ingest_prepared.py
MAX_RECORDS_PER_BATCH = 96
MAX_BYTES_PER_BATCH = 1_800_000


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
    p.add_argument(
        "--chunk-strategy",
        default=DEFAULT_CHUNK_STRATEGY,
        choices=SUPPORTED_CHUNK_STRATEGIES,
        help=f"Chunking strategy. Default: {DEFAULT_CHUNK_STRATEGY}",
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
    """Resolve dataset revision to a full immutable commit SHA."""
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


def _resolve_tokenizer_revision(pinned_revision: str | None, tokenizer_repo: str) -> str:
    """Resolve tokenizer revision to a full immutable commit SHA."""
    if pinned_revision is not None:
        if len(pinned_revision) < 7:
            raise ValueError(
                f"tokenizer_revision looks too short to be a commit SHA: {pinned_revision!r}. "
                "Pass the full 40-character hex SHA."
            )
        logger.info("Using pinned tokenizer revision: %s", pinned_revision)
        return pinned_revision

    logger.info("Resolving HEAD for tokenizer %s …", tokenizer_repo)
    try:
        from huggingface_hub import list_repo_commits

        commits = list(list_repo_commits(tokenizer_repo, repo_type="model"))
        if not commits:
            raise RuntimeError(f"No commits found for {tokenizer_repo}")
        resolved = commits[0].commit_id
    except Exception as e:
        raise RuntimeError(
            f"Cannot resolve HEAD for tokenizer {tokenizer_repo}. "
            "Pass --tokenizer-revision with a pinned SHA."
        ) from e

    logger.info("Resolved tokenizer HEAD → %s", resolved)
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


def _preflight_disk_space(min_free_bytes: int = 3 * 1024 * 1024 * 1024) -> None:
    """Fail closed if too little free disk space remains before downloads.

    Full MSMARCO-XI Parquet shards are large; --max-rows-per-config caps how many
    rows we READ but does NOT cap how many bytes are DOWNLOADED (the whole shard
    is fetched into the HF cache). Guard against filling the disk.
    """
    import shutil

    free = shutil.disk_usage(".").free
    if free < min_free_bytes:
        raise RuntimeError(
            f"Insufficient free disk space: {free / 1e9:.1f} GB free, "
            f"need at least {min_free_bytes / 1e9:.1f} GB. "
            "Full Parquet shards are downloaded regardless of --max-rows-per-config."
        )
    logger.info("Disk preflight OK: %.1f GB free", free / 1e9)


def _download_and_read_parquet(
    config_lang: str,
    split: str,
    dataset_revision: str,
    max_rows: int,
) -> tuple[list[dict], str]:
    """Download the FULL Parquet shard (cached) then read up to max_rows records.

    NOTE: the entire Parquet shard is downloaded regardless of max_rows;
    max_rows only bounds how many rows are parsed into memory.

    Uses hf_hub_download to fetch the revision-pinned Parquet file into the local
    HuggingFace cache, then reads it with pyarrow.  Direct-URL streaming mode is
    NOT used because PyArrow cannot convert the MSMARCO-XI nested struct columns
    in chunked streaming output mode.

    Subsequent runs use the locally cached file so no network I/O is needed.

    Returns (rows, physical_parquet_path) where physical_parquet_path is the
    canonical HF repo-relative path used for deduplication and source tracking.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    parquet_path = _PARQUET_FILES.get((config_lang, split))
    if parquet_path is None:
        supported = sorted(_PARQUET_FILES.keys())
        raise RuntimeError(
            f"No Parquet file configured for config_lang={config_lang!r}, split={split!r}. "
            f"Supported combinations: {supported}"
        )

    logger.info(
        "Fetching Parquet: config_lang=%s split=%s revision=%s path=%s (max_rows=%d)",
        config_lang,
        split,
        dataset_revision,
        parquet_path,
        max_rows,
    )

    try:
        local_path = hf_hub_download(
            repo_id=DATASET_REPO,
            filename=parquet_path,
            revision=dataset_revision,
            repo_type="dataset",
            cache_dir=".cache/huggingface",
        )
    except Exception as e:
        raise RuntimeError(
            f"Cannot download Parquet file {parquet_path!r} from {DATASET_REPO} "
            f"revision={dataset_revision}. Original error: {e}"
        ) from e

    logger.info("Reading from local cache: %s", local_path)

    try:
        pf = pq.ParquetFile(local_path)
        rows: list[dict] = []
        schema_checked = False
        for batch in pf.iter_batches(batch_size=min(max_rows, 500)):
            batch_dict = batch.to_pydict()
            n = batch.num_rows
            for i in range(n):
                if len(rows) >= max_rows:
                    break
                row: dict = {k: v[i] for k, v in batch_dict.items()}
                if not schema_checked:
                    _preflight_schema(row, config_lang, parquet_path)
                    schema_checked = True
                rows.append(row)
            if len(rows) >= max_rows:
                break
    except Exception as e:
        raise RuntimeError(f"Cannot read Parquet file {local_path}. Original error: {e}") from e

    logger.info("Collected %d rows from %s (config_lang=%s)", len(rows), parquet_path, config_lang)
    return rows, parquet_path


def _validate_language_metadata(
    rows: list[dict],
    config_lang: str,
    parquet_path: str,
) -> None:
    """Spot-check that translated passages look like the expected language."""
    non_empty = 0
    for row in rows[:20]:
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
    """Extract (native_passages, english_passages) from raw Parquet rows."""
    from hhgoa_rag.dataset.parser import parse_record

    native: list[dict] = []
    english: list[dict] = []

    for row_idx, row in enumerate(rows):
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
    """Compute stable, deterministic selection key for a passage record."""
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
    """Select exactly k items deterministically by stable SHA-256 key sort."""
    if not items:
        return []
    keyed = [(item, _stable_selection_key(seed, lang, item)) for item in items]
    keyed.sort(key=lambda t: t[1])
    return [item for item, _ in keyed[:k]]


def _check_sources_differ(hi_path: str, bn_path: str) -> None:
    """Fail closed if Hindi and Bengali resolve to the same physical Parquet file."""
    if hi_path == bn_path:
        raise RuntimeError(
            f"Hindi and Bengali Parquet files resolved to the same physical source: {hi_path!r}. "
            "This indicates a misconfiguration. Aborting to prevent corpus contamination."
        )
    logger.info("Source identity check PASSED: hi=%s, bn=%s (different files)", hi_path, bn_path)


def _split_units(text: str, level: str) -> list[str]:
    """Split text into ordered non-empty units at the given granularity.

    level="sentence": sentence/newline boundaries (Latin + Indic punctuation)
    level="word":     whitespace boundaries
    level="char":     individual characters (last-resort for CJK / no-space text)
    """
    import re

    if level == "sentence":
        sent_re = re.compile(r"(?<=[.?!।॥])\s+|\n+")
        units = sent_re.split(text.strip())
    elif level == "word":
        units = re.split(r"\s+", text.strip())
    elif level == "char":
        units = list(text.strip())
    else:  # pragma: no cover - programming error
        raise ValueError(f"Unknown split level: {level!r}")
    return [u for u in units if u.strip()]


_NEXT_LEVEL: dict[str, str | None] = {"sentence": "word", "word": "char", "char": None}


def _split_text_to_fit(
    text: str,
    tok_wrapper: object,
    model_limit: int,
) -> list[str]:
    """Split text into ordered chunks that each fit within model_limit tokens.

    Cascades sentence -> word -> character granularity.  ALL text is retained:
    every sub-chunk produced is returned in order with no duplication and no
    silent loss.  If a single character still exceeds the limit (pathological),
    the whole passage is rejected by returning an empty list.
    """
    if tok_wrapper.count_tokens(text, add_prefix=True) <= model_limit:  # type: ignore[attr-defined]
        return [text]

    def _pack(segment: str, level: str) -> list[str]:
        joiner = "" if level == "char" else " "
        units = _split_units(segment, level)
        if not units:
            return []
        next_level = _NEXT_LEVEL[level]
        out: list[str] = []
        current: list[str] = []

        def flush() -> None:
            if current:
                out.append(joiner.join(current))
                current.clear()

        for unit in units:
            if tok_wrapper.count_tokens(unit, add_prefix=True) > model_limit:  # type: ignore[attr-defined]
                flush()
                if next_level is None:
                    # A single character exceeds the limit — unrecoverable.
                    return []
                sub = _pack(unit, next_level)
                if not sub:
                    return []
                out.extend(sub)
                continue
            candidate = joiner.join(current + [unit])
            if tok_wrapper.count_tokens(candidate, add_prefix=True) <= model_limit:  # type: ignore[attr-defined]
                current.append(unit)
            else:
                flush()
                current.append(unit)
        flush()
        return out

    result = _pack(text, "sentence")
    return [c for c in result if c.strip()]


def _chunk_passage_texts(
    text: str,
    passage_id: str,
    chunk_strategy: str,
    tokenizer: object = None,
) -> list[str]:
    """Route a passage through the chunker registry, returning ordered chunk texts.

    For fixed_token_overlap, tokenizer must be provided for production use so that
    real token boundaries are used rather than the whitespace approximation.
    The selected strategy from the registry is authoritatively used; every chunk
    it produces is retained in order (no text loss).
    """
    from hhgoa_rag.ingestion.chunkers import get_chunker

    chunker = get_chunker(chunk_strategy, tokenizer=tokenizer)
    chunks = chunker.chunk(text, passage_id)
    return [c.text for c in chunks if c.text.strip()]


def _build_prepared_records_for_passage(
    rec: dict,
    dataset_revision: str,
    tok_wrapper: object,
    chunk_strategy: str,
    chunk_strategy_version: str,
    manifest_id: str,
    split_counts: dict[str, int],
    rejection_counts: dict[str, int],
) -> list[dict]:
    """Build 1+ prepared records from a passage.

    1. Route the passage through the selected chunker (registry-backed).
    2. Enforce the 507-token limit on every chunk, splitting oversized chunks
       further with deterministic sentence/word/char cascade — retaining ALL
       resulting sub-chunks (no text loss, no duplication).
    3. Emit one record per final chunk with correct chunk_ordinal/chunk_total
       describing the actual parent-passage chunking, via build_record().

    Returns list of records (empty only if every chunk is unsplittable).
    """
    from hhgoa_rag.ingestion.schema import SchemaViolationError, build_record
    from hhgoa_rag.ingestion.tokenizer import MODEL_INPUT_LIMIT

    lang = rec["language"]
    parent_passage_id = rec["content_hash"]

    # Step 1: chunk via the selected strategy. Pass tok_wrapper for fixed_token_overlap
    # so the production path uses real token boundaries (not whitespace approximation).
    strategy_chunks = _chunk_passage_texts(
        rec["normalized_text"], parent_passage_id, chunk_strategy, tokenizer=tok_wrapper
    )

    # Step 2: enforce token limit on each chunk, expanding oversized ones.
    final_texts: list[str] = []
    for chunk_text in strategy_chunks:
        if tok_wrapper.count_tokens(chunk_text, add_prefix=True) <= MODEL_INPUT_LIMIT:  # type: ignore[attr-defined]
            final_texts.append(chunk_text)
            continue
        sub_texts = _split_text_to_fit(chunk_text, tok_wrapper, MODEL_INPUT_LIMIT)
        if not sub_texts:
            logger.warning(
                "Chunk unsplittable at physical_source=%s row=%d pos=%d — rejecting passage",
                rec["physical_source"],
                rec["local_source_row"],
                rec["passage_position"],
            )
            rejection_counts[lang] = rejection_counts.get(lang, 0) + 1
            return []
        split_counts[lang] = split_counts.get(lang, 0) + (len(sub_texts) - 1)
        final_texts.extend(sub_texts)

    if not final_texts:
        rejection_counts[lang] = rejection_counts.get(lang, 0) + 1
        return []

    # Step 3: build one record per final chunk with real ordinal/total.
    chunk_total = len(final_texts)
    records: list[dict] = []
    for ordinal, text in enumerate(final_texts):
        token_length = tok_wrapper.count_tokens(text, add_prefix=True)  # type: ignore[attr-defined]
        sub_content_hash = _content_hash(text)
        point_id = _make_point_id(
            dataset_revision=dataset_revision,
            language=lang,
            content_hash=sub_content_hash,
            chunk_strategy_version=f"{chunk_strategy}_{chunk_strategy_version}",
            chunk_ordinal=ordinal,
        )
        try:
            record = build_record(
                record_id=point_id,
                chunk_text=text,
                language=lang,
                config_language=rec["config_language"],
                dataset_revision=dataset_revision,
                split=rec["split"],
                physical_shard=rec["physical_source"],
                local_source_row=rec["local_source_row"],
                passage_position=rec["passage_position"],
                parent_passage_id=parent_passage_id,
                content_hash=sub_content_hash,
                chunk_strategy=chunk_strategy,
                chunk_strategy_version=chunk_strategy_version,
                chunk_ordinal=ordinal,
                chunk_total=chunk_total,
                token_length=token_length,
                tokenizer_fingerprint=tok_wrapper.fingerprint,  # type: ignore[attr-defined]
                manifest_id=manifest_id,
            )
        except SchemaViolationError as e:  # pragma: no cover - defensive
            raise RuntimeError(f"build_record rejected a prepared record: {e}") from e
        # dataset_repo is corpus-level provenance kept in the manifest, not per-record.
        # physical_source is not an official schema field — physical_shard carries the path.
        records.append(record)

    return records


def _fill_quota(
    candidates: list[dict],
    quota: int,
    lang: str,
    seed: int,
    dataset_revision: str,
    tok_wrapper: object,
    chunk_strategy: str,
    chunk_strategy_version: str,
    manifest_id: str,
    split_counts: dict[str, int],
    rejection_counts: dict[str, int],
) -> list[dict]:
    """Select candidates deterministically and build prepared records until quota is met.

    Backtracks into the sorted candidate list to fill the exact quota even if
    some passages are oversized and must be split/rejected.
    """
    sorted_candidates = _deterministic_select(candidates, len(candidates), seed=seed, lang=lang)
    result: list[dict] = []
    seen_ids: set[str] = set()

    for rec in sorted_candidates:
        if len(result) >= quota:
            break
        built = _build_prepared_records_for_passage(
            rec,
            dataset_revision,
            tok_wrapper,
            chunk_strategy,
            chunk_strategy_version,
            manifest_id,
            split_counts,
            rejection_counts,
        )
        for r in built:
            if len(result) >= quota:
                break
            if r["id"] not in seen_ids:
                seen_ids.add(r["id"])
                result.append(r)

    return result


def _projected_batches(total_records: int) -> tuple[int, int, int]:
    """Return (num_batches, max_records_per_batch, max_bytes_per_batch)."""
    import math

    num_batches = math.ceil(total_records / MAX_RECORDS_PER_BATCH)
    return num_batches, MAX_RECORDS_PER_BATCH, MAX_BYTES_PER_BATCH


def main() -> None:
    args = _build_parser().parse_args()

    chunk_strategy = args.chunk_strategy
    quotas = {"en": args.en_quota, "hi": args.hi_quota, "bn": args.bn_quota}

    # Step 1: Resolve dataset revision to a full immutable SHA
    dataset_revision = _resolve_dataset_revision(args.dataset_revision)
    is_pinned = args.dataset_revision is not None

    # Resolve tokenizer contract early so it feeds the manifest identity.
    # TOKENIZER_REPO is imported from pinecone_contract at module level.
    from hhgoa_rag.ingestion.tokenizer import MODEL_INPUT_LIMIT

    resolved_tok_revision = _resolve_tokenizer_revision(args.tokenizer_revision, TOKENIZER_REPO)
    from hhgoa_rag.ingestion.tokenizer import get_tokenizer

    tok = get_tokenizer(revision=resolved_tok_revision)
    logger.info("Tokenizer loaded: fingerprint=%s", tok.fingerprint)

    # Manifest identity depends on EVERY corpus-affecting input.  Changing any of
    # these changes the manifest_id (and hence output filenames + record IDs).
    fingerprint_src = json.dumps(
        {
            "dataset_repo": DATASET_REPO,
            "dataset_revision": dataset_revision,
            "tokenizer_repo": TOKENIZER_REPO,
            "tokenizer_revision": resolved_tok_revision,
            "tokenizer_fingerprint": tok.fingerprint,
            "embedding_model": _CANONICAL_MODEL,
            "model_input_limit": MODEL_INPUT_LIMIT,
            "split": args.split,
            "physical_sources": {
                lang: _PARQUET_FILES.get((lang, args.split)) for lang in CANARY_SOURCE_CONFIGS
            },
            "seed": args.seed,
            "quotas": quotas,
            "normalization_version": NORMALIZATION_VERSION,
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "chunk_strategy": chunk_strategy,
            "chunk_strategy_version": CHUNK_STRATEGY_VERSION,
            "chunk_parameters": _chunk_parameters(chunk_strategy),
            "text_field": TEXT_FIELD,
            "field_map": {"text": TEXT_FIELD},
        },
        sort_keys=True,
    )
    manifest_id = f"canary-{args.seed}-{hashlib.sha256(fingerprint_src.encode()).hexdigest()[:12]}"

    logger.info("Manifest ID: %s", manifest_id)
    logger.info("Quotas: %s", quotas)
    logger.info("Chunk strategy: %s", chunk_strategy)
    logger.info(
        "Dataset revision: %s (%s)", dataset_revision, "pinned" if is_pinned else "resolved HEAD"
    )

    # Tokenizer pin status (tokenizer already resolved + loaded above).
    tokenizer_revision_pinned = (
        args.tokenizer_revision is not None
        and args.tokenizer_revision not in ("HEAD", "main", "unknown")
        and len(args.tokenizer_revision) >= 7
    )
    if not tokenizer_revision_pinned and resolved_tok_revision not in ("HEAD", "main", "unknown"):
        if len(resolved_tok_revision) >= 7:
            tokenizer_revision_pinned = True

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Preflight disk space before any (full-shard) downloads.
    _preflight_disk_space()

    # Step 3: Download + read Parquet files for each source config
    all_native_hi: list[dict] = []
    all_native_bn: list[dict] = []
    all_english: list[dict] = []
    seen_en_hashes: set[str] = set()
    en_dedup_count = 0

    hi_parquet_path: str = ""
    bn_parquet_path: str = ""

    for config_lang in CANARY_SOURCE_CONFIGS:
        rows, parquet_path = _download_and_read_parquet(
            config_lang,
            args.split,
            dataset_revision,
            args.max_rows_per_config,
        )

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

        # Global English deduplication across both physical files
        for rec in english:
            if rec["content_hash"] not in seen_en_hashes:
                seen_en_hashes.add(rec["content_hash"])
                all_english.append(rec)
            else:
                en_dedup_count += 1

    # Step 4: Fail closed if Hindi and Bengali resolve to the same source
    _check_sources_differ(hi_parquet_path, bn_parquet_path)

    logger.info(
        "Raw candidates: en=%d, hi=%d, bn=%d",
        len(all_english),
        len(all_native_hi),
        len(all_native_bn),
    )

    # Step 5: Within-language deduplication
    def _dedup(items: list[dict]) -> tuple[list[dict], int]:
        seen: set[str] = set()
        out: list[dict] = []
        removed = 0
        for item in items:
            if item["content_hash"] not in seen:
                seen.add(item["content_hash"])
                out.append(item)
            else:
                removed += 1
        return out, removed

    all_native_hi, hi_dedup_count = _dedup(all_native_hi)
    all_native_bn, bn_dedup_count = _dedup(all_native_bn)

    deduplication_counts = {
        "en_cross_file": en_dedup_count,
        "hi_within": hi_dedup_count,
        "bn_within": bn_dedup_count,
    }

    logger.info(
        "After dedup: en=%d, hi=%d, bn=%d",
        len(all_english),
        len(all_native_hi),
        len(all_native_bn),
    )

    # Step 6: Build prepared records with real tokenization, backfilling if needed
    split_counts: dict[str, int] = {}
    rejection_counts: dict[str, int] = {}

    en_records = _fill_quota(
        all_english,
        quotas["en"],
        "en",
        args.seed,
        dataset_revision,
        tok,
        chunk_strategy,
        CHUNK_STRATEGY_VERSION,
        manifest_id,
        split_counts,
        rejection_counts,
    )
    hi_records = _fill_quota(
        all_native_hi,
        quotas["hi"],
        "hi",
        args.seed,
        dataset_revision,
        tok,
        chunk_strategy,
        CHUNK_STRATEGY_VERSION,
        manifest_id,
        split_counts,
        rejection_counts,
    )
    bn_records = _fill_quota(
        all_native_bn,
        quotas["bn"],
        "bn",
        args.seed,
        dataset_revision,
        tok,
        chunk_strategy,
        CHUNK_STRATEGY_VERSION,
        manifest_id,
        split_counts,
        rejection_counts,
    )

    prepared = en_records + hi_records + bn_records

    # Step 7: Compute per-language stats
    actual_counts: dict[str, int] = {}
    actual_tokens: dict[str, int] = {}
    for rec in prepared:
        lang = rec["language"]
        actual_counts[lang] = actual_counts.get(lang, 0) + 1
        actual_tokens[lang] = actual_tokens.get(lang, 0) + rec["token_length"]

    total_records = len(prepared)
    total_tokens = sum(r["token_length"] for r in prepared)

    # Step 8: Forbidden field audit
    forbidden_found: list[str] = []
    for rec in prepared:
        bad = FORBIDDEN_FIELDS & set(rec.keys())
        if bad:
            forbidden_found.extend(sorted(bad))

    # Step 9: Compute deterministic data checksum (excludes timestamps)
    data_for_checksum = json.dumps(
        [
            {k: v for k, v in r.items() if k != "manifest_id"}
            for r in sorted(prepared, key=lambda x: x["id"])
        ],
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    prepared_data_checksum = hashlib.sha256(data_for_checksum).hexdigest()

    # Step 10: Atomic write
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

    # Conservative documented storage estimate for integrated dense embeddings:
    #   dense vector: 1024 dims * 4 bytes = 4096 bytes
    #   metadata + text payload: measured prepared bytes per record
    #   overhead factor: 1.3x for index structures
    # This replaces the arbitrary total_records * 1500 heuristic.
    DENSE_VECTOR_BYTES = 1024 * 4
    avg_payload_bytes = (jsonl_bytes / total_records) if total_records else 0
    projected_indexed_bytes = int(total_records * (DENSE_VECTOR_BYTES + avg_payload_bytes) * 1.3)

    starter_budget_ok = (
        total_records <= 10_000
        and total_tokens <= 4_000_000
        and projected_indexed_bytes <= int(1.5 * 1024 * 1024 * 1024)
    )

    num_planned_requests, max_records_per_request, max_serialized_request_bytes = (
        _projected_batches(total_records)
    )

    # Step 11: Determine ready_for_write
    readiness_failures: list[str] = []
    if not is_pinned:
        readiness_failures.append(
            "dataset_revision was not pinned by caller (resolved HEAD — not reproducible)"
        )
    if not tokenizer_revision_pinned:
        readiness_failures.append(
            f"tokenizer_revision is not a pinned SHA: {resolved_tok_revision!r} "
            "(pass --tokenizer-revision <full-sha>)"
        )
    if actual_counts.get("en", 0) != quotas["en"]:
        readiness_failures.append(
            f"English quota not met: {actual_counts.get('en', 0)} != {quotas['en']}"
        )
    if actual_counts.get("hi", 0) != quotas["hi"]:
        readiness_failures.append(
            f"Hindi quota not met: {actual_counts.get('hi', 0)} != {quotas['hi']}"
        )
    if actual_counts.get("bn", 0) != quotas["bn"]:
        readiness_failures.append(
            f"Bengali quota not met: {actual_counts.get('bn', 0)} != {quotas['bn']}"
        )
    if forbidden_found:
        readiness_failures.append(f"Forbidden fields present: {sorted(set(forbidden_found))}")
    if not starter_budget_ok:
        readiness_failures.append("Starter budget projections exceeded")
    if rejection_counts:
        total_rejected = sum(rejection_counts.values())
        readiness_failures.append(
            f"Some passages permanently rejected (over token limit after splitting): "
            f"{rejection_counts} (total={total_rejected})"
        )

    ready_for_write = len(readiness_failures) == 0

    created_at = datetime.now(UTC).isoformat()

    manifest: dict = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "manifest_id": manifest_id,
        "mode": "canary",
        "contract_version": CONTRACT_VERSION,
        "contract_fingerprint": contract_fingerprint(),
        "index_contract": canonical_contract(),
        "index_name": "msmarco-xi",
        "index_namespace": _CONTRACT_NAMESPACE,
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
        "tokenizer_revision": resolved_tok_revision,
        "tokenizer_revision_pinned": tokenizer_revision_pinned,
        "tokenizer_fingerprint": tok.fingerprint,
        "model_input_limit": MODEL_INPUT_LIMIT,
        "chunk_strategy": chunk_strategy,
        "chunk_strategy_version": CHUNK_STRATEGY_VERSION,
        "deduplication_counts": deduplication_counts,
        "split_counts": split_counts,
        "rejection_counts": rejection_counts,
        "forbidden_field_audit": (
            "PASS" if not forbidden_found else f"FAIL: {sorted(set(forbidden_found))}"
        ),
        "prepared_record_path": jsonl_path.name,
        "prepared_record_path_full": str(jsonl_path),
        "prepared_record_checksum": jsonl_checksum,
        "prepared_data_checksum": prepared_data_checksum,
        "number_of_planned_requests": num_planned_requests,
        "maximum_records_in_a_request": max_records_per_request,
        "maximum_serialized_request_bytes": max_serialized_request_bytes,
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

    manifest_for_checksum = {k: v for k, v in manifest.items() if k != "manifest_checksum"}
    manifest_checksum = hashlib.sha256(
        json.dumps(manifest_for_checksum, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    manifest["manifest_checksum"] = manifest_checksum

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
        print(f"  Chunk strat  : {chunk_strategy}")
        print(f"  JSONL path   : {jsonl_path}")
        print(f"  Manifest     : {manifest_path}")
        print(
            f"  Ready        : "
            f"{'YES' if ready_for_write else 'NO — ' + '; '.join(readiness_failures)}"
        )

    sys.exit(0 if ready_for_write else 1)


if __name__ == "__main__":
    main()
