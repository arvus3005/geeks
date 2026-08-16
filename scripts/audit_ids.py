#!/usr/bin/env python3
"""Comprehensive ID audit for MSMARCO-XI prepared JSONL records.

Detects:
- Duplicate record IDs
- Same ID mapped to different text or language
- Incorrect deterministic ID recomputation
- Cross-language ID collisions
- Broken parent linkage
- Invalid chunk ordinals/totals (0 <= ordinal < total, total > 0)
- Duplicate chunk content (same chunk_text across different IDs)
- Per-language counts and token distributions

Output: JSON report + human-readable summary (no secrets, no sensitive content).

Usage:
    uv run python scripts/audit_ids.py --records artifacts/prepared/<id>_records.jsonl
    uv run python scripts/audit_ids.py --fixtures tests/fixtures/smoke_passages.json --legacy
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from hhgoa_rag.pinecone_contract import TEXT_FIELD


def _recompute_id(rec: dict) -> str | None:
    """Recompute the deterministic ID for a record. Returns None if cannot."""
    try:
        from hhgoa_rag.ingestion.passage_ids import make_point_id

        return make_point_id(
            dataset_revision=rec["dataset_revision"],
            language=rec["language"],
            content_hash=rec["content_hash"],
            chunk_strategy_version=f"{rec['chunk_strategy']}_{rec['chunk_strategy_version']}",
            chunk_ordinal=rec["chunk_ordinal"],
        )
    except Exception:
        return None


def audit_records(records: list[dict]) -> dict:
    """Run all ID/linkage audits and return a structured result dict."""
    seen_ids: dict[str, dict] = {}  # id -> first record summary
    duplicate_ids: list[dict] = []
    id_text_conflicts: list[dict] = []
    id_lang_conflicts: list[dict] = []
    id_recompute_mismatches: list[dict] = []
    cross_lang_collisions: list[dict] = []
    ordinal_errors: list[dict] = []
    duplicate_text_ids: list[dict] = []
    broken_parent_links: list[dict] = []

    text_to_ids: dict[str, list[str]] = defaultdict(list)
    parent_chunks: dict[str, list[dict]] = defaultdict(list)

    lang_counts: dict[str, int] = defaultdict(int)
    lang_tokens: dict[str, list[int]] = defaultdict(list)

    for rec in records:
        rid = rec.get("id", "")
        lang = rec.get("language", "unknown")
        chunk_text = rec.get(TEXT_FIELD, "")
        token_len = rec.get("token_length", 0)

        lang_counts[lang] += 1
        if isinstance(token_len, int):
            lang_tokens[lang].append(token_len)

        if rid in seen_ids:
            first = seen_ids[rid]
            entry = {"id": rid, "first_lang": first["language"], "second_lang": lang}
            duplicate_ids.append(entry)
            if first[TEXT_FIELD] != chunk_text:
                id_text_conflicts.append({"id": rid, "lang_a": first["language"], "lang_b": lang})
            if first["language"] != lang:
                id_lang_conflicts.append(entry)
        else:
            seen_ids[rid] = {"language": lang, TEXT_FIELD: chunk_text}

        text_to_ids[chunk_text].append(rid)

        parent_id = rec.get("parent_passage_id", "")
        if parent_id:
            parent_chunks[parent_id].append(rec)

    # Cross-language collision: same content_hash + chunk_ordinal, different languages
    ch_ordinal_to_lang: dict[tuple, str] = {}
    for rec in records:
        key = (rec.get("content_hash", ""), rec.get("chunk_ordinal", 0))
        lang = rec.get("language", "unknown")
        if key in ch_ordinal_to_lang and ch_ordinal_to_lang[key] != lang:
            cross_lang_collisions.append(
                {
                    "content_hash": rec.get("content_hash"),
                    "chunk_ordinal": rec.get("chunk_ordinal"),
                    "lang_a": ch_ordinal_to_lang[key],
                    "lang_b": lang,
                    "id": rec.get("id"),
                }
            )
        else:
            ch_ordinal_to_lang[key] = lang

    # ID recomputation check
    for rec in records:
        rid = rec.get("id", "")
        recomputed = _recompute_id(rec)
        if recomputed is not None and recomputed != rid:
            id_recompute_mismatches.append(
                {
                    "id": rid,
                    "recomputed": recomputed,
                    "language": rec.get("language"),
                    "chunk_ordinal": rec.get("chunk_ordinal"),
                }
            )

    # Ordinal/total validation per parent group
    for parent_id, chunks in parent_chunks.items():
        for rec in chunks:
            ordinal = rec.get("chunk_ordinal")
            total = rec.get("chunk_total")
            if not isinstance(ordinal, int) or not isinstance(total, int):
                ordinal_errors.append({"id": rec.get("id"), "error": "non-integer ordinal/total"})
                continue
            if total <= 0:
                ordinal_errors.append(
                    {"id": rec.get("id"), "error": f"chunk_total={total} must be positive"}
                )
            elif not (0 <= ordinal < total):
                ordinal_errors.append(
                    {"id": rec.get("id"), "error": f"ordinal={ordinal} not in [0, {total})"}
                )

        ordinals = sorted(r.get("chunk_ordinal", -1) for r in chunks)
        if chunks:
            expected_total = chunks[0].get("chunk_total", -1)
            expected_ordinals = list(range(expected_total))
            if ordinals != expected_ordinals:
                ordinal_errors.append(
                    {
                        "parent_id": parent_id,
                        "error": (
                            f"missing ordinals: expected {expected_ordinals}, got {ordinals}"
                        ),
                    }
                )

    # Duplicate text detection (different IDs, same content)
    for text, ids in text_to_ids.items():
        unique_ids = list(dict.fromkeys(ids))
        if len(unique_ids) > 1:
            duplicate_text_ids.append({"chunk_text_prefix": text[:60], "ids": unique_ids[:5]})

    # Broken parent links: chunk_total > 1 but only 1 sibling found
    for parent_id, chunks in parent_chunks.items():
        if len(chunks) == 1 and chunks[0].get("chunk_total", 1) > 1:
            broken_parent_links.append(
                {
                    "parent_passage_id": parent_id,
                    "error": "chunk_total > 1 but only one sibling found",
                    "id": chunks[0].get("id"),
                    "chunk_total": chunks[0].get("chunk_total"),
                }
            )

    def _percentiles(vals: list[int]) -> dict:
        if not vals:
            return {}
        s = sorted(vals)
        n = len(s)
        return {
            "count": n,
            "p50": s[n // 2],
            "p70": s[int(n * 0.70)],
            "p95": s[int(n * 0.95)],
            "p100": s[-1],
            "mean": round(sum(s) / n, 1),
        }

    lang_token_stats = {lang: _percentiles(toks) for lang, toks in sorted(lang_tokens.items())}

    verdict = (
        "PASS"
        if (
            len(duplicate_ids) == 0
            and not id_text_conflicts
            and not id_recompute_mismatches
            and not cross_lang_collisions
            and not ordinal_errors
            and not broken_parent_links
        )
        else "FAIL"
    )

    return {
        "verdict": verdict,
        "total_records": len(records),
        "unique_ids": len(seen_ids),
        "duplicate_ids_count": len(duplicate_ids),
        "id_text_conflicts_count": len(id_text_conflicts),
        "id_lang_conflicts_count": len(id_lang_conflicts),
        "id_recompute_mismatches_count": len(id_recompute_mismatches),
        "cross_lang_collisions_count": len(cross_lang_collisions),
        "ordinal_errors_count": len(ordinal_errors),
        "duplicate_text_count": len(duplicate_text_ids),
        "broken_parent_links_count": len(broken_parent_links),
        "per_language_counts": dict(lang_counts),
        "per_language_token_stats": lang_token_stats,
        "issues": {
            "duplicate_ids": duplicate_ids[:20],
            "id_text_conflicts": id_text_conflicts[:10],
            "id_lang_conflicts": id_lang_conflicts[:10],
            "id_recompute_mismatches": id_recompute_mismatches[:10],
            "cross_lang_collisions": cross_lang_collisions[:10],
            "ordinal_errors": ordinal_errors[:10],
            "duplicate_texts": duplicate_text_ids[:5],
            "broken_parent_links": broken_parent_links[:10],
        },
    }


def _load_prepared_records(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"ERROR: invalid JSON on line {lineno}: {e}", file=sys.stderr)
                sys.exit(1)
    return records


def _load_legacy_fixtures(path: Path) -> list[dict]:
    """Load legacy smoke-fixture JSON (list of passage dicts without full schema)."""
    from hhgoa_rag.ingestion.normalizer import content_hash
    from hhgoa_rag.ingestion.passage_ids import make_point_id

    with open(path, encoding="utf-8") as f:
        fixtures = json.load(f)

    DATASET_REVISION = "smoke-fixture-v1"
    CHUNK_STRATEGY_VERSION = "passage_native_v1"
    records = []
    for p in fixtures:
        chash = content_hash(p["text"])
        pid = make_point_id(
            dataset_revision=DATASET_REVISION,
            language=p["language"],
            content_hash=chash,
            chunk_strategy_version=CHUNK_STRATEGY_VERSION,
            chunk_ordinal=0,
        )
        records.append(
            {
                "id": pid,
                TEXT_FIELD: p["text"],
                "language": p["language"],
                "content_hash": chash,
                "dataset_revision": DATASET_REVISION,
                "chunk_strategy": "passage_native",
                "chunk_strategy_version": "v1",
                "chunk_ordinal": 0,
                "chunk_total": 1,
                "token_length": 1,
                "parent_passage_id": pid,
            }
        )
    return records


def _print_summary(result: dict) -> None:
    print(f"\n{'='*60}")
    print("ID AUDIT SUMMARY")
    print(f"{'='*60}")
    print(f"  Verdict              : {result['verdict']}")
    print(f"  Total records        : {result['total_records']}")
    print(f"  Unique IDs           : {result['unique_ids']}")
    print(f"  Duplicate IDs        : {result['duplicate_ids_count']}")
    print(f"  ID text conflicts    : {result['id_text_conflicts_count']}")
    print(f"  ID lang conflicts    : {result['id_lang_conflicts_count']}")
    print(f"  Recompute mismatches : {result['id_recompute_mismatches_count']}")
    print(f"  Cross-lang collisions: {result['cross_lang_collisions_count']}")
    print(f"  Ordinal errors       : {result['ordinal_errors_count']}")
    print(f"  Duplicate texts      : {result['duplicate_text_count']}")
    print(f"  Broken parent links  : {result['broken_parent_links_count']}")
    print()
    print("  Per-language counts:")
    for lang, count in sorted(result["per_language_counts"].items()):
        stats = result["per_language_token_stats"].get(lang, {})
        p50 = stats.get("p50", "?")
        p95 = stats.get("p95", "?")
        print(f"    {lang}: {count} records  (token P50={p50}, P95={p95})")
    print(f"{'='*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Comprehensive ID/linkage audit for MSMARCO-XI")
    parser.add_argument("--records", type=Path, default=None, help="Path to prepared JSONL file")
    parser.add_argument(
        "--fixtures",
        default=None,
        help="Legacy: path to smoke_passages.json fixtures",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use legacy fixture loader (--fixtures mode)",
    )
    parser.add_argument("--output-json", action="store_true", help="Print JSON result")
    parser.add_argument("--output-file", type=Path, default=None, help="Write JSON report to file")
    args = parser.parse_args()

    if args.records:
        records = _load_prepared_records(args.records)
    elif args.fixtures or args.legacy:
        fixture_path = Path(args.fixtures or "tests/fixtures/smoke_passages.json")
        records = _load_legacy_fixtures(fixture_path)
    else:
        parser.error("Provide --records <path.jsonl> or --fixtures <path.json> --legacy")

    result = audit_records(records)

    if args.output_json:
        print(json.dumps(result, indent=2))
    else:
        _print_summary(result)

    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"JSON report written to {args.output_file}", file=sys.stderr)

    sys.exit(0 if result["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
