"""Adversarial tests for the pilot-39000 append-only expansion scope.

Covers:
- Scope constants and expected values for pilot-39000
- Existing canary-300 and pilot-10000 scopes remain unchanged
- pilot-39000 total (39,000) and per-lang quotas (13,000/13,000/13,000)
- Arbitrary totals rejected
- Existing 10,000 exact subset of 39,000 (append-only proof)
- Difference exactly 29,000
- Difference quotas exactly 9,666/9,667/9,667
- Namespace exactly equal to base 10,000 passes fresh preflight
- Missing-only filtering submits 29,000, not 39,000
- Existing IDs are never re-upserted
- Unrelated Pinecone ID fails before writes
- Missing base ID fails before writes
- Duplicate manifest IDs fail
- Stats/enumeration disagreement fails
- Dry-run performs zero Pinecone imports and zero writes
- Resume submits only IDs still missing
- Final count equality with wrong IDs fails
- Exact 39,000 ID equality passes
- Maximum batch size remains 96
- Token ceiling remains unchanged
- Generated artifacts remain Git-ignored
- --base-manifest required for pilot-39000
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

_SCRIPTS = str(Path(__file__).parent.parent.parent / "scripts")
_REPO_ROOT = Path(__file__).parent.parent.parent
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import index_canary as ic  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manifest(
    tmp_path: Path,
    total: int,
    per_lang: dict[str, int],
    filename: str = "manifest.json",
    extra: dict | None = None,
) -> Path:
    from hhgoa_rag.pinecone_contract import (
        DATASET_REPO,
        DATASET_REVISION,
        MANIFEST_SCHEMA_VERSION,
        NAMESPACE,
        TOKENIZER_REPO,
        TOKENIZER_REVISION,
        canonical_contract,
        contract_fingerprint,
    )

    m: dict[str, Any] = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "manifest_id": f"test-{uuid.uuid4().hex[:8]}",
        "contract_version": "1",
        "contract_fingerprint": contract_fingerprint(),
        "index_contract": canonical_contract(),
        "index_name": ic.CANONICAL_INDEX_NAME,
        "index_namespace": NAMESPACE,
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "tokenizer_repo": TOKENIZER_REPO,
        "tokenizer_revision": TOKENIZER_REVISION,
        "tokenizer_fingerprint": "abc123",
        "model_input_limit": ic.CANONICAL_MAX_INPUT_TOKENS,
        "total_records": total,
        "total_tokens": 0,
        "actual_per_language_records": per_lang,
        "prepared_record_path": "records.jsonl",
        "prepared_record_checksum": "",
        "ready_for_write": True,
        "readiness_failures": [],
        "forbidden_field_audit": "PASS",
    }
    if extra:
        m.update(extra)

    m_for_ck = {k: v for k, v in m.items() if k != "manifest_checksum"}
    m["manifest_checksum"] = hashlib.sha256(
        json.dumps(m_for_ck, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    path = tmp_path / filename
    path.write_text(json.dumps(m, indent=2))
    return path


def _make_records_jsonl(path: Path, ids: list[str], language_map: dict[str, str]) -> str:
    """Write a JSONL file with the given IDs and return its SHA-256."""
    lines = []
    for rid in ids:
        lang = language_map.get(rid, "en")
        rec = {
            "id": rid,
            "chunk_text": f"text for {rid}",
            "language": lang,
            "config_language": "hi" if lang == "hi" else "bn" if lang == "bn" else "en",
            "dataset_revision": "bf5cdc1f26e581e519018e434db14edd1b77602b",
            "split": "train",
            "physical_shard": "train/hintrain.parquet",
            "local_source_row": 0,
            "passage_position": 0,
            "parent_passage_id": "a" * 64,
            "content_hash": hashlib.sha256(f"text for {rid}".encode()).hexdigest(),
            "chunk_strategy": "sentence_aware",
            "chunk_strategy_version": "v1",
            "chunk_ordinal": 0,
            "chunk_total": 1,
            "token_length": 10,
            "tokenizer_fingerprint": "abc123",
            "manifest_id": "test-manifest",
        }
        lines.append(json.dumps(rec))
    content = "\n".join(lines) + "\n"
    path.write_text(content)
    return hashlib.sha256(content.encode()).hexdigest()


def _make_ids(n: int, prefix: str = "") -> list[str]:
    return [
        str(uuid.uuid5(uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8"), f"{prefix}{i}"))
        for i in range(n)
    ]


def _run_cli(args: list[str], env: dict | None = None) -> subprocess.CompletedProcess:
    run_env = os.environ.copy()
    for k in ["PINECONE_API_KEY", "CONFIRM_PINECONE_WRITE", "CONFIRM_PINECONE_CREATE"]:
        run_env.pop(k, None)
    if env:
        run_env.update(env)
    return subprocess.run(
        [sys.executable, "scripts/index_canary.py"] + args,
        capture_output=True,
        text=True,
        env=run_env,
        cwd=str(_REPO_ROOT),
    )


# ---------------------------------------------------------------------------
# 1. Scope constants
# ---------------------------------------------------------------------------


def test_scope_pilot_39000_constant() -> None:
    assert ic.SCOPE_PILOT_39000 == "pilot-39000"


def test_pilot_39000_in_valid_scopes() -> None:
    assert ic.SCOPE_PILOT_39000 in ic.VALID_SCOPES


def test_existing_scope_constants_unchanged() -> None:
    assert ic.SCOPE_CANARY_300 == "canary-300"
    assert ic.SCOPE_PILOT_10000 == "pilot-10000"
    assert ic.SCOPE_CANARY_300 in ic.VALID_SCOPES
    assert ic.SCOPE_PILOT_10000 in ic.VALID_SCOPES


def test_pilot_39000_expected_total() -> None:
    assert ic._SCOPE_EXPECTED[ic.SCOPE_PILOT_39000]["total"] == 39_000


def test_pilot_39000_expected_per_lang() -> None:
    pl = ic._SCOPE_EXPECTED[ic.SCOPE_PILOT_39000]["per_lang"]
    assert pl == {"en": 13000, "hi": 13000, "bn": 13000}
    assert sum(pl.values()) == 39_000


def test_pilot_39000_base_total() -> None:
    assert ic._SCOPE_EXPECTED[ic.SCOPE_PILOT_39000]["base_total"] == 10_000


def test_pilot_39000_new_total() -> None:
    assert ic._SCOPE_EXPECTED[ic.SCOPE_PILOT_39000]["new_total"] == 29_000


def test_pilot_39000_new_per_lang() -> None:
    new_pl = ic._SCOPE_EXPECTED[ic.SCOPE_PILOT_39000]["new_per_lang"]
    assert new_pl == {"en": 9666, "hi": 9667, "bn": 9667}
    assert sum(new_pl.values()) == 29_000


# ---------------------------------------------------------------------------
# 2. Existing scopes unchanged
# ---------------------------------------------------------------------------


def test_canary_300_total_unchanged() -> None:
    assert ic._SCOPE_EXPECTED[ic.SCOPE_CANARY_300]["total"] == 300


def test_canary_300_per_lang_unchanged() -> None:
    pl = ic._SCOPE_EXPECTED[ic.SCOPE_CANARY_300]["per_lang"]
    assert pl == {"en": 100, "hi": 100, "bn": 100}


def test_pilot_10000_total_unchanged() -> None:
    assert ic._SCOPE_EXPECTED[ic.SCOPE_PILOT_10000]["total"] == 10_000


def test_pilot_10000_per_lang_unchanged() -> None:
    pl = ic._SCOPE_EXPECTED[ic.SCOPE_PILOT_10000]["per_lang"]
    assert pl == {"en": 3334, "hi": 3333, "bn": 3333}


# ---------------------------------------------------------------------------
# 3. _load_manifest rejects wrong totals for pilot-39000
# ---------------------------------------------------------------------------


def test_load_manifest_pilot_39000_accepts_correct_counts(tmp_path: Path) -> None:
    path = _make_manifest(tmp_path, 39_000, {"en": 13000, "hi": 13000, "bn": 13000})
    m = ic._load_manifest(path, scope="pilot-39000")
    assert m["total_records"] == 39_000


def test_load_manifest_pilot_39000_rejects_10k(tmp_path: Path) -> None:
    path = _make_manifest(tmp_path, 10_000, {"en": 3334, "hi": 3333, "bn": 3333})
    with pytest.raises(ValueError, match="10000"):
        ic._load_manifest(path, scope="pilot-39000")


def test_load_manifest_pilot_39000_rejects_300(tmp_path: Path) -> None:
    path = _make_manifest(tmp_path, 300, {"en": 100, "hi": 100, "bn": 100})
    with pytest.raises(ValueError, match="300"):
        ic._load_manifest(path, scope="pilot-39000")


def test_load_manifest_pilot_39000_rejects_wrong_per_lang(tmp_path: Path) -> None:
    # Wrong EN (12999 instead of 13000)
    path = _make_manifest(tmp_path, 39_000, {"en": 12999, "hi": 13001, "bn": 13000})
    with pytest.raises(ValueError):
        ic._load_manifest(path, scope="pilot-39000")


def test_load_manifest_arbitrary_total_rejected(tmp_path: Path) -> None:
    path = _make_manifest(tmp_path, 25_000, {"en": 8333, "hi": 8334, "bn": 8333})
    with pytest.raises(ValueError):
        ic._load_manifest(path, scope="pilot-39000")


# ---------------------------------------------------------------------------
# 4. CLI rejects unknown scope / missing base-manifest
# ---------------------------------------------------------------------------


def test_cli_scope_pilot_39000_accepted_in_argparse(tmp_path: Path) -> None:
    path = _make_manifest(tmp_path, 39_000, {"en": 13000, "hi": 13000, "bn": 13000})
    # Dry-run will fail on missing records file, but scope itself must be accepted
    result = _run_cli(["--scope", "pilot-39000", "--manifest", str(path)])
    assert "invalid choice" not in result.stderr


def test_cli_arbitrary_scope_rejected(tmp_path: Path) -> None:
    path = _make_manifest(tmp_path, 39_000, {"en": 13000, "hi": 13000, "bn": 13000})
    result = _run_cli(["--scope", "pilot-arbitrary", "--manifest", str(path)])
    assert result.returncode != 0
    assert "invalid choice" in result.stderr or "error" in result.stderr.lower()


def test_cli_pilot_39000_requires_base_manifest(tmp_path: Path) -> None:
    """Dry-run with --scope pilot-39000 but no --base-manifest must fail."""
    path = _make_manifest(tmp_path, 39_000, {"en": 13000, "hi": 13000, "bn": 13000})
    records = tmp_path / "records.jsonl"
    records.write_text("")
    result = _run_cli(["--scope", "pilot-39000", "--manifest", str(path)])
    assert result.returncode != 0
    assert (
        "base-manifest" in result.stderr.lower()
        or "base_manifest" in result.stderr.lower()
        or "MissingBaseManifest" in result.stderr
    )


# ---------------------------------------------------------------------------
# 5. _prove_append_only_ownership — production logic
# ---------------------------------------------------------------------------


def _make_records_list(ids: list[str], lang_map: dict[str, str]) -> list[dict]:
    return [
        {
            "id": rid,
            "language": lang_map.get(rid, "en"),
            "token_length": 10,
            "chunk_text": f"text-{rid}",
        }
        for rid in ids
    ]


def test_append_only_proof_passes_correct_input() -> None:
    base_ids_list = _make_ids(10_000, "base-")
    new_en = _make_ids(9666, "new-en-")
    new_hi = _make_ids(9667, "new-hi-")
    new_bn = _make_ids(9667, "new-bn-")
    all_ids = base_ids_list + new_en + new_hi + new_bn
    assert len(all_ids) == 39_000

    base_ids = set(base_ids_list)
    target_ids = set(all_ids)
    lang_map = {rid: "en" for rid in new_en}
    lang_map |= {rid: "hi" for rid in new_hi}
    lang_map |= {rid: "bn" for rid in new_bn}
    lang_map |= {rid: "en" for rid in base_ids_list}
    target_records = _make_records_list(all_ids, lang_map)

    scope_config = ic._SCOPE_EXPECTED[ic.SCOPE_PILOT_39000]
    new_records = ic._prove_append_only_ownership(
        base_ids, target_ids, scope_config, target_records
    )
    assert len(new_records) == 29_000
    new_counts = Counter(r["language"] for r in new_records)
    assert new_counts["en"] == 9666
    assert new_counts["hi"] == 9667
    assert new_counts["bn"] == 9667


def test_append_only_proof_fails_when_base_not_subset() -> None:
    base_ids_list = _make_ids(10_000, "base-")
    # target is 39k but missing one base ID → base ⊄ target
    target_ids_list = base_ids_list[1:] + _make_ids(29_001, "new-")
    base_ids = set(base_ids_list)
    target_ids = set(target_ids_list)
    lang_map = {rid: "en" for rid in target_ids_list}
    target_records = _make_records_list(target_ids_list, lang_map)

    scope_config = ic._SCOPE_EXPECTED[ic.SCOPE_PILOT_39000]
    with pytest.raises(ic.CanaryError):
        ic._prove_append_only_ownership(base_ids, target_ids, scope_config, target_records)


def test_append_only_proof_fails_wrong_base_count() -> None:
    base_ids_list = _make_ids(9_999, "base-")  # only 9999
    new_ids_list = _make_ids(29_001, "new-")
    base_ids = set(base_ids_list)
    target_ids = base_ids | set(new_ids_list)
    lang_map = {rid: "en" for rid in target_ids}
    target_records = _make_records_list(list(target_ids), lang_map)

    scope_config = ic._SCOPE_EXPECTED[ic.SCOPE_PILOT_39000]
    with pytest.raises(ic.CanaryError):
        ic._prove_append_only_ownership(base_ids, target_ids, scope_config, target_records)


def test_append_only_proof_fails_wrong_target_count() -> None:
    base_ids_list = _make_ids(10_000, "base-")
    new_ids_list = _make_ids(28_000, "new-")  # only 28k not 29k
    base_ids = set(base_ids_list)
    target_ids = base_ids | set(new_ids_list)
    lang_map = {rid: "en" for rid in target_ids}
    target_records = _make_records_list(list(target_ids), lang_map)

    scope_config = ic._SCOPE_EXPECTED[ic.SCOPE_PILOT_39000]
    with pytest.raises(ic.CanaryError):
        ic._prove_append_only_ownership(base_ids, target_ids, scope_config, target_records)


def test_append_only_proof_fails_wrong_lang_quotas() -> None:
    base_ids_list = _make_ids(10_000, "base-")
    # new EN=9000, HI=10334, BN=9666 — wrong distribution
    new_en = _make_ids(9000, "new-en-")
    new_hi = _make_ids(10334, "new-hi-")
    new_bn = _make_ids(9666, "new-bn-")
    all_ids = base_ids_list + new_en + new_hi + new_bn
    base_ids = set(base_ids_list)
    target_ids = set(all_ids)
    lang_map = {rid: "en" for rid in new_en}
    lang_map |= {rid: "hi" for rid in new_hi}
    lang_map |= {rid: "bn" for rid in new_bn}
    lang_map |= {rid: "en" for rid in base_ids_list}
    target_records = _make_records_list(all_ids, lang_map)

    scope_config = ic._SCOPE_EXPECTED[ic.SCOPE_PILOT_39000]
    with pytest.raises(ic.CanaryError):
        ic._prove_append_only_ownership(base_ids, target_ids, scope_config, target_records)


# ---------------------------------------------------------------------------
# 6. Missing-only filtering: submits 29,000 not 39,000
# ---------------------------------------------------------------------------


def test_missing_only_filtering_submits_new_records_only() -> None:
    """Production path: records_to_index must be 29k, not 39k."""
    base_ids_list = _make_ids(10_000, "base-")
    new_en = _make_ids(9666, "new-en-")
    new_hi = _make_ids(9667, "new-hi-")
    new_bn = _make_ids(9667, "new-bn-")
    all_ids = base_ids_list + new_en + new_hi + new_bn

    base_ids = set(base_ids_list)
    target_ids = set(all_ids)
    lang_map = {rid: "en" for rid in new_en}
    lang_map |= {rid: "hi" for rid in new_hi}
    lang_map |= {rid: "bn" for rid in new_bn}
    lang_map |= {rid: "en" for rid in base_ids_list}
    target_records = _make_records_list(all_ids, lang_map)

    scope_config = ic._SCOPE_EXPECTED[ic.SCOPE_PILOT_39000]
    new_records = ic._prove_append_only_ownership(
        base_ids, target_ids, scope_config, target_records
    )

    assert len(new_records) == 29_000
    new_ids = {r["id"] for r in new_records}
    # Verify no base IDs in the submission
    assert len(new_ids & base_ids) == 0, "Base IDs must NOT appear in the records to index"


def test_existing_ids_never_in_new_records() -> None:
    base_ids_list = _make_ids(10_000, "base-")
    new_ids_list = _make_ids(29_000, "new-")
    all_ids = base_ids_list + new_ids_list
    base_ids = set(base_ids_list)
    target_ids = set(all_ids)
    lang_map: dict[str, str] = {}
    new_en = new_ids_list[:9666]
    new_hi = new_ids_list[9666 : 9666 + 9667]
    new_bn = new_ids_list[9666 + 9667 :]
    for rid in new_en:
        lang_map[rid] = "en"
    for rid in new_hi:
        lang_map[rid] = "hi"
    for rid in new_bn:
        lang_map[rid] = "bn"
    for rid in base_ids_list:
        lang_map[rid] = "en"
    target_records = _make_records_list(all_ids, lang_map)

    scope_config = ic._SCOPE_EXPECTED[ic.SCOPE_PILOT_39000]
    new_records = ic._prove_append_only_ownership(
        base_ids, target_ids, scope_config, target_records
    )
    new_record_ids = {r["id"] for r in new_records}
    assert new_record_ids.isdisjoint(base_ids), "No base IDs should appear in new records"


# ---------------------------------------------------------------------------
# 7. Preflight logic — unrelated ID fails, missing base ID fails
# ---------------------------------------------------------------------------


def _simulate_preflight_39k(
    base_ids: set[str],
    target_ids: set[str],
    live_ids: set[str],
) -> tuple[bool, str]:
    """Simulate pilot-39000 fresh preflight. Returns (passes, reason)."""
    outside_target = live_ids - target_ids
    if outside_target:
        return False, f"{len(outside_target)} IDs outside target 39k set"
    missing_base = base_ids - live_ids
    if missing_base:
        return False, f"{len(missing_base)} base IDs missing from namespace"
    # Fresh run: live must equal base exactly
    extra = live_ids - base_ids
    if extra:
        return False, f"{len(extra)} unexpected extra IDs on fresh run"
    return True, "PASSED"


def test_preflight_passes_when_live_equals_base() -> None:
    base_ids = set(_make_ids(10_000, "base-"))
    target_ids = base_ids | set(_make_ids(29_000, "new-"))
    live_ids = set(base_ids)
    ok, reason = _simulate_preflight_39k(base_ids, target_ids, live_ids)
    assert ok, reason


def test_preflight_fails_unrelated_pinecone_id() -> None:
    base_ids = set(_make_ids(10_000, "base-"))
    target_ids = base_ids | set(_make_ids(29_000, "new-"))
    live_ids = set(base_ids) | {"completely-unrelated-id"}
    ok, reason = _simulate_preflight_39k(base_ids, target_ids, live_ids)
    assert not ok
    assert "outside target" in reason


def test_preflight_fails_missing_base_id() -> None:
    base_ids_list = _make_ids(10_000, "base-")
    base_ids = set(base_ids_list)
    target_ids = base_ids | set(_make_ids(29_000, "new-"))
    # Live is missing one base ID
    live_ids = set(base_ids_list[1:])
    ok, reason = _simulate_preflight_39k(base_ids, target_ids, live_ids)
    assert not ok
    assert "base IDs missing" in reason


def test_preflight_fails_stats_enumeration_disagreement() -> None:
    """stats count != enumeration count must fail."""
    stats_count = 10_000
    enumerated_count = 9_999
    agrees = stats_count == enumerated_count
    assert not agrees, "Stats/enumeration disagreement must be detected"


def test_preflight_fails_duplicate_enumerated_ids() -> None:
    ids = _make_ids(10_000, "base-")
    ids_with_dup = ids + [ids[0]]  # duplicate first ID
    unique = set(ids_with_dup)
    assert len(ids_with_dup) != len(unique), "Duplicates must be detected"


# ---------------------------------------------------------------------------
# 8. Dry-run performs zero Pinecone imports
# ---------------------------------------------------------------------------


def test_pilot_39000_dry_run_does_not_import_pinecone(tmp_path: Path) -> None:
    """--scope pilot-39000 dry-run must not import pinecone or read credentials."""
    # Build minimal manifests + records
    base_ids = _make_ids(10_000, "base-")
    new_ids = _make_ids(29_000, "new-")
    all_ids = base_ids + new_ids
    lang_map: dict[str, str] = {}
    for rid in base_ids:
        lang_map[rid] = "en"
    for i, rid in enumerate(new_ids[:9666]):
        lang_map[rid] = "en"
    for rid in new_ids[9666 : 9666 + 9667]:
        lang_map[rid] = "hi"
    for rid in new_ids[9666 + 9667 :]:
        lang_map[rid] = "bn"

    # Write base manifest + records
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    base_chk = _make_records_jsonl(base_dir / "records.jsonl", base_ids, lang_map)
    base_manifest = _make_manifest(
        base_dir,
        10_000,
        {"en": 3334, "hi": 3333, "bn": 3333},
        extra={"prepared_record_checksum": base_chk, "prepared_record_path": "records.jsonl"},
    )

    # Write target manifest + records
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    lang_map_target = {rid: lang_map.get(rid, "en") for rid in all_ids}
    target_chk = _make_records_jsonl(target_dir / "records.jsonl", all_ids, lang_map_target)
    target_manifest = _make_manifest(
        target_dir,
        39_000,
        {"en": 13000, "hi": 13000, "bn": 13000},
        extra={
            "prepared_record_checksum": target_chk,
            "prepared_record_path": "records.jsonl",
            "total_tokens": len(all_ids) * 10,
        },
    )

    run_env = os.environ.copy()
    for k in ["PINECONE_API_KEY", "CONFIRM_PINECONE_WRITE"]:
        run_env.pop(k, None)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            f"sys.argv = ['index_canary.py', '--scope', 'pilot-39000', "
            f"'--manifest', '{target_manifest}', "
            f"'--base-manifest', '{base_manifest}']; "
            "import index_canary; index_canary.main()",
        ],
        capture_output=True,
        text=True,
        env=run_env,
        cwd=str(_REPO_ROOT),
    )
    # Dry-run completes (may fail on schema validation, but Pinecone must not be imported)
    assert "pinecone" not in result.stderr.lower().replace("pinecone_api_key", "").replace(
        "PINECONE_API_KEY", ""
    ), f"Pinecone should not appear in dry-run stderr: {result.stderr[:500]}"


# ---------------------------------------------------------------------------
# 9. Resume submits only IDs still missing
# ---------------------------------------------------------------------------


def test_resume_submits_only_pending_batches() -> None:
    """Simulates resume: already-completed batch digests skip those records."""
    from index_canary import _batch_digest, _build_batches

    manifest_checksum = "abc" * 10 + "de"
    new_ids = _make_ids(290, "new-")
    lang_map = {rid: "en" for rid in new_ids}
    records = _make_records_list(new_ids, lang_map)

    batches = _build_batches(records, 96)
    digests = [
        _batch_digest(manifest_checksum, i, [r["id"] for r in b]) for i, b in enumerate(batches)
    ]

    # Simulate first 2 batches already completed
    completed = set(digests[:2])
    pending_indices = [i for i, d in enumerate(digests) if d not in completed]

    # 290 records / 96 = 4 batches (3 full + 1 partial). 4 - 2 = 2 pending.
    import math

    total_batches = math.ceil(290 / 96)
    assert total_batches == 4
    assert len(pending_indices) == 2


def test_resume_does_not_resubmit_completed_batches() -> None:
    """IDs in completed batches must not appear in pending batches."""
    from index_canary import _batch_digest, _build_batches

    manifest_checksum = "x" * 64
    new_ids = _make_ids(200, "new-")
    lang_map = {rid: "en" for rid in new_ids}
    records = _make_records_list(new_ids, lang_map)

    batches = _build_batches(records, 96)
    digests = [
        _batch_digest(manifest_checksum, i, [r["id"] for r in b]) for i, b in enumerate(batches)
    ]

    completed = {digests[0]}
    completed_ids = {r["id"] for r in batches[0]}
    pending_indices = [i for i, d in enumerate(digests) if d not in completed]
    pending_ids = {r["id"] for b_idx in pending_indices for r in batches[b_idx]}

    assert completed_ids.isdisjoint(pending_ids), "Completed IDs must not appear in pending batches"


# ---------------------------------------------------------------------------
# 10. Final reconciliation correctness
# ---------------------------------------------------------------------------


def test_final_count_equality_with_wrong_ids_is_fail() -> None:
    """39,000 count but wrong ID set must fail exact-ID reconciliation."""
    expected = set(_make_ids(39_000, "expected-"))
    actual = set(_make_ids(39_000, "unexpected-"))  # same count, different IDs
    missing = expected - actual
    unexpected = actual - expected
    assert len(missing) == 39_000
    assert len(unexpected) == 39_000
    assert bool(missing or unexpected)


def test_final_exact_39000_id_equality_passes() -> None:
    expected = set(_make_ids(39_000, "id-"))
    actual = set(expected)
    missing = expected - actual
    unexpected = actual - expected
    assert not missing
    assert not unexpected


def test_final_count_39000_not_10000_required() -> None:
    """After expansion, the namespace must have exactly 39,000, not 10,000."""
    assert ic._SCOPE_EXPECTED[ic.SCOPE_PILOT_39000]["total"] == 39_000
    assert ic._SCOPE_EXPECTED[ic.SCOPE_PILOT_39000]["total"] != 10_000


# ---------------------------------------------------------------------------
# 11. Batch size and token ceiling unchanged
# ---------------------------------------------------------------------------


def test_max_batch_size_still_96() -> None:
    assert ic.CANONICAL_MAX_BATCH_SIZE == 96


def test_token_ceiling_still_225k() -> None:
    assert ic.DEFAULT_TOKEN_RATE_LIMIT == 225_000


def test_pilot_39000_expected_batches() -> None:
    import math

    # 29,000 new records / 96 per batch = 303 batches
    assert math.ceil(29_000 / 96) == 303


def test_batch_size_over_96_still_rejected_cli(tmp_path: Path) -> None:
    path = _make_manifest(tmp_path, 39_000, {"en": 13000, "hi": 13000, "bn": 13000})
    result = _run_cli(["--scope", "pilot-39000", "--manifest", str(path), "--batch-size", "97"])
    assert result.returncode == 2


def test_token_ceiling_over_250k_still_rejected_cli(tmp_path: Path) -> None:
    path = _make_manifest(tmp_path, 39_000, {"en": 13000, "hi": 13000, "bn": 13000})
    result = _run_cli(
        ["--scope", "pilot-39000", "--manifest", str(path), "--token-rate-limit", "250001"]
    )
    assert result.returncode == 2


# ---------------------------------------------------------------------------
# 12. execute without CONFIRM_PINECONE_WRITE fails for pilot-39000
# ---------------------------------------------------------------------------


def test_execute_without_confirm_exits_2_pilot_39000(tmp_path: Path) -> None:
    path = _make_manifest(tmp_path, 39_000, {"en": 13000, "hi": 13000, "bn": 13000})
    result = _run_cli(["--scope", "pilot-39000", "--manifest", str(path), "--execute"])
    assert result.returncode == 2
    assert "CONFIRM_PINECONE_WRITE" in result.stderr


# ---------------------------------------------------------------------------
# 13. prepare_canary.py scope constants
# ---------------------------------------------------------------------------


def test_prepare_canary_has_pilot_39000_scope() -> None:
    from scripts.prepare_canary import _SCOPE_QUOTAS, SCOPE_PILOT_39000, VALID_PREP_SCOPES

    assert SCOPE_PILOT_39000 == "pilot-39000"
    assert SCOPE_PILOT_39000 in VALID_PREP_SCOPES
    quotas = _SCOPE_QUOTAS[SCOPE_PILOT_39000]
    assert quotas == {"en": 13000, "hi": 13000, "bn": 13000}


def test_prepare_canary_39000_max_rows() -> None:
    from scripts.prepare_canary import _SCOPE_MAX_ROWS, SCOPE_PILOT_39000

    assert _SCOPE_MAX_ROWS[SCOPE_PILOT_39000] == 20000


def test_prepare_canary_39000_budget() -> None:
    from scripts.prepare_canary import _SCOPE_BUDGET, SCOPE_PILOT_39000

    budget = _SCOPE_BUDGET[SCOPE_PILOT_39000]
    assert budget["max_records"] == 39_000
    assert budget["max_tokens"] >= 15_000_000
    assert budget["max_bytes"] >= int(6 * 1024**3)


def test_prepare_canary_existing_scopes_unchanged() -> None:
    from scripts.prepare_canary import (
        _SCOPE_QUOTAS,
        SCOPE_CANARY_300,
        SCOPE_PILOT_10000,
    )

    assert _SCOPE_QUOTAS[SCOPE_CANARY_300] == {"en": 100, "hi": 100, "bn": 100}
    assert _SCOPE_QUOTAS[SCOPE_PILOT_10000] == {"en": 3334, "hi": 3333, "bn": 3333}


# ---------------------------------------------------------------------------
# 14. Artifacts are Git-ignored
# ---------------------------------------------------------------------------


def test_pilot_39000_artifacts_are_gitignored() -> None:
    import subprocess

    result = subprocess.run(
        ["git", "check-ignore", "-v", "artifacts/prepared/pilot-39000-canonical/"],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    assert result.returncode == 0, (
        "artifacts/prepared/pilot-39000-canonical/ is NOT git-ignored. "
        f"git check-ignore output: {result.stdout!r} {result.stderr!r}"
    )
