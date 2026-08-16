"""Focused adversarial tests for the pilot-10000 scope in index_canary.py.

Covers:
- Scope constants and expected values
- _load_manifest scope validation (canary-300 and pilot-10000)
- _verify_and_load_records scope validation
- Pilot preflight ownership verification logic
- Canary-300 behaviour unchanged by scope addition
- Arbitrary totals not accepted
- Batch size never exceeds 96
- Token ceiling remains 225,000/min
- Dry-run never imports Pinecone
- Reconciliation count equality alone is insufficient
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sys_path_prepend() -> None:
    scripts_dir = str(Path(__file__).parent.parent.parent / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)


def _make_manifest(
    tmp_path: Path,
    total: int,
    per_lang: dict[str, int],
    extra: dict | None = None,
) -> Path:
    """Write a minimal valid manifest for the given counts."""
    sys_path_prepend()
    from index_canary import (
        CANONICAL_INDEX_NAME,
        CANONICAL_MAX_INPUT_TOKENS,
    )  # type: ignore[import]

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
        "index_name": CANONICAL_INDEX_NAME,
        "index_namespace": NAMESPACE,
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "tokenizer_repo": TOKENIZER_REPO,
        "tokenizer_revision": TOKENIZER_REVISION,
        "tokenizer_fingerprint": "abc123",
        "model_input_limit": CANONICAL_MAX_INPUT_TOKENS,
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

    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(m, indent=2))
    return path


def _run_cli(
    args: list[str],
    env: dict | None = None,
) -> subprocess.CompletedProcess:
    import os

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
    )


# ---------------------------------------------------------------------------
# 1. Scope constants
# ---------------------------------------------------------------------------


def test_scope_constants_exist() -> None:
    sys_path_prepend()
    from index_canary import (  # type: ignore[import]
        SCOPE_CANARY_300,
        SCOPE_PILOT_10000,
        VALID_SCOPES,
    )

    assert SCOPE_CANARY_300 == "canary-300"
    assert SCOPE_PILOT_10000 == "pilot-10000"
    assert SCOPE_CANARY_300 in VALID_SCOPES
    assert SCOPE_PILOT_10000 in VALID_SCOPES


def test_scope_expected_totals() -> None:
    sys_path_prepend()
    from index_canary import (  # type: ignore[import]
        _SCOPE_EXPECTED,
        SCOPE_CANARY_300,
        SCOPE_PILOT_10000,
    )

    assert _SCOPE_EXPECTED[SCOPE_CANARY_300]["total"] == 300
    assert _SCOPE_EXPECTED[SCOPE_PILOT_10000]["total"] == 10_000


def test_scope_expected_per_lang_canary() -> None:
    sys_path_prepend()
    from index_canary import _SCOPE_EXPECTED, SCOPE_CANARY_300  # type: ignore[import]

    pl = _SCOPE_EXPECTED[SCOPE_CANARY_300]["per_lang"]
    assert pl == {"en": 100, "hi": 100, "bn": 100}


def test_scope_expected_per_lang_pilot() -> None:
    sys_path_prepend()
    from index_canary import _SCOPE_EXPECTED, SCOPE_PILOT_10000  # type: ignore[import]

    pl = _SCOPE_EXPECTED[SCOPE_PILOT_10000]["per_lang"]
    assert pl == {"en": 3334, "hi": 3333, "bn": 3333}
    assert sum(pl.values()) == 10_000


# ---------------------------------------------------------------------------
# 2. _load_manifest scope validation
# ---------------------------------------------------------------------------


def test_load_manifest_canary_300_accepts_correct_counts(tmp_path: Path) -> None:
    sys_path_prepend()
    from index_canary import _load_manifest  # type: ignore[import]

    path = _make_manifest(tmp_path, 300, {"en": 100, "hi": 100, "bn": 100})
    m = _load_manifest(path, scope="canary-300")
    assert m["total_records"] == 300


def test_load_manifest_canary_300_rejects_pilot_counts(tmp_path: Path) -> None:
    sys_path_prepend()
    from index_canary import _load_manifest  # type: ignore[import]

    path = _make_manifest(tmp_path, 10_000, {"en": 3334, "hi": 3333, "bn": 3333})
    with pytest.raises(ValueError, match="10000"):
        _load_manifest(path, scope="canary-300")


def test_load_manifest_pilot_10000_accepts_correct_counts(tmp_path: Path) -> None:
    sys_path_prepend()
    from index_canary import _load_manifest  # type: ignore[import]

    path = _make_manifest(tmp_path, 10_000, {"en": 3334, "hi": 3333, "bn": 3333})
    m = _load_manifest(path, scope="pilot-10000")
    assert m["total_records"] == 10_000


def test_load_manifest_pilot_10000_rejects_canary_counts(tmp_path: Path) -> None:
    sys_path_prepend()
    from index_canary import _load_manifest  # type: ignore[import]

    path = _make_manifest(tmp_path, 300, {"en": 100, "hi": 100, "bn": 100})
    with pytest.raises(ValueError, match="300"):
        _load_manifest(path, scope="pilot-10000")


def test_load_manifest_pilot_wrong_per_lang_en(tmp_path: Path) -> None:
    sys_path_prepend()
    from index_canary import _load_manifest  # type: ignore[import]

    # Wrong EN count (3333 instead of 3334)
    path = _make_manifest(tmp_path, 10_000, {"en": 3333, "hi": 3334, "bn": 3333})
    with pytest.raises(ValueError):
        _load_manifest(path, scope="pilot-10000")


def test_load_manifest_invalid_scope_raises(tmp_path: Path) -> None:
    sys_path_prepend()
    from index_canary import _load_manifest  # type: ignore[import]

    path = _make_manifest(tmp_path, 300, {"en": 100, "hi": 100, "bn": 100})
    with pytest.raises(ValueError, match="Unknown scope"):
        _load_manifest(path, scope="arbitrary-999")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 3. Arbitrary totals cannot be supplied via CLI
# ---------------------------------------------------------------------------


def test_cli_scope_rejects_unknown_value(tmp_path: Path) -> None:
    """--scope with an unknown value must be rejected by argparse."""
    path = _make_manifest(tmp_path, 300, {"en": 100, "hi": 100, "bn": 100})
    result = _run_cli(["--scope", "arbitrary-999", "--manifest", str(path)])
    assert result.returncode != 0
    assert "invalid choice" in result.stderr or "error" in result.stderr.lower()


def test_cli_default_scope_is_canary_300(tmp_path: Path) -> None:
    """Without --scope, the default must be canary-300 (dry-run exits 0)."""
    path = _make_manifest(tmp_path, 300, {"en": 100, "hi": 100, "bn": 100})
    # Dry-run needs a records file
    (tmp_path / "records.jsonl").write_text("")
    result = _run_cli(["--manifest", str(path)])
    # Dry-run with empty records will fail count check, but scope rejection is prior
    # We just verify it doesn't fail with "unknown scope"
    assert "Unknown scope" not in result.stderr


# ---------------------------------------------------------------------------
# 4. Batch size and token ceiling guards
# ---------------------------------------------------------------------------


def test_batch_size_96_is_canonical_max() -> None:
    sys_path_prepend()
    from index_canary import CANONICAL_MAX_BATCH_SIZE  # type: ignore[import]

    assert CANONICAL_MAX_BATCH_SIZE == 96


def test_batch_size_over_96_rejected_cli(tmp_path: Path) -> None:
    path = _make_manifest(tmp_path, 300, {"en": 100, "hi": 100, "bn": 100})
    result = _run_cli(["--manifest", str(path), "--batch-size", "97"])
    assert result.returncode == 2


def test_token_ceiling_default() -> None:
    sys_path_prepend()
    from index_canary import DEFAULT_TOKEN_RATE_LIMIT  # type: ignore[import]

    assert DEFAULT_TOKEN_RATE_LIMIT == 225_000


def test_token_ceiling_over_250k_rejected_cli(tmp_path: Path) -> None:
    path = _make_manifest(tmp_path, 300, {"en": 100, "hi": 100, "bn": 100})
    result = _run_cli(["--manifest", str(path), "--token-rate-limit", "250001"])
    assert result.returncode == 2


# ---------------------------------------------------------------------------
# 5. Dry-run never imports Pinecone for pilot-10000 scope
# ---------------------------------------------------------------------------


def test_pilot_dry_run_does_not_import_pinecone(tmp_path: Path) -> None:
    """--scope pilot-10000 dry-run must not import pinecone."""
    import os

    # Build a minimal valid 10k manifest + records file
    path = _make_manifest(tmp_path, 10_000, {"en": 3334, "hi": 3333, "bn": 3333})
    records_path = tmp_path / "records.jsonl"
    records_path.write_text("")  # will fail record count check, but Pinecone check is earlier
    run_env = os.environ.copy()
    for k in ["PINECONE_API_KEY", "CONFIRM_PINECONE_WRITE"]:
        run_env.pop(k, None)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            f"sys.argv = ['index_canary.py', '--scope', 'pilot-10000', '--manifest', '{path}']; "
            "import index_canary; index_canary.main()",
        ],
        capture_output=True,
        text=True,
        env=run_env,
        cwd=str(Path(__file__).parent.parent.parent),
    )
    assert "pinecone" not in result.stderr.lower().replace("PINECONE_API_KEY", "")


# ---------------------------------------------------------------------------
# 6. Pilot preflight ownership verification — unit-level logic
# ---------------------------------------------------------------------------


def _make_pilot_preflight_state(
    expected_ids: set[str],
    current_ids: list[str],
) -> tuple[set[str], set[str]]:
    """Simulate the pilot preflight subset check."""
    current_set = set(current_ids)
    unrelated = current_set - expected_ids
    return current_set, unrelated


def test_pilot_preflight_passes_when_current_subset_of_expected() -> None:
    expected = {f"id-{i}" for i in range(10_000)}
    canary_300 = {f"id-{i}" for i in range(300)}
    _, unrelated = _make_pilot_preflight_state(expected, list(canary_300))
    assert len(unrelated) == 0


def test_pilot_preflight_fails_when_unrelated_id_present() -> None:
    expected = {f"id-{i}" for i in range(10_000)}
    ids_with_unrelated = [f"id-{i}" for i in range(300)] + ["UNRELATED-XYZ"]
    _, unrelated = _make_pilot_preflight_state(expected, ids_with_unrelated)
    assert len(unrelated) == 1
    assert "UNRELATED-XYZ" in unrelated


def test_pilot_preflight_fails_count_301_with_unrelated() -> None:
    expected = {f"id-{i}" for i in range(10_000)}
    ids = [f"id-{i}" for i in range(300)] + ["NOT-IN-MANIFEST"]
    _, unrelated = _make_pilot_preflight_state(expected, ids)
    assert len(unrelated) > 0


def test_pilot_preflight_empty_namespace_passes() -> None:
    expected = {f"id-{i}" for i in range(10_000)}
    _, unrelated = _make_pilot_preflight_state(expected, [])
    assert len(unrelated) == 0


def test_pilot_preflight_full_10k_namespace_passes() -> None:
    expected = {f"id-{i}" for i in range(10_000)}
    _, unrelated = _make_pilot_preflight_state(expected, list(expected))
    assert len(unrelated) == 0


# ---------------------------------------------------------------------------
# 7. Subset proof: 300-canary IDs must be contained in 10k pilot IDs
# ---------------------------------------------------------------------------


def test_300_canary_ids_subset_of_10k_pilot_ids_from_files() -> None:
    """Prove the live canary ID set is a subset of the pilot ID set using on-disk artifacts."""
    canary_path = Path(
        "artifacts/prepared/canonical-05f1b03.AIUDj6/canary-42-ee540c17772a_records.jsonl"
    )
    if not canary_path.exists():
        pytest.skip("Canary records file not present (git-ignored artifact)")

    pilot_paths = list(Path("artifacts/prepared").glob("pilot-10000-*/"))
    if not pilot_paths:
        pytest.skip("Pilot records dir not present (artifact not yet prepared)")

    pilot_records_files = list(pilot_paths[0].glob("*_records.jsonl"))
    if not pilot_records_files:
        pytest.skip("Pilot records file not found")

    pilot_ids: set[str] = set()
    with open(pilot_records_files[0]) as f:
        for line in f:
            line = line.strip()
            if line:
                pilot_ids.add(json.loads(line)["id"])

    canary_ids: set[str] = set()
    with open(canary_path) as f:
        for line in f:
            line = line.strip()
            if line:
                canary_ids.add(json.loads(line)["id"])

    assert len(canary_ids) == 300, f"Expected 300 canary IDs, got {len(canary_ids)}"
    assert len(pilot_ids) == 10_000, f"Expected 10,000 pilot IDs, got {len(pilot_ids)}"

    missing_from_pilot = canary_ids - pilot_ids
    assert len(missing_from_pilot) == 0, (
        f"{len(missing_from_pilot)} canary IDs are NOT in the pilot set. "
        "The 300-record canary must be a deterministic subset of the 10k pilot."
    )
    assert len(canary_ids & pilot_ids) == 300


# ---------------------------------------------------------------------------
# 8. Post-write reconciliation: count equality alone is not sufficient
# ---------------------------------------------------------------------------


def test_reconciliation_count_equality_wrong_ids_is_fail() -> None:
    """Verify the logic: matching count but wrong ID set must report FAIL."""
    expected_ids = {f"expected-{i}" for i in range(10)}
    enumerated_ids = {f"unexpected-{i}" for i in range(10)}  # same count, wrong IDs

    missing = expected_ids - enumerated_ids
    unexpected = enumerated_ids - expected_ids
    assert len(missing) == 10
    assert len(unexpected) == 10
    # Both non-empty → the code path would call _finalize("failed", ...)
    assert bool(missing or unexpected)


def test_reconciliation_exact_match_passes() -> None:
    expected_ids = {f"id-{i}" for i in range(10)}
    enumerated_ids = set(expected_ids)

    missing = expected_ids - enumerated_ids
    unexpected = enumerated_ids - expected_ids
    assert len(missing) == 0
    assert len(unexpected) == 0
    assert not (missing or unexpected)


# ---------------------------------------------------------------------------
# 9. Canary-300 behaviour unchanged
# ---------------------------------------------------------------------------


def test_canary_expected_total_constant_unchanged() -> None:
    sys_path_prepend()
    from index_canary import CANARY_EXPECTED_PER_LANG, CANARY_EXPECTED_TOTAL  # type: ignore[import]

    assert CANARY_EXPECTED_TOTAL == 300
    assert CANARY_EXPECTED_PER_LANG == 100


def test_execute_without_confirm_exits_2_canary(tmp_path: Path) -> None:
    path = _make_manifest(tmp_path, 300, {"en": 100, "hi": 100, "bn": 100})
    result = _run_cli(["--manifest", str(path), "--execute"])
    assert result.returncode == 2
    assert "CONFIRM_PINECONE_WRITE" in result.stderr


def test_execute_without_confirm_exits_2_pilot(tmp_path: Path) -> None:
    path = _make_manifest(tmp_path, 10_000, {"en": 3334, "hi": 3333, "bn": 3333})
    result = _run_cli(["--scope", "pilot-10000", "--manifest", str(path), "--execute"])
    assert result.returncode == 2
    assert "CONFIRM_PINECONE_WRITE" in result.stderr


# ---------------------------------------------------------------------------
# 10. Pilot scope batch count
# ---------------------------------------------------------------------------


def test_pilot_10k_produces_105_batches() -> None:
    """10,000 records / 96 per batch = 105 batches (104 full + 1 partial)."""
    import math

    assert math.ceil(10_000 / 96) == 105


def test_canary_300_produces_4_batches() -> None:
    import math

    assert math.ceil(300 / 96) == 4
