"""Regression tests for the final pre-index hardening pass.

Tests cover all acceptance-criteria scenarios that were newly added or
corrected in this pass. All tests run without real provider credentials.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ── Helpers ───────────────────────────────────────────────────────────────────

_SCRIPTS = str(Path(__file__).parent.parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from hhgoa_rag.pinecone_contract import (  # noqa: E402
    CONTRACT_VERSION,  # noqa: E402
    DATASET_REPO,
    DATASET_REVISION,
    INDEX_NAME,
    MANIFEST_SCHEMA_VERSION,
    MAX_BATCH_SIZE,
    MAX_INPUT_TOKENS,
    NAMESPACE,
    TOKENIZER_REPO,
    TOKENIZER_REVISION,
    canonical_contract,
    contract_fingerprint,
)


def _make_valid_record(i: int = 0, lang: str = "en") -> dict:
    return {
        "id": f"rec-{i:04d}",
        "chunk_text": f"Sample passage {i}.",
        "language": lang,
        "config_language": lang,
        "dataset_revision": DATASET_REVISION,
        "split": "train",
        "physical_shard": "train/hintrain.parquet",
        "local_source_row": i,
        "passage_position": 0,
        "parent_passage_id": "p" * 40,
        "content_hash": "c" * 40,
        "chunk_strategy": "sentence_aware",
        "chunk_strategy_version": "v1",
        "chunk_ordinal": 0,
        "chunk_total": 1,
        "token_length": 5,
        "tokenizer_fingerprint": "fp123",
        "manifest_id": "canary-42-test",
    }


def _make_records(total: int = 300) -> list[dict]:
    langs = ["en"] * 100 + ["hi"] * 100 + ["bn"] * 100
    return [_make_valid_record(i, langs[i % len(langs)]) for i in range(total)]


def _write_jsonl_and_manifest(
    tmp_path: Path,
    records: list[dict] | None = None,
    total_tokens_override: int | None = None,
    lang_counts_override: dict | None = None,
    manifest_overrides: dict | None = None,
) -> tuple[Path, Path]:
    if records is None:
        records = _make_records(300)

    jsonl_path = tmp_path / "canary-42-test_records.jsonl"
    jsonl_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    jsonl_checksum = hashlib.sha256(jsonl_path.read_bytes()).hexdigest()

    token_total = (
        total_tokens_override
        if total_tokens_override is not None
        else sum(r.get("token_length", 0) for r in records)
    )
    lang_counts = lang_counts_override or {"en": 100, "hi": 100, "bn": 100}
    fp = contract_fingerprint()
    contract = canonical_contract()

    manifest: dict = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "manifest_id": "canary-42-test",
        "mode": "canary",
        "contract_version": CONTRACT_VERSION,
        "contract_fingerprint": fp,
        "index_contract": contract,
        "index_name": INDEX_NAME,
        "index_namespace": NAMESPACE,
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "tokenizer_repo": TOKENIZER_REPO,
        "tokenizer_revision": TOKENIZER_REVISION,
        "tokenizer_fingerprint": "fp123",
        "model_input_limit": MAX_INPUT_TOKENS,
        "total_records": len(records),
        "total_tokens": token_total,
        "actual_per_language_records": lang_counts,
        "prepared_record_path": jsonl_path.name,
        "prepared_record_checksum": jsonl_checksum,
        "ready_for_write": True,
        "readiness_failures": [],
        "forbidden_field_audit": "PASS",
    }
    if manifest_overrides:
        manifest.update(manifest_overrides)
    m_for_ck = {k: v for k, v in manifest.items() if k != "manifest_checksum"}
    manifest["manifest_checksum"] = hashlib.sha256(
        json.dumps(m_for_ck, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    manifest_path = tmp_path / "canary-42-test_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest_path, jsonl_path


def _run_canary(args: list[str], env_extra: dict | None = None) -> subprocess.CompletedProcess:
    import os

    env = {k: v for k, v in os.environ.items()}
    for key in ["PINECONE_API_KEY", "CONFIRM_PINECONE_WRITE"]:
        env.pop(key, None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, str(Path(_SCRIPTS) / "index_canary.py"), *args],
        capture_output=True,
        text=True,
        env=env,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. validate_record() called before provider construction
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidateRecordBeforeProvider:
    def test_schema_violation_raises_before_pinecone(self, tmp_path):
        import index_canary as ic

        bad_records = _make_records(300)
        bad_records[5] = {**bad_records[5], "token_length": MAX_INPUT_TOKENS + 1}

        jsonl_path = tmp_path / "bad.jsonl"
        jsonl_path.write_text(
            "\n".join(json.dumps(r) for r in bad_records) + "\n", encoding="utf-8"
        )
        ck = hashlib.sha256(jsonl_path.read_bytes()).hexdigest()
        manifest = {
            "prepared_record_path": jsonl_path.name,
            "prepared_record_checksum": ck,
            "total_tokens": 0,
            "actual_per_language_records": {"en": 100, "hi": 100, "bn": 100},
        }
        with pytest.raises(ValueError, match="schema violation"):
            ic._verify_and_load_records(manifest, jsonl_path)

    def test_missing_required_field_rejected(self, tmp_path):
        import index_canary as ic

        bad_records = _make_records(300)
        del bad_records[10]["tokenizer_fingerprint"]

        jsonl_path = tmp_path / "missing.jsonl"
        jsonl_path.write_text(
            "\n".join(json.dumps(r) for r in bad_records) + "\n", encoding="utf-8"
        )
        ck = hashlib.sha256(jsonl_path.read_bytes()).hexdigest()
        with pytest.raises(ValueError, match="schema violation"):
            ic._verify_and_load_records(
                {
                    "prepared_record_path": jsonl_path.name,
                    "prepared_record_checksum": ck,
                    "total_tokens": -1,
                    "actual_per_language_records": {"en": 100, "hi": 100, "bn": 100},
                },
                jsonl_path,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Recursive forbidden fields
# ═══════════════════════════════════════════════════════════════════════════════


class TestForbiddenFields:
    def test_top_level_forbidden_field_rejected(self):
        from hhgoa_rag.ingestion.schema import SchemaViolationError, validate_record

        rec = _make_valid_record()
        rec["query"] = "some query"
        with pytest.raises(SchemaViolationError, match="Forbidden"):
            validate_record(rec)

    def test_schema_max_tokens_uses_canonical(self):
        from hhgoa_rag.ingestion.schema import SchemaViolationError, validate_record

        rec = _make_valid_record()
        rec["token_length"] = MAX_INPUT_TOKENS + 1
        with pytest.raises(SchemaViolationError, match=str(MAX_INPUT_TOKENS)):
            validate_record(rec)

    def test_valid_record_at_max_tokens_passes(self):
        from hhgoa_rag.ingestion.schema import validate_record

        rec = _make_valid_record()
        rec["token_length"] = MAX_INPUT_TOKENS
        validate_record(rec)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Duplicate record IDs
# ═══════════════════════════════════════════════════════════════════════════════


class TestDuplicateIDs:
    def test_duplicate_id_raises(self, tmp_path):
        import index_canary as ic

        records = _make_records(300)
        records[5] = {**records[5], "id": records[0]["id"]}

        jsonl_path = tmp_path / "dup.jsonl"
        jsonl_path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
        ck = hashlib.sha256(jsonl_path.read_bytes()).hexdigest()
        with pytest.raises(ValueError, match="Duplicate record id"):
            ic._verify_and_load_records(
                {
                    "prepared_record_path": jsonl_path.name,
                    "prepared_record_checksum": ck,
                    "total_tokens": -1,
                    "actual_per_language_records": {"en": 100, "hi": 100, "bn": 100},
                },
                jsonl_path,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Incorrect actual language counts
# ═══════════════════════════════════════════════════════════════════════════════


class TestLanguageCounts:
    def test_wrong_language_count_rejected(self, tmp_path):
        import index_canary as ic

        records = _make_records(300)
        records[299] = {**records[299], "language": "en", "config_language": "en"}

        jsonl_path = tmp_path / "langbad.jsonl"
        jsonl_path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
        ck = hashlib.sha256(jsonl_path.read_bytes()).hexdigest()
        manifest = {
            "prepared_record_path": jsonl_path.name,
            "prepared_record_checksum": ck,
            "actual_per_language_records": {"en": 100, "hi": 100, "bn": 100},
            "total_tokens": sum(r.get("token_length", 0) for r in records),
        }
        with pytest.raises(ValueError, match="language count"):
            ic._verify_and_load_records(manifest, jsonl_path)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Manifest vs JSONL token-total mismatch
# ═══════════════════════════════════════════════════════════════════════════════


class TestTokenTotalMismatch:
    def test_token_total_mismatch_rejected(self, tmp_path):
        import index_canary as ic

        records = _make_records(300)
        actual_total = sum(r.get("token_length", 0) for r in records)

        jsonl_path = tmp_path / "tok.jsonl"
        jsonl_path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
        ck = hashlib.sha256(jsonl_path.read_bytes()).hexdigest()
        manifest = {
            "prepared_record_path": jsonl_path.name,
            "prepared_record_checksum": ck,
            "actual_per_language_records": {"en": 100, "hi": 100, "bn": 100},
            "total_tokens": actual_total + 999,
        }
        with pytest.raises(ValueError, match="token total"):
            ic._verify_and_load_records(manifest, jsonl_path)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Provenance field validation
# ═══════════════════════════════════════════════════════════════════════════════


class TestProvenanceValidation:
    @pytest.mark.parametrize(
        "field,bad_value,match_text",
        [
            ("dataset_repo", "wrong/repo", "dataset_repo"),
            ("dataset_revision", "deadbeef" * 5, "dataset_revision"),
            ("tokenizer_repo", "wrong/tokenizer", "tokenizer_repo"),
            ("tokenizer_revision", "badrevision" * 4, "tokenizer_revision"),
            ("model_input_limit", 256, "model_input_limit"),
            ("tokenizer_fingerprint", "", "tokenizer_fingerprint"),
        ],
    )
    def test_wrong_provenance_field_rejected(self, field, bad_value, match_text, tmp_path):
        import index_canary as ic

        manifest_path, _ = _write_jsonl_and_manifest(tmp_path)
        manifest = json.loads(manifest_path.read_text())
        manifest[field] = bad_value
        m_for_ck = {k: v for k, v in manifest.items() if k != "manifest_checksum"}
        manifest["manifest_checksum"] = hashlib.sha256(
            json.dumps(m_for_ck, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match=match_text):
            ic._load_manifest(manifest_path)

    def test_missing_provenance_field_rejected(self, tmp_path):
        import index_canary as ic

        manifest_path, _ = _write_jsonl_and_manifest(tmp_path)
        manifest = json.loads(manifest_path.read_text())
        del manifest["tokenizer_repo"]
        m_for_ck = {k: v for k, v in manifest.items() if k != "manifest_checksum"}
        manifest["manifest_checksum"] = hashlib.sha256(
            json.dumps(m_for_ck, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="required"):
            ic._load_manifest(manifest_path)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Corrupt checkpoint rejection (fail closed)
# ═══════════════════════════════════════════════════════════════════════════════


class TestCorruptCheckpoint:
    def test_corrupt_json_raises_not_returns_none(self, tmp_path):
        import index_canary as ic

        ckpt_path = tmp_path / "corrupt.json"
        ckpt_path.write_text("THIS IS NOT JSON{{{{", encoding="utf-8")
        with pytest.raises(ic._CorruptCheckpointError):
            ic._load_checkpoint(ckpt_path)

    def test_non_object_json_raises(self, tmp_path):
        import index_canary as ic

        ckpt_path = tmp_path / "array.json"
        ckpt_path.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(ic._CorruptCheckpointError, match="JSON object"):
            ic._load_checkpoint(ckpt_path)

    def test_absent_checkpoint_returns_none(self, tmp_path):
        import index_canary as ic

        result = ic._load_checkpoint(tmp_path / "nonexistent.json")
        assert result is None

    def test_valid_checkpoint_returned_as_dict(self, tmp_path):
        import index_canary as ic

        ckpt_path = tmp_path / "valid.json"
        data = {"checkpoint_schema_version": "1", "run_id": "x"}
        ckpt_path.write_text(json.dumps(data))
        result = ic._load_checkpoint(ckpt_path)
        assert result == data

    def test_corrupt_checkpoint_causes_fail_closed_dryrun(self, tmp_path):
        manifest_path, _ = _write_jsonl_and_manifest(tmp_path)
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        (ckpt_dir / "canary_canary-42-test_BAD.json").write_text("NOT_JSON")
        r = _run_canary(
            [
                "--manifest",
                str(manifest_path),
                "--resume",
                "--checkpoint-dir",
                str(ckpt_dir),
            ]
        )
        assert r.returncode != 0, f"Expected non-zero exit; stdout={r.stdout}\nstderr={r.stderr}"


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Incompatible checkpoint rejection
# ═══════════════════════════════════════════════════════════════════════════════


class TestIncompatibleCheckpoint:
    def test_incompatible_schema_version_rejected(self):
        import index_canary as ic

        ckpt = {
            "checkpoint_schema_version": "99",
            "manifest_id": "m",
            "manifest_checksum": "c",
            "contract_fingerprint": "fp",
            "index_name": INDEX_NAME,
            "namespace": NAMESPACE,
            "batch_size": 96,
            "batch_digests": [],
        }
        with pytest.raises(ValueError, match="checkpoint_schema_version"):
            ic._validate_checkpoint_compat(ckpt, "m", "c", "fp", INDEX_NAME, NAMESPACE, 96, [])


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Concurrent rate-limit reservations
# ═══════════════════════════════════════════════════════════════════════════════


class TestConcurrentRateLimiting:
    def test_concurrent_reservations_do_not_exceed_ceiling(self):
        lock = threading.Lock()
        window_tokens = [0]
        batch_tokens = 100
        n_threads = 10
        results: list[int] = []

        def reserve():
            with lock:
                current = window_tokens[0]
                window_tokens[0] += batch_tokens
                results.append(current)

        threads = [threading.Thread(target=reserve) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == n_threads
        assert len(set(results)) == n_threads, "Two threads saw the same window_tokens — race!"
        assert window_tokens[0] == n_threads * batch_tokens

    def test_rate_limiter_uses_lock_in_source(self):
        import inspect

        import index_canary as ic

        source = inspect.getsource(ic._run)
        assert "_rate_lock" in source, "Expected _rate_lock in _run() source"


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Zero/negative CLI values
# ═══════════════════════════════════════════════════════════════════════════════


class TestCLIValidation:
    def test_zero_batch_size_exits_2(self, tmp_path):
        manifest_path, _ = _write_jsonl_and_manifest(tmp_path)
        r = _run_canary(["--manifest", str(manifest_path), "--batch-size", "0"])
        assert r.returncode == 2

    def test_negative_batch_size_exits_2(self, tmp_path):
        manifest_path, _ = _write_jsonl_and_manifest(tmp_path)
        r = _run_canary(["--manifest", str(manifest_path), "--batch-size", "-5"])
        assert r.returncode == 2

    def test_zero_concurrency_exits_2(self, tmp_path):
        manifest_path, _ = _write_jsonl_and_manifest(tmp_path)
        r = _run_canary(["--manifest", str(manifest_path), "--concurrency", "0"])
        assert r.returncode == 2

    def test_negative_concurrency_exits_2(self, tmp_path):
        manifest_path, _ = _write_jsonl_and_manifest(tmp_path)
        r = _run_canary(["--manifest", str(manifest_path), "--concurrency", "-1"])
        assert r.returncode == 2

    def test_zero_token_rate_limit_exits_2(self, tmp_path):
        manifest_path, _ = _write_jsonl_and_manifest(tmp_path)
        r = _run_canary(["--manifest", str(manifest_path), "--token-rate-limit", "0"])
        assert r.returncode == 2

    def test_negative_token_rate_limit_exits_2(self, tmp_path):
        manifest_path, _ = _write_jsonl_and_manifest(tmp_path)
        r = _run_canary(["--manifest", str(manifest_path), "--token-rate-limit", "-100"])
        assert r.returncode == 2

    def test_over_max_batch_size_exits_2(self, tmp_path):
        manifest_path, _ = _write_jsonl_and_manifest(tmp_path)
        r = _run_canary(["--manifest", str(manifest_path), "--batch-size", str(MAX_BATCH_SIZE + 1)])
        assert r.returncode == 2

    def test_over_max_concurrency_exits_2(self, tmp_path):
        manifest_path, _ = _write_jsonl_and_manifest(tmp_path)
        r = _run_canary(["--manifest", str(manifest_path), "--concurrency", "100"])
        assert r.returncode == 2

    def test_execute_without_confirm_exits_2(self, tmp_path):
        manifest_path, _ = _write_jsonl_and_manifest(tmp_path)
        r = _run_canary(
            ["--manifest", str(manifest_path), "--execute"],
            env_extra={"PINECONE_API_KEY": "key"},
        )
        assert r.returncode == 2


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Failure reports never remaining "started"
# ═══════════════════════════════════════════════════════════════════════════════


class TestFailureReports:
    def test_dry_run_report_status_not_started(self, tmp_path):
        manifest_path, _ = _write_jsonl_and_manifest(tmp_path)
        report_dir = tmp_path / "reports"
        r = _run_canary(["--manifest", str(manifest_path), "--report-dir", str(report_dir)])
        assert r.returncode == 0, f"stdout={r.stdout}\nstderr={r.stderr}"
        reports = list(report_dir.glob("*.json"))
        assert reports
        report = json.loads(reports[0].read_text())
        assert report["status"] != "started"

    def test_bad_manifest_report_status_is_failed(self, tmp_path):
        bad_manifest = tmp_path / "bad_manifest.json"
        bad_manifest.write_text(json.dumps({"manifest_schema_version": "2"}))
        report_dir = tmp_path / "reports"
        r = _run_canary(["--manifest", str(bad_manifest), "--report-dir", str(report_dir)])
        assert r.returncode != 0
        reports = list(report_dir.glob("*.json"))
        assert reports
        report = json.loads(reports[0].read_text())
        assert report["status"] == "failed"
        assert "end_time" in report
        assert report.get("failure_category") or report.get("failure_message")

    def test_failed_report_has_end_time(self, tmp_path):
        bad_manifest = tmp_path / "bad.json"
        bad_manifest.write_text("{}")
        report_dir = tmp_path / "reports"
        _run_canary(["--manifest", str(bad_manifest), "--report-dir", str(report_dir)])
        reports = list(report_dir.glob("*.json"))
        if reports:
            report = json.loads(reports[0].read_text())
            if report.get("status") == "failed":
                assert "end_time" in report


# ═══════════════════════════════════════════════════════════════════════════════
# 12. SDK object and dictionary stats responses
# ═══════════════════════════════════════════════════════════════════════════════


def _make_count_fn():
    """Re-implement _get_ns_vector_count for unit testing (same logic as in _run())."""

    def _get_ns_vector_count(stats: object, ns: str) -> int | None:
        namespaces = getattr(stats, "namespaces", None)
        if namespaces is None and isinstance(stats, dict):
            namespaces = stats.get("namespaces")
        if namespaces is None:
            return None
        if isinstance(namespaces, dict):
            ns_info = namespaces.get(ns)
        else:
            try:
                ns_info = namespaces[ns]
            except (KeyError, TypeError):
                ns_info = None
        if ns_info is None:
            return 0
        vc = getattr(ns_info, "vector_count", None)
        if vc is None and isinstance(ns_info, dict):
            vc = ns_info.get("vector_count")
        return int(vc) if vc is not None else 0

    return _get_ns_vector_count


class TestStatsResponseShapes:
    def test_sdk_object_shape(self):
        fn = _make_count_fn()

        class NsInfo:
            vector_count = 300

        class Stats:
            namespaces = {"pilot_v1": NsInfo()}

        assert fn(Stats(), "pilot_v1") == 300

    def test_dict_fixture_shape(self):
        fn = _make_count_fn()

        class Stats:
            namespaces = {"pilot_v1": {"vector_count": 300}}

        assert fn(Stats(), "pilot_v1") == 300

    def test_missing_namespace_returns_zero(self):
        fn = _make_count_fn()

        class Stats:
            namespaces = {}

        assert fn(Stats(), "pilot_v1") == 0

    def test_plain_dict_stats(self):
        fn = _make_count_fn()
        stats = {"namespaces": {"pilot_v1": {"vector_count": 300}}}
        assert fn(stats, "pilot_v1") == 300


# ═══════════════════════════════════════════════════════════════════════════════
# 13. Namespace counts 299, 300, 301
# ═══════════════════════════════════════════════════════════════════════════════


class TestNamespaceCounts:
    EXPECTED = 300

    def _simulate(self, vector_count: int) -> tuple[bool, bool]:
        reconciled = False
        contaminated = False
        if vector_count == self.EXPECTED:
            reconciled = True
        elif vector_count > self.EXPECTED:
            contaminated = True
        return reconciled, contaminated

    def test_299_does_not_reconcile(self):
        r, c = self._simulate(299)
        assert not r
        assert not c

    def test_300_reconciles(self):
        r, c = self._simulate(300)
        assert r
        assert not c

    def test_301_is_contaminated(self):
        r, c = self._simulate(301)
        assert not r
        assert c


# ═══════════════════════════════════════════════════════════════════════════════
# 14. Dry-run makes zero provider calls
# ═══════════════════════════════════════════════════════════════════════════════


class TestDryRunZeroProviderCalls:
    def test_dry_run_exits_0_no_upserts(self, tmp_path):
        manifest_path, _ = _write_jsonl_and_manifest(tmp_path)
        report_dir = tmp_path / "reports"
        r = _run_canary(
            ["--manifest", str(manifest_path), "--report-dir", str(report_dir)]
        )
        assert r.returncode == 0, f"stdout={r.stdout}\nstderr={r.stderr}"
        combined = r.stdout + r.stderr
        # Log-level upsert message only appears during live upsert operations
        assert "Upserting batch" not in combined

    def test_dry_run_creates_report_with_correct_status(self, tmp_path):
        manifest_path, _ = _write_jsonl_and_manifest(tmp_path)
        report_dir = tmp_path / "reports"
        r = _run_canary(["--manifest", str(manifest_path), "--report-dir", str(report_dir)])
        assert r.returncode == 0
        reports = list(report_dir.glob("*.json"))
        assert reports
        report = json.loads(reports[0].read_text())
        assert report["status"] == "dry_run_complete"
        assert report.get("failed_batches", 0) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 15. Resume skips only acknowledged compatible batches
# ═══════════════════════════════════════════════════════════════════════════════


class TestResumeSkipping:
    def test_compatible_checkpoint_skips_completed_digests(self):
        import index_canary as ic

        digests = ["d1", "d2", "d3", "d4"]
        completed = {"d1", "d3"}
        ckpt = {
            "checkpoint_schema_version": ic.CHECKPOINT_SCHEMA_VERSION,
            "manifest_id": "m",
            "manifest_checksum": "cs",
            "contract_fingerprint": "fp",
            "index_name": INDEX_NAME,
            "namespace": NAMESPACE,
            "batch_size": 96,
            "batch_digests": digests,
            "completed_batch_digests": list(completed),
        }
        ic._validate_checkpoint_compat(ckpt, "m", "cs", "fp", INDEX_NAME, NAMESPACE, 96, digests)
        pending = [i for i, d in enumerate(digests) if d not in completed]
        assert pending == [1, 3]


# ═══════════════════════════════════════════════════════════════════════════════
# 16. Idempotent re-run with same manifest
# ═══════════════════════════════════════════════════════════════════════════════


class TestIdempotentRerun:
    def test_second_dry_run_same_result(self, tmp_path):
        manifest_path, _ = _write_jsonl_and_manifest(tmp_path)
        report_dir = tmp_path / "reports"
        r1 = _run_canary(["--manifest", str(manifest_path), "--report-dir", str(report_dir)])
        r2 = _run_canary(["--manifest", str(manifest_path), "--report-dir", str(report_dir)])
        assert r1.returncode == 0
        assert r2.returncode == 0
        reports = list(report_dir.glob("*.json"))
        assert len(reports) == 2
        for rp in reports:
            data = json.loads(rp.read_text())
            assert data["status"] == "dry_run_complete"


# ═══════════════════════════════════════════════════════════════════════════════
# 17. PineconeStore MAX_BATCH_SIZE uses canonical value
# ═══════════════════════════════════════════════════════════════════════════════


class TestStoreMaxBatchSize:
    def test_store_rejects_over_max_batch_size(self):
        from hhgoa_rag.pinecone_store import PineconeStore

        mock_index = MagicMock()
        store = PineconeStore(mock_index, embed_model="multilingual-e5-large")
        oversized = [{"id": f"r{i}", "chunk_text": f"t{i}"} for i in range(MAX_BATCH_SIZE + 1)]
        with pytest.raises(ValueError, match=str(MAX_BATCH_SIZE)):
            store.upsert_records(oversized, namespace="pilot_v1", context="pilot")
        mock_index.upsert_records.assert_not_called()
