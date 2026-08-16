"""Adversarial tests for the final pre-index hardening pass (v4).

Covers:
  Fix 1 — Exact post-write ID reconciliation in index_canary.py.
  Fix 2 — ingest_prepared.py --execute is blocked before any Pinecone import.
  Fix 3 — prepare_canary manifest determinism (byte-identical across dirs/times).
  Fix 4 — PineconeStore.list_vector_ids pagination hardening.
  Fix 5 — PineconeStore.count_namespace fail-closed behaviour.
  Fix 6 — measure_corpus_capacity tokenizer-load failure fails closed.

No live provider calls are made anywhere in this file.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_SCRIPTS = str(Path(__file__).parent.parent.parent / "scripts")
_REPO_ROOT = Path(__file__).parent.parent.parent
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import index_canary as ic  # noqa: E402

from hhgoa_rag import pinecone_contract as pc_contract  # noqa: E402
from hhgoa_rag.pinecone_contract import (  # noqa: E402
    CONTRACT_VERSION,
    DATASET_REPO,
    DATASET_REVISION,
    MANIFEST_SCHEMA_VERSION,
    MAX_INPUT_TOKENS,
    TOKENIZER_REPO,
    TOKENIZER_REVISION,
    canonical_contract,
    contract_fingerprint,
)
from hhgoa_rag.pinecone_store import (  # noqa: E402
    PineconeProviderError,
    PineconeStore,
)

CANONICAL_INDEX_NAME = pc_contract.INDEX_NAME
CANONICAL_NAMESPACE = pc_contract.NAMESPACE


# ── Shared manifest/record builders ───────────────────────────────────────────


def _make_record(i: int, lang: str) -> dict:
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
    return [_make_record(i, langs[i]) for i in range(total)]


def _write_manifest(tmp_path: Path, records: list[dict]) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    jsonl_path = tmp_path / "canary-42-test_records.jsonl"
    jsonl_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    jsonl_checksum = hashlib.sha256(jsonl_path.read_bytes()).hexdigest()
    token_total = sum(r.get("token_length", 0) for r in records)
    manifest: dict = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "manifest_id": "canary-42-test",
        "mode": "canary",
        "contract_version": CONTRACT_VERSION,
        "contract_fingerprint": contract_fingerprint(),
        "index_contract": canonical_contract(),
        "index_name": CANONICAL_INDEX_NAME,
        "index_namespace": CANONICAL_NAMESPACE,
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "tokenizer_repo": TOKENIZER_REPO,
        "tokenizer_revision": TOKENIZER_REVISION,
        "tokenizer_fingerprint": "fp123",
        "model_input_limit": MAX_INPUT_TOKENS,
        "total_records": len(records),
        "total_tokens": token_total,
        "actual_per_language_records": {"en": 100, "hi": 100, "bn": 100},
        "prepared_record_path": jsonl_path.name,
        "prepared_record_checksum": jsonl_checksum,
        "ready_for_write": True,
        "readiness_failures": [],
        "forbidden_field_audit": "PASS",
    }
    m_for_ck = {k: v for k, v in manifest.items() if k != "manifest_checksum"}
    manifest["manifest_checksum"] = hashlib.sha256(
        json.dumps(m_for_ck, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    manifest_path = tmp_path / "canary-42-test_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest_path


class _FakeItem:
    def __init__(self, item_id: str) -> None:
        self.id = item_id


class _FakeResp:
    def __init__(self, ids: list[str], next_token: str | None = None) -> None:
        self.vectors = [_FakeItem(i) for i in ids]
        self.pagination = MagicMock(next=next_token) if next_token is not None else None


def _run_fresh_canary(monkeypatch, tmp_path, post_write_list_side):
    """Drive a fresh (non-resume) live canary run to Step 7 with mocked provider.

    post_write_list_side: a value to assign to mock_index.list_paginated
    (return_value handled by caller). Fresh preflight requires 0 IDs then this.
    """
    manifest_path = _write_manifest(tmp_path, _make_records(300))
    report_dir = tmp_path / "reports"

    mock_pc = MagicMock()
    mock_index = MagicMock()
    mock_pc.Index.return_value = mock_index
    # Fresh preflight: count 0; Step 7: count 300.
    mock_index.describe_index_stats.side_effect = [
        {"namespaces": {"pilot_v1": {"vector_count": 0}}},
        {"namespaces": {"pilot_v1": {"vector_count": 300}}},
    ]
    mock_index.upsert_records.return_value = 96
    # Preflight ID enumeration (fresh → empty), then post-write enumeration.
    mock_index.list_paginated.side_effect = [_FakeResp([]), post_write_list_side]

    monkeypatch.setenv("CONFIRM_PINECONE_WRITE", "1")
    monkeypatch.setenv("PINECONE_API_KEY", "pcsk_fake_key_for_testing")
    monkeypatch.setattr("pinecone.Pinecone", lambda api_key: mock_pc)
    monkeypatch.setattr("hhgoa_rag.pinecone_lifecycle.validate_index", lambda pc, name: [])

    parser = ic._build_parser()
    args = parser.parse_args(
        ["--manifest", str(manifest_path), "--execute", "--report-dir", str(report_dir)]
    )
    report_data: dict = {}
    return args, mock_index, report_data


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 1 — Exact post-write ID reconciliation
# ═══════════════════════════════════════════════════════════════════════════════


class TestPostWriteExactIdReconciliation:
    def test_count_300_exact_ids_passes(self, monkeypatch, tmp_path):
        all_ids = [f"rec-{i:04d}" for i in range(300)]
        args, mock_index, report_data = _run_fresh_canary(monkeypatch, tmp_path, _FakeResp(all_ids))
        ic._run(args, True, "r", "2026-08-16T00:00:00Z", "abc", report_data)
        assert report_data["status"] == "success"
        assert report_data["count_reconciliation"].startswith("PASS")
        assert report_data["exact_id_reconciliation"].startswith("PASS")

    def test_count_300_one_id_replaced_fails(self, monkeypatch, tmp_path):
        ids = [f"rec-{i:04d}" for i in range(299)] + ["unrelated-999"]
        args, mock_index, report_data = _run_fresh_canary(monkeypatch, tmp_path, _FakeResp(ids))
        with pytest.raises(ic.CanaryError) as exc:
            ic._run(args, True, "r", "2026-08-16T00:00:00Z", "abc", report_data)
        assert exc.value.category == "PostWriteOwnershipMismatch"
        assert report_data["status"] == "failed"

    def test_count_300_missing_and_extra_fails(self, monkeypatch, tmp_path):
        # 300 IDs but with one expected missing and one extra unexpected.
        ids = [f"rec-{i:04d}" for i in range(298)] + ["extra-a", "extra-b"]
        args, mock_index, report_data = _run_fresh_canary(monkeypatch, tmp_path, _FakeResp(ids))
        with pytest.raises(ic.CanaryError) as exc:
            ic._run(args, True, "r", "2026-08-16T00:00:00Z", "abc", report_data)
        assert exc.value.category == "PostWriteOwnershipMismatch"

    def test_count_below_300_fails_before_enumeration(self, monkeypatch, tmp_path):
        manifest_path = _write_manifest(tmp_path, _make_records(300))
        report_dir = tmp_path / "reports"
        mock_pc = MagicMock()
        mock_index = MagicMock()
        mock_pc.Index.return_value = mock_index
        mock_index.describe_index_stats.side_effect = [
            {"namespaces": {"pilot_v1": {"vector_count": 0}}},  # preflight
            {"namespaces": {"pilot_v1": {"vector_count": 299}}},  # step 7 (never reaches 300)
        ]
        mock_index.upsert_records.return_value = 96
        mock_index.list_paginated.side_effect = [_FakeResp([])]  # preflight only
        monkeypatch.setenv("CONFIRM_PINECONE_WRITE", "1")
        monkeypatch.setenv("PINECONE_API_KEY", "pcsk_fake")
        monkeypatch.setattr("pinecone.Pinecone", lambda api_key: mock_pc)
        monkeypatch.setattr("hhgoa_rag.pinecone_lifecycle.validate_index", lambda pc, name: [])
        # Shrink freshness polling so the timeout is fast.
        monkeypatch.setattr(ic, "FRESHNESS_POLL_MAX_WAIT", 1)
        monkeypatch.setattr(ic, "FRESHNESS_POLL_BASE", 1)
        parser = ic._build_parser()
        args = parser.parse_args(
            ["--manifest", str(manifest_path), "--execute", "--report-dir", str(report_dir)]
        )
        report_data: dict = {}
        with pytest.raises(ic.CanaryError) as exc:
            ic._run(args, True, "r", "2026-08-16T00:00:00Z", "abc", report_data)
        assert exc.value.category == "ReconciliationTimeout"
        assert report_data["exact_id_reconciliation"].startswith("NOT RUN")

    def test_count_above_300_contaminated(self, monkeypatch, tmp_path):
        manifest_path = _write_manifest(tmp_path, _make_records(300))
        report_dir = tmp_path / "reports"
        mock_pc = MagicMock()
        mock_index = MagicMock()
        mock_pc.Index.return_value = mock_index
        mock_index.describe_index_stats.side_effect = [
            {"namespaces": {"pilot_v1": {"vector_count": 0}}},
            {"namespaces": {"pilot_v1": {"vector_count": 301}}},
        ]
        mock_index.upsert_records.return_value = 96
        mock_index.list_paginated.side_effect = [_FakeResp([])]
        monkeypatch.setenv("CONFIRM_PINECONE_WRITE", "1")
        monkeypatch.setenv("PINECONE_API_KEY", "pcsk_fake")
        monkeypatch.setattr("pinecone.Pinecone", lambda api_key: mock_pc)
        monkeypatch.setattr("hhgoa_rag.pinecone_lifecycle.validate_index", lambda pc, name: [])
        parser = ic._build_parser()
        args = parser.parse_args(
            ["--manifest", str(manifest_path), "--execute", "--report-dir", str(report_dir)]
        )
        report_data: dict = {}
        with pytest.raises(ic.CanaryError) as exc:
            ic._run(args, True, "r", "2026-08-16T00:00:00Z", "abc", report_data)
        assert exc.value.category == "ContaminatedNamespace"

    def test_enumeration_exception_fails_closed(self, monkeypatch, tmp_path):
        args, mock_index, report_data = _run_fresh_canary(
            monkeypatch, tmp_path, _FakeResp([f"rec-{i:04d}" for i in range(300)])
        )
        # Preflight enumeration OK (empty), post-write enumeration raises.
        mock_index.list_paginated.side_effect = [
            _FakeResp([]),
            RuntimeError("provider exploded during list"),
        ]
        with pytest.raises(ic.CanaryError) as exc:
            ic._run(args, True, "r", "2026-08-16T00:00:00Z", "abc", report_data)
        assert exc.value.category == "PostWriteReconciliationUnverifiable"

    def test_unsupported_enumeration_fails_closed(self, monkeypatch, tmp_path):
        manifest_path = _write_manifest(tmp_path, _make_records(300))
        report_dir = tmp_path / "reports"
        mock_pc = MagicMock()
        mock_index = MagicMock()
        mock_pc.Index.return_value = mock_index
        mock_index.describe_index_stats.side_effect = [
            {"namespaces": {"pilot_v1": {"vector_count": 0}}},
            {"namespaces": {"pilot_v1": {"vector_count": 300}}},
        ]
        mock_index.upsert_records.return_value = 96
        monkeypatch.setenv("CONFIRM_PINECONE_WRITE", "1")
        monkeypatch.setenv("PINECONE_API_KEY", "pcsk_fake")
        monkeypatch.setattr("pinecone.Pinecone", lambda api_key: mock_pc)
        monkeypatch.setattr("hhgoa_rag.pinecone_lifecycle.validate_index", lambda pc, name: [])

        # Fresh preflight enumeration OK (0 IDs). Post-write enumeration reports the
        # provider does not support ID enumeration → fail closed.
        state = {"n": 0}

        def patched(self, namespace, prefix=None, limit=100):
            state["n"] += 1
            if state["n"] == 1:
                return []
            raise PineconeProviderError("Index object does not support ID enumeration")

        monkeypatch.setattr(PineconeStore, "list_vector_ids", patched)

        parser = ic._build_parser()
        args = parser.parse_args(
            ["--manifest", str(manifest_path), "--execute", "--report-dir", str(report_dir)]
        )
        report_data: dict = {}
        with pytest.raises(ic.CanaryError) as exc:
            ic._run(args, True, "r", "2026-08-16T00:00:00Z", "abc", report_data)
        assert exc.value.category == "PostWriteReconciliationUnverifiable"

    def test_reconciliation_failure_performs_no_extra_upsert(self, monkeypatch, tmp_path):
        # Post-write mismatch: the number of upserts must equal the number of batches
        # only (4), never more, after reconciliation fails.
        ids = [f"rec-{i:04d}" for i in range(299)] + ["rogue"]
        args, mock_index, report_data = _run_fresh_canary(monkeypatch, tmp_path, _FakeResp(ids))
        with pytest.raises(ic.CanaryError):
            ic._run(args, True, "r", "2026-08-16T00:00:00Z", "abc", report_data)
        # 300 records / 96 batch size = 4 batches → exactly 4 upsert calls, no more.
        assert mock_index.upsert_records.call_count == 4

    def test_success_report_records_exact_id_verification(self, monkeypatch, tmp_path):
        all_ids = [f"rec-{i:04d}" for i in range(300)]
        args, mock_index, report_data = _run_fresh_canary(monkeypatch, tmp_path, _FakeResp(all_ids))
        ic._run(args, True, "r", "2026-08-16T00:00:00Z", "abc", report_data)
        assert "exact_id_reconciliation" in report_data
        assert "count_reconciliation" in report_data
        assert report_data["exact_id_reconciliation"] == "PASS — 300 IDs match manifest exactly"


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 2 — Legacy live path blocked before any Pinecone import
# ═══════════════════════════════════════════════════════════════════════════════


class TestLegacyLivePathBlocked:
    def _make_manifest_subprocess(self, tmp_path: Path) -> Path:
        return _write_manifest(tmp_path, _make_records(300))

    def test_execute_blocked_even_with_confirm_and_key(self, tmp_path):
        manifest_path = self._make_manifest_subprocess(tmp_path)
        env = {
            "PINECONE_API_KEY": "pcsk_fake_should_never_be_used",
            "CONFIRM_PINECONE_WRITE": "1",
            "PATH": __import__("os").environ.get("PATH", ""),
        }
        result = subprocess.run(
            [
                sys.executable,
                str(Path(_SCRIPTS) / "ingest_prepared.py"),
                "--manifest",
                str(manifest_path),
                "--execute",
            ],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(_REPO_ROOT),
        )
        assert result.returncode == 2
        assert "DISABLED" in result.stderr
        assert "index_canary.py" in result.stderr

    def test_execute_never_imports_pinecone(self, tmp_path, monkeypatch):
        """The --execute refusal must occur before importing pinecone."""
        import importlib

        ip = importlib.import_module("ingest_prepared")
        manifest_path = self._make_manifest_subprocess(tmp_path)

        # Poison the pinecone import so any attempt to import it fails loudly.
        import builtins

        real_import = builtins.__import__

        def guarded_import(name, *a, **k):
            if name == "pinecone" or name.startswith("pinecone."):
                raise AssertionError("pinecone must NOT be imported on the blocked --execute path")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", guarded_import)
        monkeypatch.setenv("CONFIRM_PINECONE_WRITE", "1")
        monkeypatch.setenv("PINECONE_API_KEY", "pcsk_fake")
        monkeypatch.setattr(
            sys, "argv", ["ingest_prepared.py", "--manifest", str(manifest_path), "--execute"]
        )
        with pytest.raises(SystemExit) as se:
            ip.main()
        assert se.value.code == 2

    def test_dry_run_still_works(self, tmp_path):
        manifest_path = self._make_manifest_subprocess(tmp_path)
        result = subprocess.run(
            [
                sys.executable,
                str(Path(_SCRIPTS) / "ingest_prepared.py"),
                "--manifest",
                str(manifest_path),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
        )
        assert result.returncode == 0
        assert "DRY RUN" in result.stdout.upper()


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 4 — Pagination hardening
# ═══════════════════════════════════════════════════════════════════════════════


class TestListVectorIdsPagination:
    def _store(self, index):
        return PineconeStore(index, embed_model="multilingual-e5-large")

    def test_repeated_token_fails_closed(self):
        index = MagicMock()
        # Both pages return the same non-null next token → loop hazard.
        index.list_paginated.side_effect = [
            _FakeResp(["a"], next_token="tok"),
            _FakeResp(["b"], next_token="tok"),
        ]
        with pytest.raises(PineconeProviderError, match="repeated pagination token"):
            self._store(index).list_vector_ids(namespace="pilot_v1")

    def test_empty_progress_page_fails_closed(self):
        index = MagicMock()
        index.list_paginated.side_effect = [_FakeResp([], next_token="next")]
        with pytest.raises(PineconeProviderError, match="no new IDs"):
            self._store(index).list_vector_ids(namespace="pilot_v1")

    def test_duplicate_ids_fail_closed(self):
        index = MagicMock()
        index.list_paginated.side_effect = [
            _FakeResp(["a", "b"], next_token="t2"),
            _FakeResp(["b", "c"], next_token=None),
        ]
        with pytest.raises(PineconeProviderError, match="Duplicate vector ID"):
            self._store(index).list_vector_ids(namespace="pilot_v1")

    def test_malformed_id_fails_closed(self):
        index = MagicMock()
        # A response with a malformed (non-string) id.
        resp = MagicMock()
        resp.vectors = [MagicMock(id=123)]
        resp.pagination = None
        index.list_paginated.return_value = resp
        with pytest.raises(PineconeProviderError, match="Malformed vector item"):
            self._store(index).list_vector_ids(namespace="pilot_v1")

    def test_empty_string_id_fails_closed(self):
        index = MagicMock()
        resp = MagicMock()
        resp.vectors = [MagicMock(id="")]
        resp.pagination = None
        index.list_paginated.return_value = resp
        with pytest.raises(PineconeProviderError, match="Malformed vector item"):
            self._store(index).list_vector_ids(namespace="pilot_v1")

    def test_multiple_valid_pages_ok(self):
        index = MagicMock()
        index.list_paginated.side_effect = [
            _FakeResp(["a", "b"], next_token="t2"),
            _FakeResp(["c", "d"], next_token="t3"),
            _FakeResp(["e"], next_token=None),
        ]
        ids = self._store(index).list_vector_ids(namespace="pilot_v1")
        assert ids == ["a", "b", "c", "d", "e"]

    def test_provider_failure_wrapped(self):
        index = MagicMock()
        index.list_paginated.side_effect = RuntimeError("network down")
        with pytest.raises(PineconeProviderError, match="list_paginated failed"):
            self._store(index).list_vector_ids(namespace="pilot_v1")


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 5 — count_namespace fail-closed
# ═══════════════════════════════════════════════════════════════════════════════


class TestCountNamespaceFailClosed:
    def _store(self, index):
        return PineconeStore(index, embed_model="multilingual-e5-large")

    def test_valid_zero(self):
        index = MagicMock()
        stats = MagicMock()
        ns_info = MagicMock()
        ns_info.vector_count = 0
        stats.namespaces = {"pilot_v1": ns_info}
        index.describe_index_stats.return_value = stats
        assert self._store(index).count_namespace("pilot_v1") == 0

    def test_absent_namespace_returns_zero(self):
        index = MagicMock()
        stats = MagicMock()
        stats.namespaces = {"other": MagicMock(vector_count=5)}
        index.describe_index_stats.return_value = stats
        assert self._store(index).count_namespace("pilot_v1") == 0

    def test_malformed_response_raises(self):
        index = MagicMock()
        stats = MagicMock()
        stats.namespaces = None
        index.describe_index_stats.return_value = stats
        with pytest.raises(PineconeProviderError, match="malformed"):
            self._store(index).count_namespace("pilot_v1")

    def test_provider_failure_raises_not_zero(self):
        index = MagicMock()
        index.describe_index_stats.side_effect = RuntimeError("503 unavailable")
        with pytest.raises(PineconeProviderError, match="provider error"):
            self._store(index).count_namespace("pilot_v1")

    def test_negative_count_raises(self):
        index = MagicMock()
        stats = MagicMock()
        ns_info = MagicMock()
        ns_info.vector_count = -1
        stats.namespaces = {"pilot_v1": ns_info}
        index.describe_index_stats.return_value = stats
        with pytest.raises(PineconeProviderError, match="negative"):
            self._store(index).count_namespace("pilot_v1")

    def test_bool_count_raises(self):
        index = MagicMock()
        stats = MagicMock()
        ns_info = MagicMock()
        ns_info.vector_count = True
        stats.namespaces = {"pilot_v1": ns_info}
        index.describe_index_stats.return_value = stats
        with pytest.raises(PineconeProviderError, match="non-integer vector_count"):
            self._store(index).count_namespace("pilot_v1")


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 6 — Capacity tokenizer-load failure fails closed
# ═══════════════════════════════════════════════════════════════════════════════


class TestCapacityTokenizerFailClosed:
    def test_tokenizer_load_failure_exits_nonzero(self, monkeypatch):
        import importlib

        mcc = importlib.import_module("measure_corpus_capacity")

        monkeypatch.setattr(sys, "argv", ["measure_corpus_capacity.py", "--sample-rows", "1"])

        def _boom(revision):
            raise RuntimeError("tokenizer download failed")

        monkeypatch.setattr("hhgoa_rag.ingestion.tokenizer.get_tokenizer", _boom)

        with pytest.raises(SystemExit) as se:
            mcc.main()
        assert se.value.code == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 3 — prepare_canary manifest determinism (unit-level)
# ═══════════════════════════════════════════════════════════════════════════════


class TestManifestDeterminism:
    def test_manifest_omits_runtime_fields(self, tmp_path):
        """A freshly written canonical manifest must not contain runtime fields."""
        manifest_path = _write_manifest(tmp_path, _make_records(300))
        manifest = json.loads(manifest_path.read_text())
        assert "created_at" not in manifest
        assert "prepared_record_path_full" not in manifest

    def test_prepare_canary_source_has_no_runtime_fields_in_manifest(self):
        """The prepare_canary source must not add created_at to the canonical manifest."""
        src = (Path(_SCRIPTS) / "prepare_canary.py").read_text()
        # created_at must no longer feed the canonical checksummed manifest.
        assert '"created_at": created_at' not in src
        # The canonical manifest dict block must not declare the full path key.
        manifest_block = src.split("manifest: dict = {", 1)[1].split("manifest_for_checksum", 1)[0]
        assert '"prepared_record_path_full"' not in manifest_block
        assert '"created_at"' not in manifest_block
        # Runtime metadata must be written to a separate sidecar file.
        assert "_runtime.json" in src
        assert "runtime_meta = {" in src
