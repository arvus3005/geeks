"""Comprehensive adversarial tests for pre-index reliability pass.

Covers:
1. Namespace preflight fail-closed adversarial scenarios
2. Exact resume ownership verification with deterministic vector IDs
3. Capacity measurement with synthetic data
4. Fresh-clone simulation without pre-existing artifacts
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_SCRIPTS = str(Path(__file__).parent.parent.parent / "scripts")
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
from hhgoa_rag.pinecone_store import PineconeStore  # noqa: E402

CANONICAL_INDEX_NAME = pc_contract.INDEX_NAME
CANONICAL_NAMESPACE = pc_contract.NAMESPACE


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
    tmp_path: Path, records: list[dict] | None = None
) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    if records is None:
        records = _make_records(300)

    jsonl_path = tmp_path / "canary-42-test_records.jsonl"
    jsonl_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    jsonl_checksum = hashlib.sha256(jsonl_path.read_bytes()).hexdigest()

    token_total = sum(r.get("token_length", 0) for r in records)
    lang_counts = {"en": 100, "hi": 100, "bn": 100}
    fp = contract_fingerprint()
    contract = canonical_contract()

    manifest: dict = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "manifest_id": "canary-42-test",
        "mode": "canary",
        "contract_version": CONTRACT_VERSION,
        "contract_fingerprint": fp,
        "index_contract": contract,
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
        "actual_per_language_records": lang_counts,
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
    return manifest_path, jsonl_path


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Adversarial Namespace Preflight Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestNamespacePreflightFailClosedAdversarial:
    @pytest.mark.parametrize(
        "stats,expected_result",
        [
            (None, None),
            ({}, None),
            ({"total_vector_count": 0}, None),  # missing namespaces key
            ({"total_vector_count": 100}, None),  # missing namespaces key
            ({"namespaces": None}, None),
            ({"namespaces": []}, None),  # non-dict
            ({"namespaces": {"pilot_v1": {}}}, None),  # missing vector_count
            ({"namespaces": {"pilot_v1": {"vector_count": None}}}, None),
            ({"namespaces": {"pilot_v1": {"vector_count": "invalid"}}}, None),
            ({"namespaces": {"pilot_v1": {"vector_count": -1}}}, None),
            ({"namespaces": {"pilot_v1": {"vector_count": True}}}, None),  # bool not allowed
            ({"namespaces": {}}, 0),  # valid empty stats map
            ({"namespaces": {"other_namespace": {"vector_count": 50}}}, 0),
            ({"namespaces": {"pilot_v1": {"vector_count": 0}}}, 0),
            ({"namespaces": {"pilot_v1": {"vector_count": 300}}}, 300),
        ],
    )
    def test_get_ns_vector_count_strict_parsing(self, stats, expected_result):
        res = ic._get_ns_vector_count(stats, CANONICAL_NAMESPACE)
        assert res == expected_result

    def test_preflight_missing_namespaces_aborts_with_zero_upserts(self, monkeypatch, tmp_path):
        manifest_path, _ = _write_jsonl_and_manifest(tmp_path)
        report_dir = tmp_path / "reports"

        mock_pc = MagicMock()
        mock_index = MagicMock()
        mock_pc.Index.return_value = mock_index
        # Return stats without namespaces field (only total_vector_count)
        mock_index.describe_index_stats.return_value = {"total_vector_count": 0}

        monkeypatch.setenv("CONFIRM_PINECONE_WRITE", "1")
        monkeypatch.setenv("PINECONE_API_KEY", "pcsk_fake_key_for_testing")
        monkeypatch.setattr("pinecone.Pinecone", lambda api_key: mock_pc)
        monkeypatch.setattr("hhgoa_rag.pinecone_lifecycle.validate_index", lambda pc, name: [])

        parser = ic._build_parser()
        args = parser.parse_args(
            ["--manifest", str(manifest_path), "--execute", "--report-dir", str(report_dir)]
        )

        report_data = {}
        with pytest.raises(ic.CanaryError) as exc_info:
            ic._run(
                args,
                live_mode=True,
                run_id="test-run",
                start_time="2026-08-16T00:00:00Z",
                git_commit="abc",
                report_data=report_data,
            )

        assert exc_info.value.category == "PreflightUnverifiable"
        mock_index.upsert_records.assert_not_called()

    def test_preflight_malformed_vector_count_aborts_with_zero_upserts(self, monkeypatch, tmp_path):
        manifest_path, _ = _write_jsonl_and_manifest(tmp_path)
        report_dir = tmp_path / "reports"

        mock_pc = MagicMock()
        mock_index = MagicMock()
        mock_pc.Index.return_value = mock_index
        mock_index.describe_index_stats.return_value = {
            "namespaces": {"pilot_v1": {"vector_count": "MALFORMED"}}
        }

        monkeypatch.setenv("CONFIRM_PINECONE_WRITE", "1")
        monkeypatch.setenv("PINECONE_API_KEY", "pcsk_fake_key_for_testing")
        monkeypatch.setattr("pinecone.Pinecone", lambda api_key: mock_pc)
        monkeypatch.setattr("hhgoa_rag.pinecone_lifecycle.validate_index", lambda pc, name: [])

        parser = ic._build_parser()
        args = parser.parse_args(
            ["--manifest", str(manifest_path), "--execute", "--report-dir", str(report_dir)]
        )

        report_data = {}
        with pytest.raises(ic.CanaryError) as exc_info:
            ic._run(
                args,
                live_mode=True,
                run_id="test-run",
                start_time="2026-08-16T00:00:00Z",
                git_commit="abc",
                report_data=report_data,
            )

        assert exc_info.value.category == "PreflightUnverifiable"
        mock_index.upsert_records.assert_not_called()

    def test_preflight_empty_response_aborts_with_zero_upserts(self, monkeypatch, tmp_path):
        manifest_path, _ = _write_jsonl_and_manifest(tmp_path)
        report_dir = tmp_path / "reports"

        mock_pc = MagicMock()
        mock_index = MagicMock()
        mock_pc.Index.return_value = mock_index
        mock_index.describe_index_stats.return_value = None

        monkeypatch.setenv("CONFIRM_PINECONE_WRITE", "1")
        monkeypatch.setenv("PINECONE_API_KEY", "pcsk_fake_key_for_testing")
        monkeypatch.setattr("pinecone.Pinecone", lambda api_key: mock_pc)
        monkeypatch.setattr("hhgoa_rag.pinecone_lifecycle.validate_index", lambda pc, name: [])

        parser = ic._build_parser()
        args = parser.parse_args(
            ["--manifest", str(manifest_path), "--execute", "--report-dir", str(report_dir)]
        )

        report_data = {}
        with pytest.raises(ic.CanaryError) as exc_info:
            ic._run(
                args,
                live_mode=True,
                run_id="test-run",
                start_time="2026-08-16T00:00:00Z",
                git_commit="abc",
                report_data=report_data,
            )

        assert exc_info.value.category == "PreflightUnverifiable"
        mock_index.upsert_records.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Exact Resume Ownership Verification Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestResumeOwnershipVerification:
    def _setup_resume_test(self, tmp_path, monkeypatch):
        manifest_path, jsonl_path = _write_jsonl_and_manifest(tmp_path)
        report_dir = tmp_path / "reports"
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)

        manifest = json.loads(manifest_path.read_text())
        records = ic._verify_and_load_records(manifest, manifest_path)
        batches = ic._build_batches(records, 96)
        batch_digests = [
            ic._batch_digest(manifest["manifest_checksum"], i, [r["id"] for r in b])
            for i, b in enumerate(batches)
        ]
        completed_digests = [batch_digests[0]]  # First batch completed (96 records)
        expected_completed_ids = [r["id"] for r in batches[0]]

        ckpt_data = {
            "checkpoint_schema_version": ic.CHECKPOINT_SCHEMA_VERSION,
            "run_id": "test-resume-run",
            "manifest_id": manifest["manifest_id"],
            "manifest_checksum": manifest["manifest_checksum"],
            "contract_fingerprint": manifest["contract_fingerprint"],
            "index_name": CANONICAL_INDEX_NAME,
            "namespace": CANONICAL_NAMESPACE,
            "batch_size": 96,
            "batch_digests": batch_digests,
            "completed_batch_digests": completed_digests,
            "total_batches": len(batches),
            "attempts": 1,
            "retries": 0,
            "started_at": "2026-08-16T00:00:00Z",
            "updated_at": "2026-08-16T00:00:00Z",
        }
        ckpt_file = ckpt_dir / f"canary_{manifest['manifest_id']}_prev.json"
        ckpt_file.write_text(json.dumps(ckpt_data))

        mock_pc = MagicMock()
        mock_index = MagicMock()
        mock_pc.Index.return_value = mock_index
        mock_index.describe_index_stats.side_effect = [
            {"namespaces": {"pilot_v1": {"vector_count": 96}}},  # preflight stats
            {"namespaces": {"pilot_v1": {"vector_count": 300}}},  # step 7 reconciliation stats
        ]
        mock_index.upsert_records.return_value = 96

        monkeypatch.setenv("CONFIRM_PINECONE_WRITE", "1")
        monkeypatch.setenv("PINECONE_API_KEY", "pcsk_fake_key_for_testing")
        monkeypatch.setattr("pinecone.Pinecone", lambda api_key: mock_pc)
        monkeypatch.setattr("hhgoa_rag.pinecone_lifecycle.validate_index", lambda pc, name: [])

        parser = ic._build_parser()
        args = parser.parse_args(
            [
                "--manifest",
                str(manifest_path),
                "--execute",
                "--resume",
                "--checkpoint-dir",
                str(ckpt_dir),
                "--report-dir",
                str(report_dir),
            ]
        )

        return args, mock_index, expected_completed_ids

    def test_exact_expected_id_set_passes_resume(self, monkeypatch, tmp_path):
        args, mock_index, expected_ids = self._setup_resume_test(tmp_path, monkeypatch)

        # Mock list_paginated to return exactly the expected IDs
        class FakeListItem:
            def __init__(self, item_id: str) -> None:
                self.id = item_id

        class FakeListResponse:
            def __init__(self, ids: list[str]) -> None:
                self.vectors = [FakeListItem(i) for i in ids]
                self.pagination = None

        mock_index.list_paginated.return_value = FakeListResponse(expected_ids)

        report_data = {}
        ic._run(
            args,
            live_mode=True,
            run_id="test-run",
            start_time="2026-08-16T00:00:00Z",
            git_commit="abc",
            report_data=report_data,
        )
        assert report_data["status"] == "success"
        assert report_data["skipped_batches"] == 1
        assert report_data["completed_batches"] == 3

    def test_same_count_wrong_ids_fails_closed(self, monkeypatch, tmp_path):
        args, mock_index, expected_ids = self._setup_resume_test(tmp_path, monkeypatch)

        # Namespace has 96 vectors, but totally wrong IDs
        wrong_ids = [f"unrelated-id-{i}" for i in range(len(expected_ids))]

        class FakeListItem:
            def __init__(self, item_id: str) -> None:
                self.id = item_id

        class FakeListResponse:
            def __init__(self, ids: list[str]) -> None:
                self.vectors = [FakeListItem(i) for i in ids]
                self.pagination = None

        mock_index.list_paginated.return_value = FakeListResponse(wrong_ids)

        report_data = {}
        with pytest.raises(ic.CanaryError) as exc_info:
            ic._run(
                args,
                live_mode=True,
                run_id="test-run",
                start_time="2026-08-16T00:00:00Z",
                git_commit="abc",
                report_data=report_data,
            )

        assert exc_info.value.category == "ResumeOwnershipMismatch"
        mock_index.upsert_records.assert_not_called()

    def test_correct_ids_plus_one_unrelated_id_fails_closed(self, monkeypatch, tmp_path):
        args, mock_index, expected_ids = self._setup_resume_test(tmp_path, monkeypatch)

        # Expected 96 IDs plus 1 extra unrelated ID
        extra_ids = expected_ids + ["rogue-vector-999"]

        class FakeListItem:
            def __init__(self, item_id: str) -> None:
                self.id = item_id

        class FakeListResponse:
            def __init__(self, ids: list[str]) -> None:
                self.vectors = [FakeListItem(i) for i in ids]
                self.pagination = None

        mock_index.list_paginated.return_value = FakeListResponse(extra_ids)

        report_data = {}
        with pytest.raises(ic.CanaryError) as exc_info:
            ic._run(
                args,
                live_mode=True,
                run_id="test-run",
                start_time="2026-08-16T00:00:00Z",
                git_commit="abc",
                report_data=report_data,
            )

        assert exc_info.value.category == "ResumeOwnershipMismatch"
        mock_index.upsert_records.assert_not_called()

    def test_missing_expected_id_fails_closed(self, monkeypatch, tmp_path):
        args, mock_index, expected_ids = self._setup_resume_test(tmp_path, monkeypatch)

        # 95 out of 96 IDs present
        missing_ids = expected_ids[:-1]

        class FakeListItem:
            def __init__(self, item_id: str) -> None:
                self.id = item_id

        class FakeListResponse:
            def __init__(self, ids: list[str]) -> None:
                self.vectors = [FakeListItem(i) for i in ids]
                self.pagination = None

        mock_index.list_paginated.return_value = FakeListResponse(missing_ids)

        report_data = {}
        with pytest.raises(ic.CanaryError) as exc_info:
            ic._run(
                args,
                live_mode=True,
                run_id="test-run",
                start_time="2026-08-16T00:00:00Z",
                git_commit="abc",
                report_data=report_data,
            )

        assert exc_info.value.category == "ResumeOwnershipMismatch"
        mock_index.upsert_records.assert_not_called()

    def test_pagination_multiple_pages_handled_correctly(self):
        mock_index = MagicMock()

        class FakeListItem:
            def __init__(self, item_id: str) -> None:
                self.id = item_id

        class FakePagination:
            def __init__(self, next_token: str | None) -> None:
                self.next = next_token

        class FakePage:
            def __init__(self, ids: list[str], next_tok: str | None) -> None:
                self.vectors = [FakeListItem(i) for i in ids]
                self.pagination = FakePagination(next_tok)

        page1 = FakePage(["id-1", "id-2"], next_tok="page2_token")
        page2 = FakePage(["id-3", "id-4"], next_tok=None)

        mock_index.list_paginated.side_effect = [page1, page2]

        store = PineconeStore(mock_index, embed_model="multilingual-e5-large")
        result_ids = store.list_vector_ids(namespace="pilot_v1")

        assert result_ids == ["id-1", "id-2", "id-3", "id-4"]
        assert mock_index.list_paginated.call_count == 2

    def test_id_enumeration_provider_exception_fails_closed(self, monkeypatch, tmp_path):
        args, mock_index, _ = self._setup_resume_test(tmp_path, monkeypatch)

        mock_index.list_paginated.side_effect = RuntimeError("500 Internal Error during list")

        report_data = {}
        with pytest.raises(ic.CanaryError) as exc_info:
            ic._run(
                args,
                live_mode=True,
                run_id="test-run",
                start_time="2026-08-16T00:00:00Z",
                git_commit="abc",
                report_data=report_data,
            )

        assert exc_info.value.category == "ResumeOwnershipUnverifiable"
        mock_index.upsert_records.assert_not_called()

    def test_unsupported_id_enumeration_fails_closed(self, monkeypatch, tmp_path):
        args, mock_index, _ = self._setup_resume_test(tmp_path, monkeypatch)

        del mock_index.list_paginated
        del mock_index.list

        report_data = {}
        with pytest.raises(ic.CanaryError) as exc_info:
            ic._run(
                args,
                live_mode=True,
                run_id="test-run",
                start_time="2026-08-16T00:00:00Z",
                git_commit="abc",
                report_data=report_data,
            )

        assert exc_info.value.category == "ResumeOwnershipUnverifiable"
        mock_index.upsert_records.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Capacity Measurement Tool Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestCorpusCapacityMeasurement:
    def test_measure_sample_data_synthetic_rows(self):
        from measure_corpus_capacity import measure_sample_data

        # Synthetic rows where passages and counts are known in advance
        synthetic_rows_hi = [
            {
                "passages": {
                    "English_passages": [
                        "Common English passage 1.",
                        "Unique English passage HI 1.",
                    ],
                    "Translated_passages": ["Hindi translation 1.", "Hindi translation 2."],
                }
            },
            {
                "passages": {
                    "English_passages": [
                        "Common English passage 1.",
                        "Unique English passage HI 2.",
                    ],
                    "Translated_passages": ["Hindi translation 3.", "Hindi translation 4."],
                }
            },
        ]
        synthetic_rows_bn = [
            {
                "passages": {
                    "English_passages": [
                        "Common English passage 1.",
                        "Unique English passage BN 1.",
                    ],
                    "Translated_passages": ["Bengali translation 1.", "Bengali translation 2."],
                }
            }
        ]

        report = measure_sample_data(
            rows_by_config={"hi": synthetic_rows_hi, "bn": synthetic_rows_bn},
            chunk_strategy="passage_native",
            dimension=1024,
        )

        m = report["measured_sample_metrics"]
        assert m["sample_rows_inspected"] == 3
        assert m["raw_english_passages"] == 6
        # "Common English passage 1." is repeated 3 times across rows -> 4 unique English passages total
        assert m["unique_english_passages"] == 4
        assert m["english_cross_config_duplicates"] == 2
        assert m["raw_translated_passages"] == 6

        p = report["pinecone_integrated_vector_model"]
        assert p["dense_vector_bytes_per_record"] == 1024 * 4
        assert "canary_300_records_storage_mb" in p
        assert "extrapolated_target_3_languages_storage_gb" in p


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Fresh-Clone Canary Preparation Simulation Test
# ═══════════════════════════════════════════════════════════════════════════════


class TestFreshCloneCanarySimulation:
    def test_fresh_clone_dry_run_from_scratch(self, tmp_path):
        """Simulate fresh clone where artifacts/prepared does not exist."""
        fresh_dir = tmp_path / "fresh_clone"
        fresh_dir.mkdir()
        prep_dir = fresh_dir / "artifacts" / "prepared"
        report_dir = fresh_dir / "artifacts" / "reports"

        assert not prep_dir.exists()

        # Step 1: Generate records directly in isolated dir
        manifest_path, jsonl_path = _write_jsonl_and_manifest(prep_dir)
        assert manifest_path.exists()
        assert jsonl_path.exists()

        # Step 2: Validate manifest and execute dry-run
        manifest = ic._load_manifest(manifest_path)
        records = ic._verify_and_load_records(manifest, manifest_path)
        assert len(records) == 300

        # Step 3: Run canary dry-run
        ic._run_canary = MagicMock()  # check dry run logic
        parser = ic._build_parser()
        args = parser.parse_args(
            ["--manifest", str(manifest_path), "--report-dir", str(report_dir)]
        )
        report_data = {}
        ic._run(
            args,
            live_mode=False,
            run_id="fresh-clone-test",
            start_time="2026-08-16T00:00:00Z",
            git_commit="abc",
            report_data=report_data,
        )
        assert report_data["status"] == "dry_run_complete"
        assert report_data["total_batches"] == 4
        assert report_data["pending_batches"] == 4
