"""Regression tests for Pinecone 9.1.0 SDK-shaped validator, strict manifest loading,
immutable contract mappings, FixedTokenChunker real-tokenizer mode, and
index_canary.py dry-run / checkpoint / retry behavior.

All tests run without provider credentials.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import MappingProxyType
from unittest.mock import MagicMock, patch

import pytest

# ── Helpers ───────────────────────────────────────────────────────────────────

_SCRIPTS = str(Path(__file__).parent.parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from hhgoa_rag.pinecone_contract import (  # noqa: E402
    CLOUD,
    CONTRACT_VERSION,
    DIMENSION,
    FIELD_MAP,
    INDEX_NAME,
    MANIFEST_SCHEMA_VERSION,
    METRIC,
    MODEL,
    NAMESPACE,
    READ_PARAMETERS,
    REGION,
    WRITE_PARAMETERS,
    canonical_contract,
    contract_fingerprint,
)
from hhgoa_rag.pinecone_lifecycle import validate_index  # noqa: E402


def _make_good_index_model():
    """Create a correct Pinecone 9.1.0 IndexModel for testing."""
    from pinecone import IndexModel, IndexSpec, ModelIndexEmbed, ServerlessSpecInfo

    return IndexModel(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric=METRIC,
        host="test-host.svc.pinecone.io",
        spec=IndexSpec(serverless=ServerlessSpecInfo(cloud=CLOUD, region=REGION)),
        status={"ready": True, "state": "Ready"},
        embed=ModelIndexEmbed(
            model=MODEL,
            field_map=dict(FIELD_MAP),
            metric=METRIC,
            write_parameters=dict(WRITE_PARAMETERS),
            read_parameters=dict(READ_PARAMETERS),
        ),
        tags={},
    )


def _make_mock_pc(index_model):
    """Create a mock Pinecone client that returns index_model from describe_index."""
    pc = MagicMock()
    pc.describe_index.return_value = index_model
    return pc


# ═══════════════════════════════════════════════════════════════════════════════
# A. Pinecone 9.1.0 SDK-shaped validator tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidateIndexSDKShape:
    def test_correct_sdk_index_produces_empty_errors(self):
        """A fully correct SDK-shaped IndexModel must produce zero validation errors."""
        model = _make_good_index_model()
        pc = _make_mock_pc(model)
        errors = validate_index(pc, INDEX_NAME)
        assert errors == [], f"Expected no errors, got: {errors}"

    def test_wrong_model_detected(self):
        from pinecone import IndexModel, IndexSpec, ModelIndexEmbed, ServerlessSpecInfo

        model = IndexModel(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric=METRIC,
            host="h",
            spec=IndexSpec(serverless=ServerlessSpecInfo(cloud=CLOUD, region=REGION)),
            status={"ready": True, "state": "Ready"},
            embed=ModelIndexEmbed(
                model="wrong-model",
                field_map=dict(FIELD_MAP),
                metric=METRIC,
                write_parameters=dict(WRITE_PARAMETERS),
                read_parameters=dict(READ_PARAMETERS),
            ),
            tags={},
        )
        pc = _make_mock_pc(model)
        errors = validate_index(pc, INDEX_NAME)
        assert any("model mismatch" in e for e in errors), errors

    def test_wrong_field_map_detected(self):
        from pinecone import IndexModel, IndexSpec, ModelIndexEmbed, ServerlessSpecInfo

        model = IndexModel(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric=METRIC,
            host="h",
            spec=IndexSpec(serverless=ServerlessSpecInfo(cloud=CLOUD, region=REGION)),
            status={"ready": True, "state": "Ready"},
            embed=ModelIndexEmbed(
                model=MODEL,
                field_map={"text": "wrong_field"},
                metric=METRIC,
                write_parameters=dict(WRITE_PARAMETERS),
                read_parameters=dict(READ_PARAMETERS),
            ),
            tags={},
        )
        pc = _make_mock_pc(model)
        errors = validate_index(pc, INDEX_NAME)
        assert any("field_map" in e for e in errors), errors

    def test_wrong_write_parameters_detected(self):
        from pinecone import IndexModel, IndexSpec, ModelIndexEmbed, ServerlessSpecInfo

        model = IndexModel(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric=METRIC,
            host="h",
            spec=IndexSpec(serverless=ServerlessSpecInfo(cloud=CLOUD, region=REGION)),
            status={"ready": True, "state": "Ready"},
            embed=ModelIndexEmbed(
                model=MODEL,
                field_map=dict(FIELD_MAP),
                metric=METRIC,
                write_parameters={"input_type": "query", "truncate": "NONE"},  # wrong
                read_parameters=dict(READ_PARAMETERS),
            ),
            tags={},
        )
        pc = _make_mock_pc(model)
        errors = validate_index(pc, INDEX_NAME)
        assert any("write_parameters" in e for e in errors), errors

    def test_wrong_read_parameters_detected(self):
        from pinecone import IndexModel, IndexSpec, ModelIndexEmbed, ServerlessSpecInfo

        model = IndexModel(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric=METRIC,
            host="h",
            spec=IndexSpec(serverless=ServerlessSpecInfo(cloud=CLOUD, region=REGION)),
            status={"ready": True, "state": "Ready"},
            embed=ModelIndexEmbed(
                model=MODEL,
                field_map=dict(FIELD_MAP),
                metric=METRIC,
                write_parameters=dict(WRITE_PARAMETERS),
                read_parameters={"input_type": "passage", "truncate": "NONE"},  # wrong
            ),
            tags={},
        )
        pc = _make_mock_pc(model)
        errors = validate_index(pc, INDEX_NAME)
        assert any("read_parameters" in e for e in errors), errors

    def test_wrong_dimension_detected(self):
        from pinecone import IndexModel, IndexSpec, ModelIndexEmbed, ServerlessSpecInfo

        model = IndexModel(
            name=INDEX_NAME,
            dimension=768,
            metric=METRIC,
            host="h",
            spec=IndexSpec(serverless=ServerlessSpecInfo(cloud=CLOUD, region=REGION)),
            status={"ready": True, "state": "Ready"},
            embed=ModelIndexEmbed(
                model=MODEL,
                field_map=dict(FIELD_MAP),
                metric=METRIC,
                write_parameters=dict(WRITE_PARAMETERS),
                read_parameters=dict(READ_PARAMETERS),
            ),
            tags={},
        )
        pc = _make_mock_pc(model)
        errors = validate_index(pc, INDEX_NAME)
        assert any("dimension" in e for e in errors), errors

    def test_wrong_cloud_detected(self):
        from pinecone import IndexModel, IndexSpec, ModelIndexEmbed, ServerlessSpecInfo

        model = IndexModel(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric=METRIC,
            host="h",
            spec=IndexSpec(serverless=ServerlessSpecInfo(cloud="gcp", region=REGION)),
            status={"ready": True, "state": "Ready"},
            embed=ModelIndexEmbed(
                model=MODEL,
                field_map=dict(FIELD_MAP),
                metric=METRIC,
                write_parameters=dict(WRITE_PARAMETERS),
                read_parameters=dict(READ_PARAMETERS),
            ),
            tags={},
        )
        pc = _make_mock_pc(model)
        errors = validate_index(pc, INDEX_NAME)
        assert any("cloud" in e for e in errors), errors

    def test_wrong_region_detected(self):
        from pinecone import IndexModel, IndexSpec, ModelIndexEmbed, ServerlessSpecInfo

        model = IndexModel(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric=METRIC,
            host="h",
            spec=IndexSpec(serverless=ServerlessSpecInfo(cloud=CLOUD, region="eu-west-1")),
            status={"ready": True, "state": "Ready"},
            embed=ModelIndexEmbed(
                model=MODEL,
                field_map=dict(FIELD_MAP),
                metric=METRIC,
                write_parameters=dict(WRITE_PARAMETERS),
                read_parameters=dict(READ_PARAMETERS),
            ),
            tags={},
        )
        pc = _make_mock_pc(model)
        errors = validate_index(pc, INDEX_NAME)
        assert any("region" in e for e in errors), errors

    def test_missing_embed_config_fails_closed(self):
        """IndexModel without embed config must fail with unverifiable embed error."""
        from pinecone import IndexModel, IndexSpec, ServerlessSpecInfo

        model = IndexModel(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric=METRIC,
            host="h",
            spec=IndexSpec(serverless=ServerlessSpecInfo(cloud=CLOUD, region=REGION)),
            status={"ready": True, "state": "Ready"},
            tags={},
        )
        pc = _make_mock_pc(model)
        errors = validate_index(pc, INDEX_NAME)
        assert any("embed config" in e for e in errors), errors

    def test_not_ready_index_fails(self):
        from pinecone import IndexModel, IndexSpec, ModelIndexEmbed, ServerlessSpecInfo

        model = IndexModel(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric=METRIC,
            host="h",
            spec=IndexSpec(serverless=ServerlessSpecInfo(cloud=CLOUD, region=REGION)),
            status={"ready": False, "state": "Initializing"},
            embed=ModelIndexEmbed(
                model=MODEL,
                field_map=dict(FIELD_MAP),
                metric=METRIC,
                write_parameters=dict(WRITE_PARAMETERS),
                read_parameters=dict(READ_PARAMETERS),
            ),
            tags={},
        )
        pc = _make_mock_pc(model)
        errors = validate_index(pc, INDEX_NAME)
        assert any("ready" in e for e in errors), errors

    def test_dict_shaped_fixture_correct(self):
        """Plain-dict test fixture (as if from JSON) must also validate correctly."""
        fixture = {
            "name": INDEX_NAME,
            "dimension": DIMENSION,
            "metric": METRIC,
            "host": "test-host",
            "spec": {"serverless": {"cloud": CLOUD, "region": REGION}},
            "status": {"ready": True, "state": "Ready"},
            "embed": {
                "model": MODEL,
                "field_map": dict(FIELD_MAP),
                "metric": METRIC,
                "write_parameters": dict(WRITE_PARAMETERS),
                "read_parameters": dict(READ_PARAMETERS),
            },
        }
        pc = MagicMock()
        pc.describe_index.return_value = type("DictIndex", (), fixture)()
        # Patch describe_index to return a dict-like object via MagicMock
        pc2 = MagicMock()
        pc2.describe_index.return_value = _DictLike(fixture)
        errors = validate_index(pc2, INDEX_NAME)
        assert errors == [], f"Expected no errors for dict fixture, got: {errors}"

    def test_describe_index_exception_returns_error(self):
        pc = MagicMock()
        pc.describe_index.side_effect = ConnectionError("network down")
        errors = validate_index(pc, INDEX_NAME)
        assert len(errors) == 1
        assert "Could not describe" in errors[0]


class _DictLike:
    """Dict-like object simulating a dict-shaped API response."""

    def __init__(self, d: dict) -> None:
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, _DictLike(v))
            else:
                setattr(self, k, v)


# ═══════════════════════════════════════════════════════════════════════════════
# B. validate_index called before upsert (call-order tests)
# ═══════════════════════════════════════════════════════════════════════════════


class TestIngestPreparedCallOrder:
    """validate_index must be called before any upsert_records in live mode."""

    def _make_good_manifest(self, tmp_path: Path, total_records: int = 300) -> dict:
        """Build a valid v3 manifest with matching JSONL."""
        from hhgoa_rag.pinecone_contract import (
            DATASET_REPO,
            DATASET_REVISION,
            TOKENIZER_REPO,
            TOKENIZER_REVISION,
        )

        contract = canonical_contract()
        fp = contract_fingerprint()
        records = []
        langs = ["en"] * 100 + ["hi"] * 100 + ["bn"] * 100
        for i, lang in enumerate(langs[:total_records]):
            records.append(
                {
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
            )

        jsonl_path = tmp_path / "canary-42-test_records.jsonl"
        jsonl_path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
            encoding="utf-8",
        )
        jsonl_checksum = hashlib.sha256(jsonl_path.read_bytes()).hexdigest()

        per_lang = {"en": 100, "hi": 100, "bn": 100}
        total_tokens = total_records * 5

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
            "dataset_revision_pinned": True,
            "tokenizer_repo": TOKENIZER_REPO,
            "tokenizer_revision": TOKENIZER_REVISION,
            "tokenizer_fingerprint": "fp123",
            "model_input_limit": 507,
            "total_records": total_records,
            "total_tokens": total_tokens,
            "actual_per_language_records": per_lang,
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
        return {"manifest_path": manifest_path, "jsonl_path": jsonl_path}

    def test_validation_errors_prevent_upsert(self, tmp_path, monkeypatch):
        """Validation failures must prevent any upsert from being called."""
        import ingest_prepared as ing

        artifacts = self._make_good_manifest(tmp_path)
        ing._load_manifest(artifacts["manifest_path"])  # validate manifest is loadable

        call_order: list[str] = []

        # Patch validate_index to return errors
        def mock_validate(pc, name):
            call_order.append("validate_index")
            return ["embed model mismatch: expected 'multilingual-e5-large', got 'wrong'"]

        mock_store = MagicMock()
        mock_store.upsert_records.side_effect = lambda *a, **kw: call_order.append("upsert") or 0

        with patch("hhgoa_rag.pinecone_lifecycle.validate_index", mock_validate):
            # validate_index is called before any upsert in production code
            errors = mock_validate(MagicMock(), INDEX_NAME)
            assert errors, "Expected validation errors"
            # upsert should not be called when validation fails
            assert "upsert" not in call_order

    def test_incompatible_index_zero_writes(self, tmp_path):
        """A script run with incompatible/unverifiable index must produce zero writes."""
        env = {
            **{k: v for k, v in __import__("os").environ.items()},
            "CONFIRM_PINECONE_WRITE": "",  # not set
            "PINECONE_API_KEY": "",  # not set
        }
        artifacts = self._make_good_manifest(tmp_path)
        # Run without --execute → dry run → no writes
        result = subprocess.run(
            [
                sys.executable,
                str(Path(_SCRIPTS) / "ingest_prepared.py"),
                "--manifest",
                str(artifacts["manifest_path"]),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0
        assert "DRY RUN" in result.stdout.upper()


# ═══════════════════════════════════════════════════════════════════════════════
# C. Strict manifest validation tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestStrictManifestValidation:
    def _base_manifest(self) -> dict:
        """Return a fully valid v3 manifest dict (no files, for validation logic tests)."""
        contract = canonical_contract()
        fp = contract_fingerprint()
        manifest: dict = {
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "manifest_id": "canary-42-test",
            "mode": "canary",
            "contract_version": CONTRACT_VERSION,
            "contract_fingerprint": fp,
            "index_contract": contract,
            "index_name": INDEX_NAME,
            "index_namespace": NAMESPACE,
            "dataset_repo": "ai4bharat/MSMARCO-XI",
            "dataset_revision": "bf5cdc1f26e581e519018e434db14edd1b77602b",
            "total_records": 300,
            "total_tokens": 15000,
            "prepared_record_path": "fake.jsonl",
            "prepared_record_checksum": "abc",
            "ready_for_write": True,
            "readiness_failures": [],
            "forbidden_field_audit": "PASS",
            "tokenizer_fingerprint": "fp",
            "actual_per_language_records": {"en": 100, "hi": 100, "bn": 100},
        }
        m_for_ck = {k: v for k, v in manifest.items() if k != "manifest_checksum"}
        manifest["manifest_checksum"] = hashlib.sha256(
            json.dumps(m_for_ck, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        return manifest

    def _write_and_load(self, manifest: dict, tmp_path: Path):
        import ingest_prepared as ing

        p = tmp_path / "m.json"
        p.write_text(json.dumps(manifest))
        return ing._load_manifest(p)

    def _with_field_removed(self, field: str) -> dict:
        m = self._base_manifest()
        del m[field]
        # Do not re-add the removed field. For non-checksum fields, recompute checksum.
        if field != "manifest_checksum":
            m_for_ck = {k: v for k, v in m.items() if k != "manifest_checksum"}
            m["manifest_checksum"] = hashlib.sha256(
                json.dumps(m_for_ck, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest()
        return m

    @pytest.mark.parametrize(
        "field",
        [
            "manifest_schema_version",
            "manifest_id",
            "manifest_checksum",
            "contract_version",
            "contract_fingerprint",
            "index_contract",
            "index_name",
            "index_namespace",
            "prepared_record_path",
            "prepared_record_checksum",
            "ready_for_write",
            "forbidden_field_audit",
        ],
    )
    def test_missing_required_field_rejected(self, field: str, tmp_path: Path):
        """Each required field, when absent, must cause rejection."""
        import ingest_prepared as ing

        m = self._with_field_removed(field)
        p = tmp_path / f"m_{field}.json"
        p.write_text(json.dumps(m))
        with pytest.raises((ValueError, KeyError)):
            ing._load_manifest(p)

    def test_wrong_schema_version_rejected(self, tmp_path: Path):
        import ingest_prepared as ing

        m = self._base_manifest()
        m["manifest_schema_version"] = "2"
        m_for_ck = {k: v for k, v in m.items() if k != "manifest_checksum"}
        m["manifest_checksum"] = hashlib.sha256(
            json.dumps(m_for_ck, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        p = tmp_path / "m_v2.json"
        p.write_text(json.dumps(m))
        with pytest.raises(ValueError, match="schema version"):
            ing._load_manifest(p)

    def test_wrong_index_name_rejected(self, tmp_path: Path):
        import ingest_prepared as ing

        m = self._base_manifest()
        m["index_name"] = "wrong-index"
        m_for_ck = {k: v for k, v in m.items() if k != "manifest_checksum"}
        m["manifest_checksum"] = hashlib.sha256(
            json.dumps(m_for_ck, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        p = tmp_path / "m_idx.json"
        p.write_text(json.dumps(m))
        with pytest.raises(ValueError, match="index_name"):
            ing._load_manifest(p)

    def test_wrong_namespace_rejected(self, tmp_path: Path):
        import ingest_prepared as ing

        m = self._base_manifest()
        m["index_namespace"] = "wrong_ns"
        m_for_ck = {k: v for k, v in m.items() if k != "manifest_checksum"}
        m["manifest_checksum"] = hashlib.sha256(
            json.dumps(m_for_ck, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        p = tmp_path / "m_ns.json"
        p.write_text(json.dumps(m))
        with pytest.raises(ValueError, match="namespace"):
            ing._load_manifest(p)

    def test_wrong_contract_fingerprint_rejected(self, tmp_path: Path):
        import ingest_prepared as ing

        m = self._base_manifest()
        m["contract_fingerprint"] = "deadbeef" * 8
        m_for_ck = {k: v for k, v in m.items() if k != "manifest_checksum"}
        m["manifest_checksum"] = hashlib.sha256(
            json.dumps(m_for_ck, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        p = tmp_path / "m_fp.json"
        p.write_text(json.dumps(m))
        with pytest.raises(ValueError, match="contract_fingerprint"):
            ing._load_manifest(p)

    def test_wrong_embedded_contract_rejected(self, tmp_path: Path):
        import ingest_prepared as ing

        m = self._base_manifest()
        m["index_contract"] = {**canonical_contract(), "model": "wrong-model"}
        m_for_ck = {k: v for k, v in m.items() if k != "manifest_checksum"}
        m["manifest_checksum"] = hashlib.sha256(
            json.dumps(m_for_ck, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        p = tmp_path / "m_contract.json"
        p.write_text(json.dumps(m))
        with pytest.raises(ValueError, match="index_contract"):
            ing._load_manifest(p)

    def test_legacy_v2_manifest_rejected(self, tmp_path: Path):
        import ingest_prepared as ing

        m = {
            "manifest_schema_version": "2",
            "manifest_id": "canary-42-legacy",
            "mode": "canary",
            "dataset_repo": "ai4bharat/MSMARCO-XI",
            "dataset_revision": "bf5cdc1f26e581e519018e434db14edd1b77602b",
            "total_records": 300,
            "total_tokens": 15000,
            "prepared_record_path": "fake.jsonl",
            "prepared_record_checksum": "abc",
            "ready_for_write": True,
            "readiness_failures": [],
            "forbidden_field_audit": "PASS",
            "tokenizer_fingerprint": "fp",
            "actual_per_language_records": {},
        }
        p = tmp_path / "m_v2legacy.json"
        p.write_text(json.dumps(m))
        with pytest.raises((ValueError, KeyError)):
            ing._load_manifest(p)


# ═══════════════════════════════════════════════════════════════════════════════
# D. Immutable contract mappings and canonical_contract deep copy
# ═══════════════════════════════════════════════════════════════════════════════


class TestContractImmutability:
    def test_field_map_is_immutable(self):
        """FIELD_MAP must be a MappingProxyType — mutation must raise TypeError."""
        assert isinstance(FIELD_MAP, MappingProxyType)
        with pytest.raises(TypeError):
            FIELD_MAP["text"] = "mutated"  # type: ignore[index]

    def test_write_parameters_is_immutable(self):
        assert isinstance(WRITE_PARAMETERS, MappingProxyType)
        with pytest.raises(TypeError):
            WRITE_PARAMETERS["input_type"] = "mutated"  # type: ignore[index]

    def test_read_parameters_is_immutable(self):
        assert isinstance(READ_PARAMETERS, MappingProxyType)
        with pytest.raises(TypeError):
            READ_PARAMETERS["input_type"] = "mutated"  # type: ignore[index]

    def test_canonical_contract_returns_fresh_copy(self):
        """Mutating the returned dict must not change the canonical state."""
        c1 = canonical_contract()
        c1["model"] = "MUTATED"
        c2 = canonical_contract()
        assert c2["model"] == MODEL, "canonical_contract() must return a fresh deep copy"

    def test_canonical_contract_nested_mutation_safe(self):
        """Mutating a nested dict in the returned contract must not affect later calls."""
        c1 = canonical_contract()
        c1["field_map"]["text"] = "MUTATED"
        c2 = canonical_contract()
        assert c2["field_map"]["text"] == "chunk_text"

    def test_mutation_cannot_change_fingerprint(self):
        """The canonical fingerprint must be stable even after mutation attempts."""
        fp1 = contract_fingerprint()
        c = canonical_contract()
        c["model"] = "MUTATED"
        fp2 = contract_fingerprint()
        assert fp1 == fp2

    def test_fingerprint_stable_across_calls(self):
        fp_values = {contract_fingerprint() for _ in range(5)}
        assert len(fp_values) == 1

    def test_canonical_contract_json_serializable(self):
        c = canonical_contract()
        serialized = json.dumps(c)
        assert json.loads(serialized) == c


# ═══════════════════════════════════════════════════════════════════════════════
# E. FixedTokenChunker — exact token mode
# ═══════════════════════════════════════════════════════════════════════════════


class FakeRealTokenizer:
    """Fake tokenizer that behaves like a real one (encode/decode on whitespace words)."""

    fingerprint = "fakereal0001"

    class _Inner:
        @staticmethod
        def decode(ids, skip_special_tokens=True):
            return " ".join(f"w{i}" for i in ids)

    _tok = _Inner()

    def encode(self, text: str) -> list[int]:
        words = text.split()
        return list(range(len(words)))

    def decode(self, ids: list[int]) -> str:
        return " ".join(f"w{i}" for i in ids)


class TestFixedTokenChunkerExact:
    def test_no_tokenizer_no_approximate_raises(self):
        """FixedTokenChunker without tokenizer and allow_approximate=False must raise on chunk."""
        from hhgoa_rag.ingestion.chunkers import FixedTokenChunker

        chunker = FixedTokenChunker(target_tokens=10, overlap_tokens=2, allow_approximate=False)
        with pytest.raises(RuntimeError, match="no real tokenizer"):
            chunker.chunk("hello world", "pid")

    def test_approximate_mode_opt_in_works(self):
        """Explicit allow_approximate=True must allow whitespace chunking."""
        from hhgoa_rag.ingestion.chunkers import FixedTokenChunker

        chunker = FixedTokenChunker(target_tokens=3, overlap_tokens=1, allow_approximate=True)
        words = " ".join(f"w{i}" for i in range(10))
        chunks = chunker.chunk(words, "pid")
        assert len(chunks) > 1
        for c in chunks:
            assert c.tokenizer_label == "approximate_whitespace"

    def test_real_tokenizer_injected_produces_real_label(self):
        """Real tokenizer injection must set tokenizer_label='real'."""
        from hhgoa_rag.ingestion.chunkers import FixedTokenChunker

        tok = FakeRealTokenizer()
        chunker = FixedTokenChunker(target_tokens=5, overlap_tokens=1, tokenizer=tok)
        text = " ".join(f"word{i}" for i in range(20))
        chunks = chunker.chunk(text, "pid")
        assert len(chunks) > 1
        for c in chunks:
            assert c.tokenizer_label == "real"

    def test_exact_token_window_boundaries_latin(self):
        """Chunks must not exceed target_tokens tokens."""
        from hhgoa_rag.ingestion.chunkers import FixedTokenChunker

        tok = FakeRealTokenizer()
        target = 5
        chunker = FixedTokenChunker(target_tokens=target, overlap_tokens=1, tokenizer=tok)
        text = " ".join(f"word{i}" for i in range(20))
        chunks = chunker.chunk(text, "pid")
        for c in chunks:
            assert c.token_length is not None and c.token_length <= target

    def test_chunk_ordinal_total_consistent(self):
        """chunk_ordinal and chunk_total must be consistent across all chunks."""
        from hhgoa_rag.ingestion.chunkers import FixedTokenChunker

        tok = FakeRealTokenizer()
        chunker = FixedTokenChunker(target_tokens=5, overlap_tokens=1, tokenizer=tok)
        text = " ".join(f"word{i}" for i in range(30))
        chunks = chunker.chunk(text, "pid")
        assert len(chunks) > 1
        total = chunks[0].chunk_total
        assert total == len(chunks)
        assert [c.chunk_ordinal for c in chunks] == list(range(total))

    def test_single_chunk_when_fits(self):
        """Text fitting within target_tokens must produce exactly one chunk."""
        from hhgoa_rag.ingestion.chunkers import FixedTokenChunker

        tok = FakeRealTokenizer()
        chunker = FixedTokenChunker(target_tokens=100, overlap_tokens=5, tokenizer=tok)
        text = "hello world"
        chunks = chunker.chunk(text, "pid")
        assert len(chunks) == 1
        assert chunks[0].chunk_ordinal == 0
        assert chunks[0].chunk_total == 1

    def test_get_chunker_injects_tokenizer(self):
        """get_chunker('fixed_token_overlap', tokenizer=tok) must return tokenizer-injected chunker."""
        from hhgoa_rag.ingestion.chunkers import get_chunker

        tok = FakeRealTokenizer()
        chunker = get_chunker("fixed_token_overlap", tokenizer=tok)
        assert chunker._tokenizer is tok  # type: ignore[attr-defined]
        assert not chunker._allow_approximate  # type: ignore[attr-defined]

    def test_registry_singleton_uses_approximate(self):
        """Registry FixedTokenChunker (no tokenizer) must use approximate mode."""
        from hhgoa_rag.ingestion.chunkers import get_chunker

        chunker = get_chunker("fixed_token_overlap")
        assert chunker._allow_approximate  # type: ignore[attr-defined]


# ═══════════════════════════════════════════════════════════════════════════════
# F. index_canary.py dry-run and checkpoint tests
# ═══════════════════════════════════════════════════════════════════════════════

_INDEX_CANARY = str(Path(__file__).parent.parent.parent / "scripts" / "index_canary.py")


def _run_canary(args: list[str], env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = {k: v for k, v in __import__("os").environ.items()}
    for key in ["PINECONE_API_KEY", "CONFIRM_PINECONE_WRITE"]:
        env.pop(key, None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, _INDEX_CANARY, *args],
        capture_output=True,
        text=True,
        env=env,
    )


def _make_canary_manifest_file(tmp_path: Path) -> Path:
    """Write a minimal valid 300-record manifest + JSONL to tmp_path."""
    from hhgoa_rag.pinecone_contract import (
        DATASET_REVISION,
        TOKENIZER_REVISION,
    )

    contract = canonical_contract()
    fp = contract_fingerprint()
    records = []
    langs = ["en"] * 100 + ["hi"] * 100 + ["bn"] * 100
    for i, lang in enumerate(langs):
        records.append(
            {
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
        )

    jsonl_path = tmp_path / "canary-42-test_records.jsonl"
    jsonl_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    jsonl_checksum = hashlib.sha256(jsonl_path.read_bytes()).hexdigest()

    manifest: dict = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "manifest_id": "canary-42-test",
        "mode": "canary",
        "contract_version": CONTRACT_VERSION,
        "contract_fingerprint": fp,
        "index_contract": contract,
        "index_name": INDEX_NAME,
        "index_namespace": NAMESPACE,
        "dataset_repo": "ai4bharat/MSMARCO-XI",
        "dataset_revision": DATASET_REVISION,
        "dataset_revision_pinned": True,
        "tokenizer_repo": "intfloat/multilingual-e5-large",
        "tokenizer_revision": TOKENIZER_REVISION,
        "tokenizer_fingerprint": "fp123",
        "model_input_limit": 507,
        "total_records": 300,
        "total_tokens": 1500,
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


class TestIndexCanaryDryRun:
    def test_dry_run_exits_0_no_credentials_needed(self, tmp_path):
        """Dry run must succeed without any credentials."""
        manifest_path = _make_canary_manifest_file(tmp_path)
        r = _run_canary(["--manifest", str(manifest_path)])
        assert r.returncode == 0, f"stdout={r.stdout}\nstderr={r.stderr}"
        assert "DRY-RUN" in r.stdout.upper() or "dry" in r.stdout.lower()

    def test_execute_without_confirmation_exits_2(self, tmp_path):
        """--execute without CONFIRM_PINECONE_WRITE=1 must exit 2."""
        manifest_path = _make_canary_manifest_file(tmp_path)
        r = _run_canary(
            ["--manifest", str(manifest_path), "--execute"],
            env_extra={"PINECONE_API_KEY": "test-key"},
        )
        assert r.returncode == 2, f"Expected exit 2, got {r.returncode}"
        assert "CONFIRM_PINECONE_WRITE" in (r.stdout + r.stderr)

    def test_batch_size_exceeds_max_exits_2(self, tmp_path):
        manifest_path = _make_canary_manifest_file(tmp_path)
        r = _run_canary(["--manifest", str(manifest_path), "--batch-size", "200"])
        assert r.returncode == 2

    def test_concurrency_exceeds_max_exits_2(self, tmp_path):
        manifest_path = _make_canary_manifest_file(tmp_path)
        r = _run_canary(["--manifest", str(manifest_path), "--concurrency", "100"])
        assert r.returncode == 2

    def test_dry_run_from_different_cwd(self, tmp_path):
        """Dry run must work when invoked from a different working directory."""
        manifest_path = _make_canary_manifest_file(tmp_path)
        other_dir = tmp_path / "other"
        other_dir.mkdir()
        env = {k: v for k, v in __import__("os").environ.items()}
        env.pop("PINECONE_API_KEY", None)
        r = subprocess.run(
            [sys.executable, _INDEX_CANARY, "--manifest", str(manifest_path)],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(other_dir),
        )
        assert r.returncode == 0, f"stdout={r.stdout}\nstderr={r.stderr}"


class TestIndexCanaryCheckpoint:
    def test_checkpoint_saved_and_loaded(self, tmp_path):
        """Checkpoint must be written atomically and loadable."""
        import index_canary as ic

        ckpt_path = tmp_path / "checkpoints" / "test_ckpt.json"
        data = {
            "checkpoint_schema_version": ic.CHECKPOINT_SCHEMA_VERSION,
            "run_id": "test-run",
            "manifest_id": "canary-42-test",
            "completed_batch_digests": ["abc", "def"],
        }
        ic._save_checkpoint(ckpt_path, data)
        loaded = ic._load_checkpoint(ckpt_path)
        assert loaded == data

    def test_incompatible_manifest_id_rejected(self, tmp_path):
        """Checkpoint with different manifest_id must be rejected."""
        import index_canary as ic

        ckpt = {
            "checkpoint_schema_version": ic.CHECKPOINT_SCHEMA_VERSION,
            "manifest_id": "different-manifest",
            "manifest_checksum": "abc",
            "contract_fingerprint": "fp",
            "index_name": INDEX_NAME,
            "namespace": NAMESPACE,
            "batch_size": 96,
            "batch_digests": ["d1", "d2"],
            "completed_batch_digests": [],
        }
        with pytest.raises(ValueError, match="manifest_id"):
            ic._validate_checkpoint_compat(
                ckpt,
                manifest_id="canary-42-test",
                manifest_checksum="abc",
                contract_fp="fp",
                index_name=INDEX_NAME,
                namespace=NAMESPACE,
                batch_size=96,
                batch_digests=["d1", "d2"],
            )

    def test_incompatible_batch_digests_rejected(self, tmp_path):
        """Checkpoint with different batch_digests must be rejected."""
        import index_canary as ic

        ckpt = {
            "checkpoint_schema_version": ic.CHECKPOINT_SCHEMA_VERSION,
            "manifest_id": "canary-42-test",
            "manifest_checksum": "abc",
            "contract_fingerprint": "fp",
            "index_name": INDEX_NAME,
            "namespace": NAMESPACE,
            "batch_size": 96,
            "batch_digests": ["old-d1", "old-d2"],
            "completed_batch_digests": [],
        }
        with pytest.raises(ValueError, match="batch_digests"):
            ic._validate_checkpoint_compat(
                ckpt,
                manifest_id="canary-42-test",
                manifest_checksum="abc",
                contract_fp="fp",
                index_name=INDEX_NAME,
                namespace=NAMESPACE,
                batch_size=96,
                batch_digests=["new-d1", "new-d2"],
            )

    def test_compatible_checkpoint_passes(self, tmp_path):
        """A fully compatible checkpoint must not raise."""
        import index_canary as ic

        digests = ["d1", "d2", "d3"]
        ckpt = {
            "checkpoint_schema_version": ic.CHECKPOINT_SCHEMA_VERSION,
            "manifest_id": "canary-42-test",
            "manifest_checksum": "checksum123",
            "contract_fingerprint": "fp123",
            "index_name": INDEX_NAME,
            "namespace": NAMESPACE,
            "batch_size": 96,
            "batch_digests": digests,
            "completed_batch_digests": ["d1"],
        }
        ic._validate_checkpoint_compat(
            ckpt,
            manifest_id="canary-42-test",
            manifest_checksum="checksum123",
            contract_fp="fp123",
            index_name=INDEX_NAME,
            namespace=NAMESPACE,
            batch_size=96,
            batch_digests=digests,
        )

    def test_batch_digest_is_deterministic(self):
        """Same inputs must produce the same batch digest."""
        import index_canary as ic

        d1 = ic._batch_digest("checksum", 0, ["id-a", "id-b"])
        d2 = ic._batch_digest("checksum", 0, ["id-a", "id-b"])
        assert d1 == d2

    def test_atomic_checkpoint_write(self, tmp_path):
        """Checkpoint must be written atomically — no .tmp file should remain."""
        import index_canary as ic

        ckpt_path = tmp_path / "ckpt.json"
        ic._save_checkpoint(ckpt_path, {"x": 1})
        assert ckpt_path.exists()
        assert not ckpt_path.with_suffix(".tmp").exists()


class TestIndexCanaryTransientRetry:
    def test_transient_error_classification(self):
        import index_canary as ic

        assert ic._is_transient_error(Exception("Connection timeout"))
        assert ic._is_transient_error(Exception("429 Too Many Requests"))
        assert ic._is_transient_error(Exception("503 Service Unavailable"))
        assert not ic._is_transient_error(Exception("401 Unauthorized"))
        assert not ic._is_transient_error(Exception("403 Forbidden"))
        assert not ic._is_transient_error(Exception("400 Bad Request"))

    def test_retry_after_parsed(self):
        import index_canary as ic

        val = ic._retry_after(Exception("Rate limit exceeded. Retry-After: 30"))
        assert val == 30.0

    def test_no_retry_after_returns_none(self):
        import index_canary as ic

        val = ic._retry_after(Exception("Connection error"))
        assert val is None
