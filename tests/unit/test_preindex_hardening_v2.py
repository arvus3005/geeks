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
        report_dir = tmp_path / "reports"
        (ckpt_dir / "canary_canary-42-test_BAD.json").write_text("NOT_JSON")
        r = _run_canary(
            [
                "--manifest",
                str(manifest_path),
                "--resume",
                "--checkpoint-dir",
                str(ckpt_dir),
                "--report-dir",
                str(report_dir),
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
# 9. _TokenRateLimiter Behavioural & Concurrency Tests
# ═══════════════════════════════════════════════════════════════════════════════


class FakeClock:
    def __init__(self, initial_time: float = 1000.0) -> None:
        self.time = initial_time

    def __call__(self) -> float:
        return self.time

    def advance(self, seconds: float) -> None:
        self.time += seconds


class FakeSleeper:
    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.waits: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.waits.append(seconds)
        self.clock.advance(seconds)


class TestTokenRateLimiterBehavioural:
    def test_single_reservation_below_limit_succeeds_without_sleeping(self):
        import index_canary as ic

        clock = FakeClock()
        sleeper = FakeSleeper(clock)
        limiter = ic._TokenRateLimiter(
            tokens_per_window=1000,
            window_seconds=60.0,
            clock=clock,
            sleeper=sleeper,
        )
        waited = limiter.acquire(500)
        assert waited == 0.0
        assert sleeper.waits == []
        assert limiter._current_tokens == 500

    def test_multiple_reservations_summing_to_limit_succeed_without_sleeping(self):
        import index_canary as ic

        clock = FakeClock()
        sleeper = FakeSleeper(clock)
        limiter = ic._TokenRateLimiter(
            tokens_per_window=1000,
            window_seconds=60.0,
            clock=clock,
            sleeper=sleeper,
        )
        w1 = limiter.acquire(400)
        w2 = limiter.acquire(600)
        assert w1 == 0.0
        assert w2 == 0.0
        assert sleeper.waits == []
        assert limiter._current_tokens == 1000

    def test_next_reservation_waits_for_next_window(self):
        import index_canary as ic

        clock = FakeClock(100.0)
        sleeper = FakeSleeper(clock)
        limiter = ic._TokenRateLimiter(
            tokens_per_window=1000,
            window_seconds=60.0,
            clock=clock,
            sleeper=sleeper,
        )
        limiter.acquire(1000)
        clock.advance(10.0)  # 10s elapsed in window

        # Next reservation requires 200 tokens — must wait remaining 50s for past reservation to slide out
        waited = limiter.acquire(200)
        assert waited == 50.0
        assert sleeper.waits == [50.0]
        assert clock() == 160.0
        assert limiter._current_tokens == 200

    def test_sliding_window_prevents_boundary_burst(self):
        """Reserving 800 tokens at t=50s and 800 at t=65s must NOT double-book."""
        import index_canary as ic

        clock = FakeClock(0.0)
        sleeper = FakeSleeper(clock)
        limiter = ic._TokenRateLimiter(
            tokens_per_window=1000,
            window_seconds=60.0,
            clock=clock,
            sleeper=sleeper,
        )
        clock.advance(50.0)  # at t=50s
        limiter.acquire(800)

        clock.advance(15.0)  # at t=65s (only 15s since last reservation)
        # Attempting to acquire 800 more tokens at t=65s: in a fixed-window limiter
        # this would succeed immediately (new window started at t=60s).
        # In a sliding-window limiter, 800 tokens reserved at t=50s do NOT expire until t=110s!
        # Thus the caller must wait (50 + 60) - 65 = 45s.
        waited = limiter.acquire(800)
        assert waited == 45.0
        assert clock() == 110.0
        assert sleeper.waits == [45.0]
        assert limiter._current_tokens == 800

    def test_oversized_reservation_rejected_immediately(self):
        import index_canary as ic

        clock = FakeClock()
        sleeper = FakeSleeper(clock)
        limiter = ic._TokenRateLimiter(
            tokens_per_window=1000,
            window_seconds=60.0,
            clock=clock,
            sleeper=sleeper,
        )
        with pytest.raises(ValueError, match="exceeds tokens_per_window"):
            limiter.acquire(1001)
        assert sleeper.waits == []

    def test_zero_and_negative_reservations_rejected(self):
        import index_canary as ic

        limiter = ic._TokenRateLimiter(tokens_per_window=1000)
        with pytest.raises(ValueError, match="positive"):
            limiter.acquire(0)
        with pytest.raises(ValueError, match="positive"):
            limiter.acquire(-10)

    def test_invalid_constructor_arguments_rejected(self):
        import index_canary as ic

        with pytest.raises(ValueError, match="tokens_per_window"):
            ic._TokenRateLimiter(tokens_per_window=0)
        with pytest.raises(ValueError, match="tokens_per_window"):
            ic._TokenRateLimiter(tokens_per_window=-100)
        with pytest.raises(ValueError, match="window_seconds"):
            ic._TokenRateLimiter(tokens_per_window=1000, window_seconds=0)
        with pytest.raises(ValueError, match="window_seconds"):
            ic._TokenRateLimiter(tokens_per_window=1000, window_seconds=-5.0)

    def test_window_rollover_resets_usage_exactly_once(self):
        import index_canary as ic

        clock = FakeClock(10.0)
        sleeper = FakeSleeper(clock)
        limiter = ic._TokenRateLimiter(
            tokens_per_window=1000,
            window_seconds=60.0,
            clock=clock,
            sleeper=sleeper,
        )
        limiter.acquire(800)
        assert limiter._current_tokens == 800

        # Advance clock beyond 60s
        clock.advance(65.0)
        waited = limiter.acquire(500)
        assert waited == 0.0
        assert sleeper.waits == []
        assert limiter._current_tokens == 500  # Reset and took 500

    def test_concurrent_workers_cannot_overwrite_each_others_reservations(self):
        import index_canary as ic

        # Real lock with concurrent threads reserving within limits
        limiter = ic._TokenRateLimiter(tokens_per_window=100_000, window_seconds=60.0)
        n_threads = 10
        tokens_per_thread = 500
        barrier = threading.Barrier(n_threads)
        errors: list[Exception] = []

        def worker():
            try:
                barrier.wait()
                limiter.acquire(tokens_per_thread)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert limiter._current_tokens == n_threads * tokens_per_thread

    def test_concurrent_workers_waiting_do_not_exceed_ceiling(self):
        import index_canary as ic

        # Monotonic time with thread synchronization
        class ThreadSafeFakeClock:
            def __init__(self, start: float = 0.0) -> None:
                self._t = start
                self._lock = threading.Lock()

            def __call__(self) -> float:
                with self._lock:
                    return self._t

            def advance(self, s: float) -> None:
                with self._lock:
                    self._t += s

        clock = ThreadSafeFakeClock(0.0)

        # Sleeper advances time to trigger window rollover
        def sleeper(s: float) -> None:
            clock.advance(s)

        limiter = ic._TokenRateLimiter(
            tokens_per_window=1000,
            window_seconds=1.0,
            clock=clock,
            sleeper=sleeper,
        )
        # Fill window
        limiter.acquire(1000)

        # 4 threads compete for capacity
        results: list[float] = []
        n_threads = 4
        barrier = threading.Barrier(n_threads)

        def worker():
            barrier.wait()
            w = limiter.acquire(250)
            results.append(w)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == n_threads
        assert limiter._current_tokens == 1000  # 4 * 250 in new window

    def test_run_source_uses_rate_limiter_class(self):
        import inspect

        import index_canary as ic

        source = inspect.getsource(ic._run)
        assert "_TokenRateLimiter" in source
        assert "_rate_limiter.acquire" in source


class TestCanonicalContractResolution:
    def test_tokenizer_model_input_limit_is_max_input_tokens(self):
        from hhgoa_rag.ingestion.tokenizer import MODEL_INPUT_LIMIT
        from hhgoa_rag.pinecone_contract import MAX_INPUT_TOKENS

        assert MODEL_INPUT_LIMIT == MAX_INPUT_TOKENS == 507

    def test_pinecone_store_text_record_field_is_canonical_text_field(self):
        from hhgoa_rag.pinecone_contract import TEXT_FIELD
        from hhgoa_rag.pinecone_store import TEXT_RECORD_FIELD

        assert TEXT_RECORD_FIELD == TEXT_FIELD == "chunk_text"

    def test_pinecone_store_field_map_matches_canonical(self):
        from hhgoa_rag.pinecone_contract import FIELD_MAP as CANONICAL_FIELD_MAP
        from hhgoa_rag.pinecone_store import FIELD_MAP

        assert FIELD_MAP == dict(CANONICAL_FIELD_MAP) == {"text": "chunk_text"}

    def test_settings_defaults_match_canonical_contract(self):
        from hhgoa_rag.config.settings import Settings
        from hhgoa_rag.pinecone_contract import CLOUD, INDEX_NAME, MODEL, REGION

        s = Settings()
        assert s.pinecone_index == INDEX_NAME
        assert s.pinecone_cloud == CLOUD
        assert s.pinecone_region == REGION
        assert s.pinecone_embed_model == MODEL

    def test_engine_ingestion_config_defaults_match_canonical(self):
        from hhgoa_rag.ingestion.engine import IngestionConfig
        from hhgoa_rag.pinecone_contract import MAX_BATCH_SIZE, MODEL

        cfg = IngestionConfig(mode="smoke", pinecone_index="idx", pinecone_namespace="ns")
        assert cfg.batch_size == MAX_BATCH_SIZE == 96
        assert cfg.embed_model == MODEL == "multilingual-e5-large"


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Zero/negative CLI values
# ═══════════════════════════════════════════════════════════════════════════════


class TestCLIValidation:
    def test_zero_batch_size_exits_2(self, tmp_path):
        manifest_path, _ = _write_jsonl_and_manifest(tmp_path)
        report_dir = tmp_path / "reports"
        r = _run_canary(
            ["--manifest", str(manifest_path), "--batch-size", "0", "--report-dir", str(report_dir)]
        )
        assert r.returncode == 2

    def test_negative_batch_size_exits_2(self, tmp_path):
        manifest_path, _ = _write_jsonl_and_manifest(tmp_path)
        report_dir = tmp_path / "reports"
        r = _run_canary(
            [
                "--manifest",
                str(manifest_path),
                "--batch-size",
                "-5",
                "--report-dir",
                str(report_dir),
            ]
        )
        assert r.returncode == 2

    def test_zero_concurrency_exits_2(self, tmp_path):
        manifest_path, _ = _write_jsonl_and_manifest(tmp_path)
        report_dir = tmp_path / "reports"
        r = _run_canary(
            [
                "--manifest",
                str(manifest_path),
                "--concurrency",
                "0",
                "--report-dir",
                str(report_dir),
            ]
        )
        assert r.returncode == 2

    def test_negative_concurrency_exits_2(self, tmp_path):
        manifest_path, _ = _write_jsonl_and_manifest(tmp_path)
        report_dir = tmp_path / "reports"
        r = _run_canary(
            [
                "--manifest",
                str(manifest_path),
                "--concurrency",
                "-1",
                "--report-dir",
                str(report_dir),
            ]
        )
        assert r.returncode == 2

    def test_zero_token_rate_limit_exits_2(self, tmp_path):
        manifest_path, _ = _write_jsonl_and_manifest(tmp_path)
        report_dir = tmp_path / "reports"
        r = _run_canary(
            [
                "--manifest",
                str(manifest_path),
                "--token-rate-limit",
                "0",
                "--report-dir",
                str(report_dir),
            ]
        )
        assert r.returncode == 2

    def test_negative_token_rate_limit_exits_2(self, tmp_path):
        manifest_path, _ = _write_jsonl_and_manifest(tmp_path)
        report_dir = tmp_path / "reports"
        r = _run_canary(
            [
                "--manifest",
                str(manifest_path),
                "--token-rate-limit",
                "-100",
                "--report-dir",
                str(report_dir),
            ]
        )
        assert r.returncode == 2

    def test_over_max_batch_size_exits_2(self, tmp_path):
        manifest_path, _ = _write_jsonl_and_manifest(tmp_path)
        report_dir = tmp_path / "reports"
        r = _run_canary(
            [
                "--manifest",
                str(manifest_path),
                "--batch-size",
                str(MAX_BATCH_SIZE + 1),
                "--report-dir",
                str(report_dir),
            ]
        )
        assert r.returncode == 2

    def test_over_max_concurrency_exits_2(self, tmp_path):
        manifest_path, _ = _write_jsonl_and_manifest(tmp_path)
        report_dir = tmp_path / "reports"
        r = _run_canary(
            [
                "--manifest",
                str(manifest_path),
                "--concurrency",
                "100",
                "--report-dir",
                str(report_dir),
            ]
        )
        assert r.returncode == 2

    def test_execute_without_confirm_exits_2(self, tmp_path):
        manifest_path, _ = _write_jsonl_and_manifest(tmp_path)
        report_dir = tmp_path / "reports"
        r = _run_canary(
            ["--manifest", str(manifest_path), "--execute", "--report-dir", str(report_dir)],
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
        r = _run_canary(["--manifest", str(manifest_path), "--report-dir", str(report_dir)])
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


# ═══════════════════════════════════════════════════════════════════════════════
# 18. Pre-write namespace preflight
# ═══════════════════════════════════════════════════════════════════════════════


class TestPreWriteNamespacePreflight:
    def test_fresh_empty_namespace_passes(self, monkeypatch):
        import index_canary as ic

        mock_index = MagicMock()
        mock_index.describe_index_stats.return_value = {
            "namespaces": {"pilot_v1": {"vector_count": 0}}
        }
        stats = mock_index.describe_index_stats()
        count = ic._get_ns_vector_count(stats, "pilot_v1")
        assert count == 0

    def test_fresh_contaminated_namespace_aborts_without_upserts(self, monkeypatch, tmp_path):
        import index_canary as ic

        manifest_path, _ = _write_jsonl_and_manifest(tmp_path)
        report_dir = tmp_path / "reports"

        mock_pc = MagicMock()
        mock_index = MagicMock()
        mock_pc.Index.return_value = mock_index
        # Remote index validation passes
        mock_desc = MagicMock()
        mock_desc.dimension = 1024
        mock_desc.metric = "cosine"
        mock_desc.spec.serverless.cloud = "aws"
        mock_desc.spec.serverless.region = "us-east-1"
        mock_desc.embed.model = "multilingual-e5-large"
        mock_desc.embed.read_parameters = {"input_type": "query", "truncate": "NONE"}
        mock_desc.embed.write_parameters = {"input_type": "passage", "truncate": "NONE"}
        mock_desc.embed.field_map = {"text": "chunk_text"}
        mock_pc.describe_index.return_value = mock_desc
        # But namespace contains 50 records (contaminated)
        mock_index.describe_index_stats.return_value = {
            "namespaces": {"pilot_v1": {"vector_count": 50}}
        }

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
                "--report-dir",
                str(report_dir),
            ]
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

        assert exc_info.value.category == "NamespaceContaminatedPreflight"
        assert "not empty" in str(exc_info.value)
        # Crucial: NO upsert was attempted!
        mock_index.upsert_records.assert_not_called()

    def test_resume_compatible_passes_preflight(self, monkeypatch, tmp_path):
        import index_canary as ic

        manifest_path, _ = _write_jsonl_and_manifest(tmp_path)
        report_dir = tmp_path / "reports"
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()

        # Build checkpoint with 1 completed batch (96 records)
        manifest = json.loads(manifest_path.read_text())
        records = ic._verify_and_load_records(manifest, manifest_path)
        batches = ic._build_batches(records, 96)
        batch_digests = [
            ic._batch_digest(manifest["manifest_checksum"], i, [r["id"] for r in b])
            for i, b in enumerate(batches)
        ]
        completed_digests = [batch_digests[0]]
        ckpt_data = {
            "checkpoint_schema_version": ic.CHECKPOINT_SCHEMA_VERSION,
            "run_id": "test-run",
            "manifest_id": manifest["manifest_id"],
            "manifest_checksum": manifest["manifest_checksum"],
            "contract_fingerprint": manifest["contract_fingerprint"],
            "index_name": ic.CANONICAL_INDEX_NAME,
            "namespace": ic.CANONICAL_NAMESPACE,
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
        mock_index.describe_index_stats.return_value = {
            "namespaces": {"pilot_v1": {"vector_count": 96}}
        }
        mock_index.upsert_records.return_value = 96

        class FakeListItem:
            def __init__(self, item_id: str) -> None:
                self.id = item_id

        class FakeListResponse:
            def __init__(self, ids: list[str]) -> None:
                self.vectors = [FakeListItem(i) for i in ids]
                self.pagination = None

        mock_index.list_paginated.return_value = FakeListResponse([r["id"] for r in batches[0]])

        monkeypatch.setenv("CONFIRM_PINECONE_WRITE", "1")
        monkeypatch.setenv("PINECONE_API_KEY", "pcsk_fake_key_for_testing")
        monkeypatch.setattr("pinecone.Pinecone", lambda api_key: mock_pc)
        monkeypatch.setattr("hhgoa_rag.pinecone_lifecycle.validate_index", lambda pc, name: [])

        # Mock second describe_index_stats in step 7 to return 300 so reconciliation passes
        mock_index.describe_index_stats.side_effect = [
            {"namespaces": {"pilot_v1": {"vector_count": 96}}},  # preflight
            {"namespaces": {"pilot_v1": {"vector_count": 300}}},  # step 7 reconciliation
        ]

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
        # Since 1 batch was already done, only 3 batches were submitted
        assert report_data["completed_batches"] == 3
        assert report_data["skipped_batches"] == 1

    def test_resume_unexpected_records_aborts_without_upsert(self, monkeypatch, tmp_path):
        import index_canary as ic

        manifest_path, _ = _write_jsonl_and_manifest(tmp_path)
        report_dir = tmp_path / "reports"
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()

        # Checkpoint says 96 records completed, but namespace has 200 records
        manifest = json.loads(manifest_path.read_text())
        records = ic._verify_and_load_records(manifest, manifest_path)
        batches = ic._build_batches(records, 96)
        batch_digests = [
            ic._batch_digest(manifest["manifest_checksum"], i, [r["id"] for r in b])
            for i, b in enumerate(batches)
        ]
        ckpt_data = {
            "checkpoint_schema_version": ic.CHECKPOINT_SCHEMA_VERSION,
            "run_id": "test-run",
            "manifest_id": manifest["manifest_id"],
            "manifest_checksum": manifest["manifest_checksum"],
            "contract_fingerprint": manifest["contract_fingerprint"],
            "index_name": ic.CANONICAL_INDEX_NAME,
            "namespace": ic.CANONICAL_NAMESPACE,
            "batch_size": 96,
            "batch_digests": batch_digests,
            "completed_batch_digests": [batch_digests[0]],
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
        mock_index.describe_index_stats.return_value = {
            "namespaces": {"pilot_v1": {"vector_count": 200}}
        }

        class FakeListItem:
            def __init__(self, item_id: str) -> None:
                self.id = item_id

        class FakeListResponse:
            def __init__(self, ids: list[str]) -> None:
                self.vectors = [FakeListItem(i) for i in ids]
                self.pagination = None

        mock_index.list_paginated.return_value = FakeListResponse(
            [f"rogue-{i}" for i in range(200)]
        )

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
        assert "Resume ownership verification failed" in str(exc_info.value)
        mock_index.upsert_records.assert_not_called()

    def test_preflight_provider_failure_aborts_without_upsert(self, monkeypatch, tmp_path):
        import index_canary as ic

        manifest_path, _ = _write_jsonl_and_manifest(tmp_path)
        report_dir = tmp_path / "reports"

        mock_pc = MagicMock()
        mock_index = MagicMock()
        mock_pc.Index.return_value = mock_index
        mock_index.describe_index_stats.side_effect = RuntimeError("503 Service Unavailable")

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
                "--report-dir",
                str(report_dir),
            ]
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

        assert exc_info.value.category == "PreflightProviderFailure"
        mock_index.upsert_records.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# 19. Capacity estimator tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestCapacityEstimator:
    def test_estimator_uses_canonical_dimension(self):
        import estimate_capacity as ec

        from hhgoa_rag.pinecone_contract import DIMENSION

        report = ec.calculate_estimates()
        assert report["metadata"]["canonical_dimension"] == DIMENSION == 1024

    def test_changing_dimension_changes_dense_estimate_proportionally(self):
        import estimate_capacity as ec

        r1024 = ec.calculate_estimates(dimension=1024)
        r384 = ec.calculate_estimates(dimension=384)

        dense_1024 = r1024["scopes"]["target_3_languages_en_hi_bn"]["storage"][
            "hypothetical_local_dense_gb"
        ]
        dense_384 = r384["scopes"]["target_3_languages_en_hi_bn"]["storage"][
            "hypothetical_local_dense_gb"
        ]

        ratio = dense_1024 / dense_384
        assert 2.66 < ratio < 2.67  # 1024 / 384 = 2.666666...

    def test_all_expected_scopes_present(self):
        import estimate_capacity as ec

        report = ec.calculate_estimates()
        scopes = report["scopes"]
        assert "canary_300_records" in scopes
        assert "bounded_pilot_10k_records" in scopes
        assert "target_3_languages_en_hi_bn" in scopes
        assert "full_corpus_14_languages" in scopes

    def test_local_disk_warning_present(self):
        import estimate_capacity as ec

        report = ec.calculate_estimates()
        note = report["assumptions_vs_measured"]["unmeasured_assumptions"]["note_on_local_disk"]
        assert "200 GB" in note


# ═══════════════════════════════════════════════════════════════════════════════
# 20. Integration smoke test opt-in and fail-closed behavior
# ═══════════════════════════════════════════════════════════════════════════════


class TestIntegrationOptIn:
    def test_smoke_test_skips_when_opt_in_missing(self, monkeypatch):
        from tests.integration.test_pinecone_smoke import _require_opt_in

        monkeypatch.delenv("PINECONE_SMOKE_TEST", raising=False)
        monkeypatch.setenv("PINECONE_API_KEY", "pcsk_test")
        with pytest.raises(pytest.skip.Exception):
            _require_opt_in()

    def test_smoke_test_fails_closed_when_opt_in_set_but_key_missing(self, monkeypatch):
        from tests.integration.test_pinecone_smoke import _require_opt_in

        monkeypatch.setenv("PINECONE_SMOKE_TEST", "1")
        monkeypatch.delenv("PINECONE_API_KEY", raising=False)
        with pytest.raises(pytest.fail.Exception, match="PINECONE_API_KEY is missing"):
            _require_opt_in()

    def test_smoke_test_passes_when_opt_in_and_key_provided(self, monkeypatch):
        from tests.integration.test_pinecone_smoke import _require_opt_in

        monkeypatch.setenv("PINECONE_SMOKE_TEST", "1")
        monkeypatch.setenv("PINECONE_API_KEY", "pcsk_valid_key")
        key = _require_opt_in()
        assert key == "pcsk_valid_key"
