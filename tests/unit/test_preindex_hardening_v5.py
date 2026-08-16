"""Adversarial tests for the final pre-index hardening pass (v5).

Covers:
  Fix 1 — Strict integer acceptance in PineconeStore.count_namespace() AND
          index_canary._get_ns_vector_count(): reject floats, numeric strings,
          bool, negatives, None; accept only genuine non-negative int.
  Fix 2 — Environment-only Pinecone credentials: --pinecone-api-key removed
          from reconcile/describe/smoke/validate; missing env fails before
          provider construction.
  Fix 3 — reconcile_corpus.py --expected-count secondary count assertion.

No live provider calls are made anywhere in this file.
"""

from __future__ import annotations

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

from hhgoa_rag.pinecone_store import (  # noqa: E402
    PineconeProviderError,
    PineconeStore,
)

FAKE_KEY = "fake-api-key-for-testing"


def _stats_with_count(value: object) -> MagicMock:
    """Build a mock describe_index_stats() response with pilot_v1 vector_count=value."""
    stats = MagicMock()
    ns_info = MagicMock()
    ns_info.vector_count = value
    stats.namespaces = {"pilot_v1": ns_info}
    return stats


def _store_returning(value: object) -> PineconeStore:
    index = MagicMock()
    index.describe_index_stats.return_value = _stats_with_count(value)
    return PineconeStore(index, embed_model="multilingual-e5-large")


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 1 — Strict integer acceptance
# ═══════════════════════════════════════════════════════════════════════════════


# Accepted: genuine non-negative ints.
ACCEPTED = [0, 1, 300]
# Rejected: bool, floats, numeric strings, negatives.
REJECTED = [True, False, 300.0, 300.9, 0.0, 0.5, "300", "0", -1, -5]


class TestCountNamespaceStrictInteger:
    @pytest.mark.parametrize("value", ACCEPTED)
    def test_accepts_genuine_int(self, value):
        assert _store_returning(value).count_namespace("pilot_v1") == value

    @pytest.mark.parametrize("value", REJECTED)
    def test_rejects_malformed(self, value):
        with pytest.raises(PineconeProviderError):
            _store_returning(value).count_namespace("pilot_v1")

    def test_none_count_rejected(self):
        with pytest.raises(PineconeProviderError):
            _store_returning(None).count_namespace("pilot_v1")

    def test_missing_vector_count_rejected(self):
        index = MagicMock()
        stats = MagicMock()
        # ns_info is a plain dict WITHOUT a vector_count key.
        stats.namespaces = {"pilot_v1": {}}
        index.describe_index_stats.return_value = stats
        store = PineconeStore(index, embed_model="multilingual-e5-large")
        with pytest.raises(PineconeProviderError):
            store.count_namespace("pilot_v1")

    def test_float_never_coerced_to_int(self):
        # 300.9 must NOT become 300.
        with pytest.raises(PineconeProviderError):
            _store_returning(300.9).count_namespace("pilot_v1")


class TestGetNsVectorCountStrictInteger:
    @pytest.mark.parametrize("value", ACCEPTED)
    def test_accepts_genuine_int(self, value):
        assert ic._get_ns_vector_count(_stats_with_count(value), "pilot_v1") == value

    @pytest.mark.parametrize("value", REJECTED + [None])
    def test_rejects_malformed_returns_none(self, value):
        # Canary path returns None ("unverifiable") for malformed values.
        assert ic._get_ns_vector_count(_stats_with_count(value), "pilot_v1") is None

    def test_string_never_coerced(self):
        assert ic._get_ns_vector_count(_stats_with_count("300"), "pilot_v1") is None

    def test_float_never_coerced(self):
        assert ic._get_ns_vector_count(_stats_with_count(300.0), "pilot_v1") is None


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 2 — Environment-only Pinecone credentials
# ═══════════════════════════════════════════════════════════════════════════════

_ENV_ONLY_SCRIPTS = [
    "reconcile_corpus.py",
    "describe_pinecone_index.py",
    "smoke_query_pinecone.py",
    "validate_pinecone_config.py",
]


class TestEnvironmentOnlyCredentials:
    @pytest.mark.parametrize("script", _ENV_ONLY_SCRIPTS)
    def test_help_has_no_api_key_flag(self, script):
        result = subprocess.run(
            [sys.executable, f"scripts/{script}", "--help"],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
        )
        assert "--pinecone-api-key" not in result.stdout
        assert "--pinecone-api-key" not in result.stderr

    @pytest.mark.parametrize("script", _ENV_ONLY_SCRIPTS)
    def test_api_key_flag_is_rejected(self, script):
        result = subprocess.run(
            [
                sys.executable,
                f"scripts/{script}",
                *_required_args(script),
                "--pinecone-api-key",
                FAKE_KEY,
            ],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
        )
        # argparse rejects an unknown option with exit code 2.
        assert result.returncode == 2
        assert "--pinecone-api-key" in result.stderr

    @pytest.mark.parametrize("script", _ENV_ONLY_SCRIPTS)
    def test_missing_env_credential_fails_before_provider(self, script):
        env = {k: v for k, v in _clean_env().items() if k not in ("PINECONE_API_KEY",)}
        result = subprocess.run(
            [sys.executable, f"scripts/{script}", *_required_args(script)],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
            env=env,
        )
        assert result.returncode != 0
        assert "PINECONE_API_KEY" in result.stderr

    @pytest.mark.parametrize("script", _ENV_ONLY_SCRIPTS)
    @pytest.mark.parametrize("blank", ["", " ", "   ", "\t", "\n", " \t\n "])
    def test_blank_env_credential_fails_before_provider(self, script, blank):
        env = _clean_env()
        env["PINECONE_API_KEY"] = blank
        result = subprocess.run(
            [sys.executable, f"scripts/{script}", *_required_args(script)],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
            env=env,
        )
        assert result.returncode != 0
        assert "PINECONE_API_KEY" in result.stderr
        # The blank value must never be echoed back.
        assert blank not in result.stdout or blank.strip() == ""


def _required_args(script: str) -> list[str]:
    """Minimum required CLI args so parsing succeeds and reaches credential check."""
    if script == "reconcile_corpus.py":
        return ["--expected-count", "300"]
    return []


def _clean_env() -> dict:
    import os

    env = dict(os.environ)
    env.pop("PINECONE_API_KEY", None)
    return env


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 3 — reconcile_corpus.py --expected-count
# ═══════════════════════════════════════════════════════════════════════════════


def _run_reconcile(args: list[str], count_value: object, provider_error: bool = False):
    """Run reconcile_corpus.main() in-process with a mocked PineconeStore.

    Returns the SystemExit code and the captured JSON (if --output-json).
    """
    import importlib
    import os

    rc = importlib.import_module("reconcile_corpus")

    fake_store = MagicMock()
    if provider_error:
        fake_store.count_namespace.side_effect = PineconeProviderError("boom")
    else:
        fake_store.count_namespace.return_value = count_value
    fake_store.describe_index_stats.return_value = {"namespaces": {}}

    fake_pc_module = MagicMock()
    fake_pc_module.Pinecone.return_value.Index.return_value = MagicMock()

    import sys as _sys

    orig_argv = _sys.argv
    orig_env = os.environ.get("PINECONE_API_KEY")
    os.environ["PINECONE_API_KEY"] = FAKE_KEY
    _sys.argv = ["reconcile_corpus.py", *args]

    import unittest.mock as um

    try:
        with (
            um.patch.dict(
                _sys.modules,
                {"pinecone": fake_pc_module},
            ),
            um.patch.object(rc, "PineconeStore", return_value=fake_store, create=True),
            um.patch("hhgoa_rag.pinecone_store.PineconeStore", return_value=fake_store),
        ):
            try:
                rc.main()
                return 0
            except SystemExit as e:
                return e.code if isinstance(e.code, int) else 1
    finally:
        _sys.argv = orig_argv
        if orig_env is None:
            os.environ.pop("PINECONE_API_KEY", None)
        else:
            os.environ["PINECONE_API_KEY"] = orig_env


def _run_reconcile_json(
    args: list[str], count_value: object, provider_error: bool = False
) -> tuple[int, dict]:
    """Run reconcile_corpus.main() in-process with a mocked store, capturing the
    JSON printed to stdout. No live provider call occurs (pinecone is stubbed).
    """
    import contextlib
    import importlib
    import io
    import os
    import unittest.mock as um

    rc = importlib.import_module("reconcile_corpus")

    fake_store = MagicMock()
    if provider_error:
        fake_store.count_namespace.side_effect = PineconeProviderError("boom")
    else:
        fake_store.count_namespace.return_value = count_value
    fake_store.describe_index_stats.return_value = {"namespaces": {}}

    fake_pc_module = MagicMock()
    fake_pc_module.Pinecone.return_value.Index.return_value = MagicMock()

    orig_argv = sys.argv
    orig_env = os.environ.get("PINECONE_API_KEY")
    os.environ["PINECONE_API_KEY"] = FAKE_KEY
    sys.argv = ["reconcile_corpus.py", *args]

    buf = io.StringIO()
    code = 0
    try:
        with (
            um.patch.dict(sys.modules, {"pinecone": fake_pc_module}),
            um.patch.object(rc, "PineconeStore", return_value=fake_store, create=True),
            um.patch("hhgoa_rag.pinecone_store.PineconeStore", return_value=fake_store),
            contextlib.redirect_stdout(buf),
        ):
            try:
                rc.main()
            except SystemExit as e:
                code = e.code if isinstance(e.code, int) else 1
    finally:
        sys.argv = orig_argv
        if orig_env is None:
            os.environ.pop("PINECONE_API_KEY", None)
        else:
            os.environ["PINECONE_API_KEY"] = orig_env

    out = buf.getvalue()
    start = out.find("{")
    payload = json.loads(out[start:]) if start >= 0 else {}
    return code, payload


class TestReconcileExpectedCount:
    def test_exact_match_exits_zero(self):
        code = _run_reconcile(["--namespace", "pilot_v1", "--expected-count", "300"], 300)
        assert code == 0

    def test_lower_count_exits_nonzero(self):
        code = _run_reconcile(["--namespace", "pilot_v1", "--expected-count", "300"], 299)
        assert code != 0

    def test_higher_count_exits_nonzero(self):
        code = _run_reconcile(["--namespace", "pilot_v1", "--expected-count", "300"], 301)
        assert code != 0

    def test_provider_failure_exits_nonzero(self):
        code = _run_reconcile(
            ["--namespace", "pilot_v1", "--expected-count", "300"], 0, provider_error=True
        )
        assert code != 0

    def test_malformed_count_exits_nonzero(self):
        # count_namespace raises for malformed values; reconcile catches and exits nonzero.
        code = _run_reconcile(
            ["--namespace", "pilot_v1", "--expected-count", "300"], 0, provider_error=True
        )
        assert code != 0

    def test_missing_expectation_fails_before_provider(self):
        # --expected-count is REQUIRED. Its absence must exit non-zero, and must
        # do so before any provider construction. argparse exits with code 2.
        result = subprocess.run(
            [sys.executable, "scripts/reconcile_corpus.py", "--namespace", "pilot_v1"],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
        )
        assert result.returncode != 0
        assert "expected-count" in result.stderr

    def test_invalid_expectation_zero_exits_nonzero(self):
        code = _run_reconcile(["--namespace", "pilot_v1", "--expected-count", "0"], 300)
        assert code != 0

    def test_invalid_expectation_negative_exits_nonzero(self):
        code = _run_reconcile(["--namespace", "pilot_v1", "--expected-count", "-5"], 300)
        assert code != 0

    def test_invalid_expectation_float_rejected_by_argparse(self):
        # type=int rejects a float literal before the provider is constructed.
        result = subprocess.run(
            [
                sys.executable,
                "scripts/reconcile_corpus.py",
                "--namespace",
                "pilot_v1",
                "--expected-count",
                "300.5",
            ],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
        )
        assert result.returncode != 0

    def test_json_output_exact_match_is_accurate_and_secret_free(self):
        rc, payload = _run_reconcile_json(
            ["--namespace", "pilot_v1", "--expected-count", "300", "--output-json"], 300
        )
        assert rc == 0
        assert payload["index"]
        assert payload["namespace"] == "pilot_v1"
        assert payload["expected_count"] == 300
        assert payload["actual_count"] == 300
        assert payload["status"] == "pass"
        assert FAKE_KEY not in json.dumps(payload)

    def test_json_output_mismatch_is_accurate(self):
        rc, payload = _run_reconcile_json(
            ["--namespace", "pilot_v1", "--expected-count", "300", "--output-json"], 299
        )
        assert rc != 0
        assert payload["actual_count"] == 299
        assert payload["status"] == "mismatch"

    def test_json_output_provider_error_is_unverifiable_and_secret_free(self):
        rc, payload = _run_reconcile_json(
            ["--namespace", "pilot_v1", "--expected-count", "300", "--output-json"],
            0,
            provider_error=True,
        )
        assert rc != 0
        assert payload["actual_count"] is None
        assert payload["status"] == "unverifiable"
        assert FAKE_KEY not in json.dumps(payload)


# ═══════════════════════════════════════════════════════════════════════════════
# Fix 4 — Secret scanner covers tracked files under artifacts/ (no broad exempt)
# ═══════════════════════════════════════════════════════════════════════════════


def _load_scan_secrets():
    import importlib

    return importlib.import_module("scan_secrets")


class TestSecretScanner:
    def test_artifacts_not_broadly_exempt(self):
        ss = _load_scan_secrets()
        assert "artifacts" not in ss._SKIP_PREFIXES

    def test_detects_fake_secret_in_artifacts_text_fixture(self, tmp_path):
        ss = _load_scan_secrets()
        artifacts_dir = tmp_path / "artifacts" / "reports"
        artifacts_dir.mkdir(parents=True)
        fixture = artifacts_dir / "leaky_fixture.json"
        # Obvious fake Pinecone key pattern.
        fixture.write_text('{"api_key": "pcsk_' + "a" * 40 + '"}')
        hits = ss.scan_file(fixture)
        assert hits, "Scanner must detect a fake secret in a tracked artifacts fixture"

    def test_binary_file_skipped(self, tmp_path):
        ss = _load_scan_secrets()
        binf = tmp_path / "blob.json"
        binf.write_bytes(b"\x00\x01pcsk_" + b"a" * 40)
        assert ss.scan_file(binf) == []

    def test_scanner_output_names_file_without_secret(self, tmp_path, capsys, monkeypatch):
        ss = _load_scan_secrets()
        secret = "pcsk_" + "b" * 40
        (tmp_path / "config.json").write_text('{"k": "' + secret + '"}')
        # Not a git tree → filesystem-walk fallback exercised.
        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit) as exc:
            ss.main()
        assert exc.value.code == 1
        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "config.json" in combined
        assert secret not in combined

    def test_clean_tree_passes(self, tmp_path, capsys, monkeypatch):
        ss = _load_scan_secrets()
        (tmp_path / "ok.py").write_text("x = 1\n")
        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit) as exc:
            ss.main()
        assert exc.value.code == 0
