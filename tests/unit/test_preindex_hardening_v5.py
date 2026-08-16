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
            [sys.executable, f"scripts/{script}", "--pinecone-api-key", FAKE_KEY],
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
            [sys.executable, f"scripts/{script}"],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
            env=env,
        )
        assert result.returncode != 0
        assert "PINECONE_API_KEY" in result.stderr


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

    def test_missing_expectation_still_reports(self):
        # No --expected-count: status is pass, exit 0.
        code = _run_reconcile(["--namespace", "pilot_v1"], 300)
        assert code == 0

    def test_invalid_expectation_zero_exits_nonzero(self):
        code = _run_reconcile(["--namespace", "pilot_v1", "--expected-count", "0"], 300)
        assert code != 0

    def test_invalid_expectation_negative_exits_nonzero(self):
        code = _run_reconcile(["--namespace", "pilot_v1", "--expected-count", "-5"], 300)
        assert code != 0
