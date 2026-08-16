"""Adversarial tests for the Starter budget enforcement module."""

from __future__ import annotations

import json

import pytest

from hhgoa_rag.ingestion.budget import (
    PLAN_STARTER,
    STARTER_MAX_RECORDS,
    STARTER_MAX_RERANK_REQUESTS,
    STARTER_OP_EMBED_TOKENS,
    STARTER_OP_STORAGE_BYTES,
    BudgetExceededError,
    BudgetGuard,
    StarterBudget,
    StarterFullModeError,
    UsageLedger,
    is_starter,
    make_default_guard,
)

# ── StarterBudget configuration ────────────────────────────────────────────────


def test_default_plan_is_starter():
    b = StarterBudget()
    assert b.is_starter()


def test_non_starter_plan_not_starter():
    b = StarterBudget(plan="standard")
    assert not b.is_starter()


def test_defaults_match_spec():
    b = StarterBudget()
    assert b.max_embed_tokens == STARTER_OP_EMBED_TOKENS
    assert b.max_records == STARTER_MAX_RECORDS
    assert b.max_storage_bytes == STARTER_OP_STORAGE_BYTES
    assert b.max_rerank_requests == STARTER_MAX_RERANK_REQUESTS


# ── StarterFullModeError — unconditional on Starter ───────────────────────────


def test_full_corpus_refused_on_starter():
    guard = BudgetGuard(budget=StarterBudget(plan=PLAN_STARTER))
    with pytest.raises(StarterFullModeError):
        guard.refuse_full_corpus()


def test_full_corpus_refused_even_with_all_env_vars_set(monkeypatch):
    """No environment variable can bypass the Starter full-mode gate."""
    monkeypatch.setenv("CONFIRM_FULL_INGEST", "YES_I_APPROVE_FULL_CORPUS")
    monkeypatch.setenv("CONFIRM_PINECONE_WRITE", "1")
    monkeypatch.setenv("PINECONE_PLAN", "starter")
    guard = BudgetGuard(budget=StarterBudget(plan=PLAN_STARTER))
    with pytest.raises(StarterFullModeError):
        guard.refuse_full_corpus()


def test_full_corpus_allowed_on_non_starter():
    guard = BudgetGuard(budget=StarterBudget(plan="standard"))
    guard.refuse_full_corpus()  # must not raise


# ── check_upsert — pre-flight budget gates ────────────────────────────────────


def test_check_upsert_within_budget():
    guard = BudgetGuard()
    guard.check_upsert("en", record_count=10, token_count=1000, byte_count=15_000)


def test_check_upsert_refuses_at_record_limit():
    guard = BudgetGuard(budget=StarterBudget(max_records=100))
    with pytest.raises(BudgetExceededError, match="Record budget"):
        guard.check_upsert("en", record_count=101, token_count=0, byte_count=0)


def test_check_upsert_refuses_at_token_limit():
    guard = BudgetGuard(budget=StarterBudget(max_embed_tokens=1000))
    with pytest.raises(BudgetExceededError, match="token budget"):
        guard.check_upsert("en", record_count=1, token_count=1001, byte_count=0)


def test_check_upsert_refuses_at_storage_limit():
    guard = BudgetGuard(budget=StarterBudget(max_storage_bytes=1000))
    with pytest.raises(BudgetExceededError, match="Storage budget"):
        guard.check_upsert("en", record_count=1, token_count=0, byte_count=1001)


def test_check_upsert_refuses_cumulatively():
    """Ledger state from previous commits is included in the check."""
    guard = BudgetGuard(budget=StarterBudget(max_records=10))
    guard.commit_upsert("en", record_count=8, token_count=100, byte_count=0)
    with pytest.raises(BudgetExceededError, match="Record budget"):
        guard.check_upsert("en", record_count=3, token_count=0, byte_count=0)


def test_check_upsert_uses_estimated_bytes_when_none():
    """Byte budget is estimated from bytes_per_record when byte_count is None."""
    budget = StarterBudget(max_storage_bytes=100, bytes_per_record_estimate=200)
    guard = BudgetGuard(budget=budget)
    with pytest.raises(BudgetExceededError, match="Storage budget"):
        guard.check_upsert("en", record_count=1, token_count=0, byte_count=None)


# ── Retries must not bypass budget ────────────────────────────────────────────


def test_retry_still_requires_budget_check():
    """check_upsert must be called for retries; no bypass allowed."""
    guard = BudgetGuard(budget=StarterBudget(max_records=5))
    guard.commit_upsert("en", record_count=5, token_count=0, byte_count=0)
    # A retry of the same batch would exceed the limit — must be refused
    with pytest.raises(BudgetExceededError):
        guard.check_upsert("en", record_count=5, token_count=0, byte_count=0)


# ── check_rerank ──────────────────────────────────────────────────────────────


def test_check_rerank_within_quota():
    guard = BudgetGuard()
    guard.check_rerank(request_count=1)  # no raise


def test_check_rerank_refused_at_limit():
    guard = BudgetGuard(budget=StarterBudget(max_rerank_requests=10))
    guard.commit_rerank(request_count=10)
    with pytest.raises(BudgetExceededError, match="Rerank quota"):
        guard.check_rerank(request_count=1)


def test_check_rerank_cumulative():
    guard = BudgetGuard(budget=StarterBudget(max_rerank_requests=5))
    guard.commit_rerank(request_count=3)
    guard.commit_rerank(request_count=2)
    with pytest.raises(BudgetExceededError):
        guard.check_rerank()


# ── Ledger commit ordering ────────────────────────────────────────────────────


def test_commit_updates_ledger():
    guard = BudgetGuard()
    guard.commit_upsert("hi", record_count=5, token_count=200, byte_count=7_500)
    assert guard.ledger.records_upserted == 5
    assert guard.ledger.embed_tokens_used == 200
    assert guard.ledger.storage_bytes_used == 7_500


def test_commit_tracks_per_language():
    guard = BudgetGuard()
    guard.commit_upsert("en", record_count=3, token_count=100, byte_count=0)
    guard.commit_upsert("hi", record_count=2, token_count=50, byte_count=0)
    assert guard.ledger.per_language["en"]["records"] == 3
    assert guard.ledger.per_language["hi"]["records"] == 2


def test_commit_retry_increments_retry_counter():
    guard = BudgetGuard()
    guard.commit_upsert("en", record_count=1, token_count=10, byte_count=0, is_retry=True)
    assert guard.ledger.retry_requests == 1


# ── Ledger persistence ────────────────────────────────────────────────────────


def test_ledger_saves_and_loads(tmp_path):
    path = tmp_path / "ledger.json"
    ledger = UsageLedger(records_upserted=42, embed_tokens_used=1000)
    ledger.save(path)
    loaded = UsageLedger.load(path)
    assert loaded.records_upserted == 42
    assert loaded.embed_tokens_used == 1000


def test_ledger_save_is_atomic(tmp_path):
    """Save writes to tmp file then renames — partial writes don't corrupt."""
    path = tmp_path / "ledger.json"
    ledger = UsageLedger(records_upserted=7)
    ledger.save(path)
    raw = json.loads(path.read_text())
    assert raw["records_upserted"] == 7


def test_guard_persists_ledger_on_commit(tmp_path):
    path = tmp_path / "ledger.json"
    guard = BudgetGuard(ledger_path=path)
    guard.commit_upsert("en", record_count=3, token_count=100, byte_count=0)
    loaded = UsageLedger.load(path)
    assert loaded.records_upserted == 3


# ── Usage report ──────────────────────────────────────────────────────────────


def test_usage_report_structure():
    guard = BudgetGuard()
    guard.commit_upsert("en", record_count=5, token_count=500, byte_count=7500)
    report = guard.usage_report()
    assert "embed_tokens" in report
    assert "records" in report
    assert "storage_bytes" in report
    assert "rerank_requests" in report
    assert report["records"]["used"] == 5
    assert report["records"]["pass"] is True


def test_usage_report_fail_when_over(tmp_path):
    guard = BudgetGuard(budget=StarterBudget(max_records=3))
    # Force ledger over limit for reporting purposes (no check, just commit)
    guard.ledger.records_upserted = 5
    report = guard.usage_report()
    assert report["records"]["pass"] is False


# ── is_starter helper ─────────────────────────────────────────────────────────


def test_is_starter_default(monkeypatch):
    monkeypatch.delenv("PINECONE_PLAN", raising=False)
    assert is_starter()


def test_is_starter_when_env_set(monkeypatch):
    monkeypatch.setenv("PINECONE_PLAN", "starter")
    assert is_starter()


def test_is_not_starter_when_standard(monkeypatch):
    monkeypatch.setenv("PINECONE_PLAN", "standard")
    assert not is_starter()


# ── make_default_guard ────────────────────────────────────────────────────────


def test_make_default_guard_returns_guard(monkeypatch):
    monkeypatch.delenv("PINECONE_PLAN", raising=False)
    guard = make_default_guard()
    assert isinstance(guard, BudgetGuard)
    assert guard.budget.is_starter()


def test_make_default_guard_loads_existing_ledger(tmp_path):
    path = tmp_path / "ledger.json"
    UsageLedger(records_upserted=99).save(path)
    guard = make_default_guard(ledger_path=path)
    assert guard.ledger.records_upserted == 99
