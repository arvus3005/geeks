"""Pinecone Starter zero-cost budget enforcement.

All ingestion, retrieval and reranking paths must check this module before
making any real Pinecone API call. The budget ledger is atomic and fail-closed.

Starter plan limits (configurable; defaults are conservative sub-limits):
  - Storage:        2 GB hard cap; 1.5 GB operational ceiling
  - Embedding:      5,000,000 tokens/month; 4,000,000 operational ingestion ceiling
  - Records:        10,000 pilot default
  - Rerank:         500 requests/month (reserve for live judging)
  - Write units:    configurable ceiling below 2,000,000

Full-corpus ingestion is permanently blocked when PINECONE_PLAN=starter.
No environment variable can bypass this check.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..pinecone_contract import FIELD_MAP as _CONTRACT_FIELD_MAP

# ── Plan constants ─────────────────────────────────────────────────────────────

PLAN_STARTER = "starter"

STARTER_HARD_STORAGE_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB
STARTER_OP_STORAGE_BYTES = int(1.5 * 1024 * 1024 * 1024)  # 1.5 GB ceiling
STARTER_MONTHLY_EMBED_TOKENS = 5_000_000
STARTER_OP_EMBED_TOKENS = 4_000_000  # ingestion ceiling; 1M reserved for queries
STARTER_MAX_RECORDS = 10_000
STARTER_MAX_RERANK_REQUESTS = 500
STARTER_MAX_WRITE_UNITS = 1_800_000  # conservative; hard Starter cap is ~2M

RERANK_MODEL = "bge-reranker-v2-m3"

# Re-export contract field map as a plain dict for JSON-compatibility
FIELD_MAP: dict[str, str] = dict(_CONTRACT_FIELD_MAP)

# Estimated bytes per indexed record (text + metadata overhead)
BYTES_PER_RECORD_ESTIMATE = 1_500


class BudgetExceededError(Exception):
    """Raised before any API call when a budget limit would be exceeded."""


class StarterFullModeError(Exception):
    """Raised unconditionally when full-corpus mode is attempted on Starter plan."""


# ── Budget configuration ───────────────────────────────────────────────────────


@dataclass
class StarterBudget:
    """Configurable Pinecone Starter operational budget."""

    plan: str = PLAN_STARTER
    max_embed_tokens: int = STARTER_OP_EMBED_TOKENS
    max_records: int = STARTER_MAX_RECORDS
    max_storage_bytes: int = STARTER_OP_STORAGE_BYTES
    max_rerank_requests: int = STARTER_MAX_RERANK_REQUESTS
    max_write_units: int = STARTER_MAX_WRITE_UNITS
    bytes_per_record_estimate: int = BYTES_PER_RECORD_ESTIMATE

    def is_starter(self) -> bool:
        return self.plan.lower() == PLAN_STARTER


# ── Usage ledger ───────────────────────────────────────────────────────────────


@dataclass
class UsageLedger:
    """Atomic counters tracking actual and projected Pinecone consumption."""

    embed_tokens_used: int = 0
    records_upserted: int = 0
    storage_bytes_used: int = 0
    rerank_requests_used: int = 0
    write_units_used: int = 0
    retry_requests: int = 0
    per_language: dict[str, dict[str, int]] = field(default_factory=dict)

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def account_upsert(
        self,
        language: str,
        record_count: int,
        token_count: int,
        byte_count: int,
        is_retry: bool = False,
    ) -> None:
        with self._lock:
            self.records_upserted += record_count
            self.embed_tokens_used += token_count
            self.storage_bytes_used += byte_count
            if is_retry:
                self.retry_requests += 1
            lang = self.per_language.setdefault(language, {"records": 0, "tokens": 0})
            lang["records"] += record_count
            lang["tokens"] += token_count

    def account_rerank(self, request_count: int = 1) -> None:
        with self._lock:
            self.rerank_requests_used += request_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "embed_tokens_used": self.embed_tokens_used,
            "records_upserted": self.records_upserted,
            "storage_bytes_used": self.storage_bytes_used,
            "rerank_requests_used": self.rerank_requests_used,
            "write_units_used": self.write_units_used,
            "retry_requests": self.retry_requests,
            "per_language": dict(self.per_language),
        }

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> UsageLedger:
        data = json.loads(path.read_text())
        data.pop("_lock", None)
        return cls(**data)


# ── Budget enforcement ─────────────────────────────────────────────────────────


class BudgetGuard:
    """Enforce Starter limits before every API call. Thread-safe."""

    def __init__(
        self,
        budget: StarterBudget | None = None,
        ledger: UsageLedger | None = None,
        ledger_path: Path | None = None,
    ) -> None:
        self.budget = budget or StarterBudget()
        self.ledger = ledger or UsageLedger()
        self._ledger_path = ledger_path

    # ── Write-path guards ──────────────────────────────────────────────────────

    def refuse_full_corpus(self) -> None:
        """Unconditionally block full-corpus mode on Starter. No bypass."""
        if self.budget.is_starter():
            raise StarterFullModeError(
                "Full-corpus ingestion is permanently blocked on Pinecone Starter plan. "
                "PINECONE_PLAN=starter is active. "
                "No flag or environment variable can override this safety gate."
            )

    def check_upsert(
        self,
        language: str,
        record_count: int,
        token_count: int,
        byte_count: int | None = None,
    ) -> None:
        """Raise BudgetExceededError if the proposed upsert would exceed any limit.

        Must be called BEFORE the Pinecone API call. Never bypass for retries.
        """
        if byte_count is None:
            byte_count = record_count * self.budget.bytes_per_record_estimate

        new_records = self.ledger.records_upserted + record_count
        new_tokens = self.ledger.embed_tokens_used + token_count
        new_storage = self.ledger.storage_bytes_used + byte_count

        if new_records > self.budget.max_records:
            raise BudgetExceededError(
                f"Record budget exceeded: {new_records} > {self.budget.max_records} max."
            )
        if new_tokens > self.budget.max_embed_tokens:
            raise BudgetExceededError(
                f"Embedding token budget exceeded: {new_tokens} > {self.budget.max_embed_tokens} max."
            )
        if new_storage > self.budget.max_storage_bytes:
            gb = self.budget.max_storage_bytes / (1024**3)
            raise BudgetExceededError(
                f"Storage budget exceeded: projected {new_storage / (1024**3):.2f} GB > {gb:.2f} GB max."
            )

    def check_rerank(self, request_count: int = 1) -> None:
        """Raise BudgetExceededError if rerank quota would be exceeded."""
        new_total = self.ledger.rerank_requests_used + request_count
        if new_total > self.budget.max_rerank_requests:
            raise BudgetExceededError(
                f"Rerank quota exceeded: {new_total} > {self.budget.max_rerank_requests} max/month."
            )

    # ── Commit (call AFTER successful Pinecone acknowledgement) ────────────────

    def commit_upsert(
        self,
        language: str,
        record_count: int,
        token_count: int,
        byte_count: int | None = None,
        is_retry: bool = False,
    ) -> None:
        if byte_count is None:
            byte_count = record_count * self.budget.bytes_per_record_estimate
        self.ledger.account_upsert(language, record_count, token_count, byte_count, is_retry)
        if self._ledger_path:
            self.ledger.save(self._ledger_path)

    def commit_rerank(self, request_count: int = 1) -> None:
        self.ledger.account_rerank(request_count)
        if self._ledger_path:
            self.ledger.save(self._ledger_path)

    # ── Reporting ──────────────────────────────────────────────────────────────

    def usage_report(self) -> dict[str, Any]:
        b = self.budget
        lg = self.ledger
        embed_pct = (
            round(100 * lg.embed_tokens_used / b.max_embed_tokens, 1) if b.max_embed_tokens else 0
        )
        rec_pct = round(100 * lg.records_upserted / b.max_records, 1) if b.max_records else 0
        stor_pct = (
            round(100 * lg.storage_bytes_used / b.max_storage_bytes, 1)
            if b.max_storage_bytes
            else 0
        )
        rer_pct = (
            round(100 * lg.rerank_requests_used / b.max_rerank_requests, 1)
            if b.max_rerank_requests
            else 0
        )
        return {
            "plan": b.plan,
            "embed_tokens": {
                "used": lg.embed_tokens_used,
                "limit": b.max_embed_tokens,
                "pct": embed_pct,
                "pass": lg.embed_tokens_used <= b.max_embed_tokens,
            },
            "records": {
                "used": lg.records_upserted,
                "limit": b.max_records,
                "pct": rec_pct,
                "pass": lg.records_upserted <= b.max_records,
            },
            "storage_bytes": {
                "used": lg.storage_bytes_used,
                "limit": b.max_storage_bytes,
                "pct": stor_pct,
                "pass": lg.storage_bytes_used <= b.max_storage_bytes,
            },
            "rerank_requests": {
                "used": lg.rerank_requests_used,
                "limit": b.max_rerank_requests,
                "pct": rer_pct,
                "pass": lg.rerank_requests_used <= b.max_rerank_requests,
            },
            "retry_requests": lg.retry_requests,
            "per_language": lg.per_language,
        }


# ── Module-level helpers ───────────────────────────────────────────────────────


def get_plan() -> str:
    return os.environ.get("PINECONE_PLAN", PLAN_STARTER).lower()


def is_starter() -> bool:
    return get_plan() == PLAN_STARTER


def make_default_guard(ledger_path: Path | None = None) -> BudgetGuard:
    """Return a BudgetGuard using environment-configured plan."""
    plan = get_plan()
    budget = StarterBudget(plan=plan)
    ledger = (
        UsageLedger.load(ledger_path) if ledger_path and ledger_path.exists() else UsageLedger()
    )
    return BudgetGuard(budget=budget, ledger=ledger, ledger_path=ledger_path)
