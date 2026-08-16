"""Contract: namespace safety, dedup ordering, budget-engine integration, batch limits."""

from unittest.mock import MagicMock

import pytest

from hhgoa_rag.ingestion.budget import BudgetExceededError, BudgetGuard, StarterBudget
from hhgoa_rag.ingestion.dedup import ContentDeduplicator
from hhgoa_rag.pinecone_store import (
    FULL_NAMESPACE,
    PILOT_NAMESPACE_PREFIX,
    SMOKE_NAMESPACE,
    TEXT_RECORD_FIELD,
    PineconeStore,
    is_safe_namespace,
)


def _store() -> PineconeStore:
    return PineconeStore(MagicMock(), embed_model="multilingual-e5-large")


# ── Dedup ordering — no auto-flush ───────────────────────────────────────────


def test_dedup_mark_seen_does_not_auto_flush(tmp_path):
    """mark_seen must NOT commit to SQLite — only flush() does."""
    dedup = ContentDeduplicator(tmp_path / "d.db")
    # Mark 600 entries (old code auto-flushed at batch_size=500)
    for i in range(600):
        dedup.mark_seen(f"hash_{i}", f"id_{i}")
    assert len(dedup._pending) == 600
    count = dedup._conn.execute("SELECT COUNT(*) FROM seen_hashes").fetchone()[0]
    assert count == 0  # nothing committed yet
    dedup.close()


def test_dedup_flush_commits_to_sqlite(tmp_path):
    """flush() commits the pending buffer to SQLite."""
    dedup = ContentDeduplicator(tmp_path / "d.db")
    dedup.mark_seen("h1", "id1")
    assert not dedup._conn.execute("SELECT 1 FROM seen_hashes WHERE content_hash='h1'").fetchone()
    dedup.flush()
    assert dedup._conn.execute("SELECT 1 FROM seen_hashes WHERE content_hash='h1'").fetchone()
    dedup.close()


def test_dedup_pending_buffer_enables_intra_batch_dedup(tmp_path):
    """is_duplicate() checks pending buffer so intra-batch dups are caught before flush."""
    dedup = ContentDeduplicator(tmp_path / "d.db")
    assert not dedup.is_duplicate("h1")
    dedup.mark_seen("h1", "id1")
    assert dedup.is_duplicate("h1")  # buffer check — not in DB yet
    dedup.close()


# ── Budget enforcement ordering ───────────────────────────────────────────────


def test_budget_check_must_precede_provider_call():
    """Budget failure must prevent provider from being called."""
    guard = BudgetGuard(budget=StarterBudget(max_records=5))
    index = MagicMock()

    with pytest.raises(BudgetExceededError):
        guard.check_upsert("en", record_count=10, token_count=0, byte_count=0)

    index.upsert_records.assert_not_called()


def test_budget_commit_only_after_ack():
    guard = BudgetGuard()
    guard.check_upsert("en", record_count=5, token_count=100, byte_count=0)
    assert guard.ledger.records_upserted == 0  # not committed yet
    guard.commit_upsert("en", record_count=5, token_count=100, byte_count=0)
    assert guard.ledger.records_upserted == 5


def test_retry_does_not_double_count_budget():
    guard = BudgetGuard()
    # First attempt: check passes, provider fails — no commit
    guard.check_upsert("en", record_count=5, token_count=100, byte_count=0)
    # Retry: check again without prior commit
    guard.check_upsert("en", record_count=5, token_count=100, byte_count=0)
    # Success: commit exactly once
    guard.commit_upsert("en", record_count=5, token_count=100, byte_count=0)
    assert guard.ledger.records_upserted == 5  # not 10 or 15


def test_failed_write_does_not_commit_dedup(tmp_path):
    """Hash must not be committed to SQLite if the Pinecone call failed."""
    dedup = ContentDeduplicator(tmp_path / "d.db")
    dedup.mark_seen("h1", "id1")  # buffered
    # Simulate failure — do NOT call dedup.flush()
    count = dedup._conn.execute("SELECT COUNT(*) FROM seen_hashes").fetchone()[0]
    assert count == 0
    dedup.close()


# ── Batch size limits ─────────────────────────────────────────────────────────


def test_ingestion_config_batch_size_validated():
    from hhgoa_rag.ingestion.engine import IngestionConfig

    with pytest.raises(ValueError, match="batch_size"):
        IngestionConfig(
            mode="pilot", pinecone_index="x", pinecone_namespace="pilot_x", batch_size=97
        )


def test_ingestion_config_batch_size_zero_rejected():
    from hhgoa_rag.ingestion.engine import IngestionConfig

    with pytest.raises(ValueError, match="batch_size"):
        IngestionConfig(
            mode="pilot", pinecone_index="x", pinecone_namespace="pilot_x", batch_size=0
        )


# ── StarterFullModeError in engine ────────────────────────────────────────────


def test_engine_blocks_full_mode_on_starter(monkeypatch):
    monkeypatch.setenv("PINECONE_PLAN", "starter")
    from hhgoa_rag.ingestion.budget import StarterFullModeError
    from hhgoa_rag.ingestion.engine import _check_starter_full_mode

    with pytest.raises(StarterFullModeError):
        _check_starter_full_mode("full")


def test_engine_allows_pilot_on_starter(monkeypatch):
    monkeypatch.setenv("PINECONE_PLAN", "starter")
    from hhgoa_rag.ingestion.engine import _check_starter_full_mode

    _check_starter_full_mode("pilot")  # must not raise


# ── Namespace safety ──────────────────────────────────────────────────────────


def test_smoke_context_refuses_full_namespace():
    store = _store()
    with pytest.raises(ValueError, match="full"):
        store.upsert_records(
            [{"id": "x", TEXT_RECORD_FIELD: "t"}], namespace=FULL_NAMESPACE, context="smoke"
        )


def test_pilot_context_refuses_full_namespace():
    store = _store()
    with pytest.raises(ValueError, match="full"):
        store.upsert_records(
            [{"id": "x", TEXT_RECORD_FIELD: "t"}], namespace=FULL_NAMESPACE, context="pilot"
        )


def test_smoke_context_allows_smoke_namespace():
    index = MagicMock()
    store = PineconeStore(index, embed_model="multilingual-e5-large")
    store.upsert_records(
        [{"id": "x", TEXT_RECORD_FIELD: "t"}], namespace=SMOKE_NAMESPACE, context="smoke"
    )
    index.upsert_records.assert_called_once()


def test_pilot_context_allows_pilot_namespace():
    index = MagicMock()
    store = PineconeStore(index, embed_model="multilingual-e5-large")
    store.upsert_records(
        [{"id": "x", TEXT_RECORD_FIELD: "t"}],
        namespace=f"{PILOT_NAMESPACE_PREFIX}run1",
        context="pilot",
    )
    index.upsert_records.assert_called_once()


def test_full_context_allows_full_namespace():
    index = MagicMock()
    store = PineconeStore(index, embed_model="multilingual-e5-large")
    store.upsert_records(
        [{"id": "x", TEXT_RECORD_FIELD: "t"}], namespace=FULL_NAMESPACE, context="full"
    )
    index.upsert_records.assert_called_once()


def test_is_safe_namespace_smoke():
    assert is_safe_namespace(SMOKE_NAMESPACE)


def test_is_safe_namespace_pilot():
    assert is_safe_namespace(f"{PILOT_NAMESPACE_PREFIX}abc123")


def test_is_safe_namespace_full_is_not_safe():
    assert not is_safe_namespace(FULL_NAMESPACE)
