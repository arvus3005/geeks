"""Tests for ContentDeduplicator with crash-safe no-auto-flush semantics."""

import tempfile
from pathlib import Path

from hhgoa_rag.ingestion.dedup import ContentDeduplicator


def test_dedup_not_duplicate_first():
    with tempfile.TemporaryDirectory() as d:
        dedup = ContentDeduplicator(Path(d) / "test.db")
        assert not dedup.is_duplicate("abc123")
        dedup.close()


def test_dedup_is_duplicate_after_mark_in_buffer():
    """mark_seen adds to pending buffer; is_duplicate detects it immediately."""
    with tempfile.TemporaryDirectory() as d:
        dedup = ContentDeduplicator(Path(d) / "test.db")
        dedup.mark_seen("abc123", "passage001")
        assert dedup.is_duplicate("abc123")
        dedup.close()


def test_dedup_close_discards_pending():
    """close() MUST NOT flush pending hashes.

    Unacknowledged pending entries are discarded on close so that they can be
    safely replayed on the next run.  Only flush() (called after Pinecone ack)
    commits hashes to SQLite.
    """
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "test.db"
        dedup = ContentDeduplicator(db)
        dedup.mark_seen("xyz", "p1")
        dedup.close()
        # After close with pending-but-not-flushed, DB must remain empty
        dedup2 = ContentDeduplicator(db)
        assert not dedup2.is_duplicate("xyz")
        dedup2.close()


def test_dedup_no_auto_flush_many_marks():
    """mark_seen does NOT auto-flush to SQLite — only explicit flush() does."""
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "test.db"
        dedup = ContentDeduplicator(db)
        for i in range(600):
            dedup.mark_seen(f"hash{i}", f"pid{i}")
        # All 600 are in the pending buffer
        assert len(dedup._pending) == 600
        # None committed to DB yet
        count = dedup._conn.execute("SELECT COUNT(*) FROM seen_hashes").fetchone()[0]
        assert count == 0
        dedup.close()
        # close() discards pending; DB still empty
        dedup2 = ContentDeduplicator(db)
        count2 = dedup2._conn.execute("SELECT COUNT(*) FROM seen_hashes").fetchone()[0]
        assert count2 == 0
        dedup2.close()


def test_dedup_explicit_flush_commits():
    """flush() commits buffered hashes without closing the connection."""
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "test.db"
        dedup = ContentDeduplicator(db)
        dedup.mark_seen("h1", "id1")
        assert not dedup._conn.execute(
            "SELECT 1 FROM seen_hashes WHERE content_hash='h1'"
        ).fetchone()
        dedup.flush()
        assert dedup._conn.execute("SELECT 1 FROM seen_hashes WHERE content_hash='h1'").fetchone()
        assert len(dedup._pending) == 0
        dedup.close()


def test_dedup_not_committed_without_flush():
    """Without flush(), hashes remain uncommitted in the pending buffer."""
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "test.db"
        dedup = ContentDeduplicator(db)
        dedup.mark_seen("pending_hash", "p1")
        count = dedup._conn.execute("SELECT COUNT(*) FROM seen_hashes").fetchone()[0]
        assert count == 0  # not committed
        dedup.close()


def test_dedup_is_duplicate_after_db_commit():
    """is_duplicate() returns True for hashes committed to DB (not just buffer)."""
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "test.db"
        dedup = ContentDeduplicator(db)
        dedup.mark_seen("committed", "id1")
        dedup.flush()  # commit to DB
        dedup._pending.clear()  # clear buffer to force DB check
        assert dedup.is_duplicate("committed")  # should still be found in DB
        dedup.close()
