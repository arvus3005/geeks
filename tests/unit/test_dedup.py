import tempfile
from pathlib import Path

from hhgoa_rag.ingestion.dedup import ContentDeduplicator


def test_dedup_not_duplicate_first():
    with tempfile.TemporaryDirectory() as d:
        dedup = ContentDeduplicator(Path(d) / "test.db")
        assert not dedup.is_duplicate("abc123")
        dedup.close()


def test_dedup_is_duplicate_after_mark():
    with tempfile.TemporaryDirectory() as d:
        dedup = ContentDeduplicator(Path(d) / "test.db")
        dedup.mark_seen("abc123", "passage001")
        assert dedup.is_duplicate("abc123")
        dedup.close()


def test_dedup_persists():
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "test.db"
        dedup = ContentDeduplicator(db)
        dedup.mark_seen("xyz", "p1")
        dedup.close()
        dedup2 = ContentDeduplicator(db)
        assert dedup2.is_duplicate("xyz")
        dedup2.close()


def test_dedup_batch_flush():
    with tempfile.TemporaryDirectory() as d:
        dedup = ContentDeduplicator(Path(d) / "test.db", batch_size=3)
        for i in range(6):
            dedup.mark_seen(f"hash{i}", f"pid{i}")
        # After 6 marks with batch_size=3, two flushes should have occurred
        dedup.close()
        dedup2 = ContentDeduplicator(Path(d) / "test.db")
        for i in range(6):
            assert dedup2.is_duplicate(f"hash{i}")
        dedup2.close()


def test_dedup_batch_not_committed_until_flush():
    with tempfile.TemporaryDirectory() as d:
        dedup = ContentDeduplicator(Path(d) / "test.db", batch_size=100)
        dedup.mark_seen("pending", "p1")
        # Pending — not yet committed, but close() flushes
        dedup.close()
        dedup2 = ContentDeduplicator(Path(d) / "test.db")
        assert dedup2.is_duplicate("pending")
        dedup2.close()
