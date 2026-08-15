import pytest
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
