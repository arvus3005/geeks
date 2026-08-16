import sqlite3
from pathlib import Path


class ContentDeduplicator:
    """SQLite-backed deduplication tracker.

    Crash-consistency contract:
      - mark_seen() adds to an in-memory buffer only; never auto-flushes.
      - flush() is the only path that commits hashes to SQLite.
      - Callers MUST call flush() only AFTER the associated Pinecone batch is
        successfully acknowledged, so a crash before ack leaves the DB unchanged
        and the engine safely re-processes the records on resume.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS seen_hashes "
            "(content_hash TEXT PRIMARY KEY, first_passage_id TEXT)"
        )
        self._conn.commit()
        self._pending: list[tuple[str, str]] = []

    def is_duplicate(self, content_hash: str) -> bool:
        """Return True if content_hash is committed to DB or reserved in pending buffer."""
        if any(h == content_hash for h, _ in self._pending):
            return True
        cur = self._conn.execute("SELECT 1 FROM seen_hashes WHERE content_hash=?", (content_hash,))
        return cur.fetchone() is not None

    def mark_seen(self, content_hash: str, passage_id: str) -> None:
        """Reserve hash in the in-memory buffer.

        Does NOT write to SQLite — caller must call flush() after Pinecone ack.
        """
        self._pending.append((content_hash, passage_id))

    def flush(self) -> None:
        """Commit buffered hashes to SQLite.  Call only after Pinecone acknowledges the batch."""
        if self._pending:
            self._conn.executemany("INSERT OR IGNORE INTO seen_hashes VALUES (?,?)", self._pending)
            self._conn.commit()
            self._pending.clear()

    def close(self) -> None:
        """Close the connection.  Discards any unacknowledged pending reservations.

        IMPORTANT: close() MUST NOT flush pending hashes.  Pending hashes are
        only committed by an explicit flush() call after Pinecone acknowledges the
        batch.  Calling close() with pending entries means the caller never
        received acknowledgement — the pending reservations are intentionally
        discarded so the records can be safely replayed on the next run.
        """
        self._pending.clear()
        self._conn.close()
