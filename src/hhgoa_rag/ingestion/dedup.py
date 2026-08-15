import sqlite3
from pathlib import Path


class ContentDeduplicator:
    def __init__(self, db_path: Path, batch_size: int = 500):
        self.db_path = db_path
        self.batch_size = batch_size
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS seen_hashes "
            "(content_hash TEXT PRIMARY KEY, first_passage_id TEXT)"
        )
        self._conn.commit()
        self._pending: list[tuple[str, str]] = []

    def is_duplicate(self, content_hash: str) -> bool:
        # Check pending buffer first (not yet committed)
        if any(h == content_hash for h, _ in self._pending):
            return True
        cur = self._conn.execute("SELECT 1 FROM seen_hashes WHERE content_hash=?", (content_hash,))
        return cur.fetchone() is not None

    def mark_seen(self, content_hash: str, passage_id: str) -> None:
        self._pending.append((content_hash, passage_id))
        if len(self._pending) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if self._pending:
            self._conn.executemany("INSERT OR IGNORE INTO seen_hashes VALUES (?,?)", self._pending)
            self._conn.commit()
            self._pending.clear()

    def close(self) -> None:
        self.flush()
        self._conn.close()
