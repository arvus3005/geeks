import sqlite3
from pathlib import Path


class ContentDeduplicator:
    """SQLite-backed deduplicator for scalable on-disk dedup."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS seen_hashes "
            "(content_hash TEXT PRIMARY KEY, first_passage_id TEXT)"
        )
        self._conn.commit()

    def is_duplicate(self, content_hash: str) -> bool:
        cur = self._conn.execute(
            "SELECT 1 FROM seen_hashes WHERE content_hash=?", (content_hash,)
        )
        return cur.fetchone() is not None

    def mark_seen(self, content_hash: str, passage_id: str) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO seen_hashes VALUES (?,?)",
            (content_hash, passage_id),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
