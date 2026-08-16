"""Crash-consistent checkpoint state for resumable ingestion.

Checkpoints are written atomically (temp file + rename) so a crash mid-write
never leaves a partially-valid checkpoint.  A resume refuses if any key schema
field (dataset, model, index, chunk strategy, num_workers) differs from the
checkpoint.

Backward-incompatibility rule: checkpoints that predate the num_workers field
are rejected as incompatible.  Fail closed rather than silently accept unknown
sharding assumptions.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

_REQUIRED_FIELDS = {
    "run_id",
    "dataset_repo",
    "config_language",
    "split",
    "source_shard",
    "chunk_strategy",
    "chunk_strategy_version",
    "embed_model",
    "pinecone_index",
    "pinecone_namespace",
    "schema_fingerprint",
    "num_workers",
}


@dataclass
class IngestCheckpoint:
    run_id: str
    dataset_repo: str
    dataset_revision: str | None
    config_language: str
    split: str
    source_shard: int
    chunk_strategy: str
    chunk_strategy_version: str
    embed_model: str
    pinecone_index: str
    pinecone_namespace: str
    schema_fingerprint: str
    last_acknowledged_row: int
    cumulative_source_rows: int
    cumulative_valid_occurrences: int
    cumulative_duplicate_occurrences: int
    cumulative_rejected_occurrences: int
    cumulative_chunks_emitted: int
    cumulative_indexed_points: int
    started_at: str
    updated_at: str
    mode: str
    num_workers: int  # must be stored and validated; no default — absence fails closed
    status: str = "running"
    warnings: list[str] = field(default_factory=list)

    def is_compatible(self, other: IngestCheckpoint) -> tuple[bool, list[str]]:
        """Return (compatible, list_of_incompatibilities)."""
        mismatches = []
        for attr in (
            "dataset_repo",
            "dataset_revision",
            "config_language",
            "split",
            "source_shard",
            "chunk_strategy",
            "chunk_strategy_version",
            "embed_model",
            "pinecone_index",
            "pinecone_namespace",
            "schema_fingerprint",
            "num_workers",
        ):
            if getattr(self, attr) != getattr(other, attr):
                mismatches.append(
                    f"{attr}: checkpoint={getattr(self, attr)!r} != new={getattr(other, attr)!r}"
                )
        return len(mismatches) == 0, mismatches

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(asdict(self), f, indent=2)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> IngestCheckpoint:
        with open(path) as f:
            data = json.load(f)
        missing = _REQUIRED_FIELDS - set(data.keys())
        if missing:
            raise RuntimeError(
                f"Checkpoint {path} is missing required fields: {sorted(missing)}. "
                "This checkpoint is from an older format and cannot be safely resumed. "
                "Start a new ingestion run."
            )
        return cls(**data)

    @classmethod
    def checkpoint_path(
        cls, checkpoint_dir: Path, run_id: str, config_language: str, split: str, shard: int
    ) -> Path:
        return checkpoint_dir / f"{run_id}_{config_language}_{split}_shard{shard:04d}.json"


def make_schema_fingerprint(
    pinecone_index: str, pinecone_namespace: str, embed_model: str, chunk_strategy_version: str
) -> str:
    import hashlib

    s = f"{pinecone_index}|ns={pinecone_namespace}|embed={embed_model}|chunk={chunk_strategy_version}"
    return hashlib.sha256(s.encode()).hexdigest()[:16]
