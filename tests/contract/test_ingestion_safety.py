"""Contract: ingestion safety gates for full and pilot modes."""

import pytest


def test_full_mode_refuses_without_flag():
    """Full mode must refuse if --confirm-full-ingest is absent."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "scripts/ingest_all.py", "--mode", "full"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "confirm-full-ingest" in result.stderr.lower() or "DO NOT RUN" in result.stderr


def test_ingest_shard_full_refuses_without_flag():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "scripts/ingest_shard.py",
            "--config-lang",
            "bn",
            "--split",
            "train",
            "--mode",
            "full",
            "--collection",
            "msmarco_xi_passages_pilot_v001",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "confirm-full-ingest" in result.stderr.lower() or "DO NOT RUN" in result.stderr


def test_resume_full_refuses_without_flag(tmp_path):
    """Resume of a full-mode checkpoint must refuse without --confirm-full-ingest."""
    import json

    ckpt_path = tmp_path / "ckpt.json"
    ckpt_path.write_text(
        json.dumps(
            {
                "run_id": "test",
                "dataset_repo": "ai4bharat/MSMARCO-XI",
                "dataset_revision": None,
                "config_language": "bn",
                "split": "train",
                "source_shard": 0,
                "chunk_strategy": "passage_native",
                "chunk_strategy_version": "v1",
                "dense_model_id": "fake",
                "sparse_model_name": "Qdrant/bm25",
                "physical_collection": "msmarco_xi_passages_v001",
                "schema_fingerprint": "abc",
                "last_acknowledged_row": 0,
                "cumulative_source_rows": 0,
                "cumulative_valid_occurrences": 0,
                "cumulative_duplicate_occurrences": 0,
                "cumulative_rejected_occurrences": 0,
                "cumulative_chunks_emitted": 0,
                "cumulative_qdrant_points": 0,
                "started_at": "2026-08-15T00:00:00+00:00",
                "updated_at": "2026-08-15T00:00:00+00:00",
                "mode": "full",
                "status": "running",
                "warnings": [],
            }
        )
    )
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "scripts/resume_ingest.py", "--checkpoint", str(ckpt_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "confirm-full-ingest" in result.stderr.lower() or "DO NOT RUN" in result.stderr


def test_alias_refused_for_full_ingest_target():
    """Alias switch must refuse production alias for smoke/pilot collections."""
    from unittest.mock import MagicMock

    from hhgoa_rag.qdrant_lifecycle import switch_alias

    client = MagicMock()
    with pytest.raises(ValueError, match="smoke/pilot"):
        switch_alias(client, "msmarco_xi_passages_smoke_v001", smoke_ok=False)
