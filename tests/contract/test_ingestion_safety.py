"""Contract: ingestion safety gates for full and pilot modes."""



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
            "--config-lang", "bn",
            "--split", "train",
            "--mode", "full",
            "--pinecone-index", "msmarco-xi",
            "--pinecone-namespace", "full",
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
        json.dumps({
            "run_id": "test",
            "dataset_repo": "ai4bharat/MSMARCO-XI",
            "dataset_revision": None,
            "config_language": "bn",
            "split": "train",
            "source_shard": 0,
            "chunk_strategy": "passage_native",
            "chunk_strategy_version": "v1",
            "embed_model": "multilingual-e5-large",
            "pinecone_index": "msmarco-xi",
            "pinecone_namespace": "full",
            "schema_fingerprint": "abc",
            "last_acknowledged_row": 0,
            "cumulative_source_rows": 0,
            "cumulative_valid_occurrences": 0,
            "cumulative_duplicate_occurrences": 0,
            "cumulative_rejected_occurrences": 0,
            "cumulative_chunks_emitted": 0,
            "cumulative_indexed_points": 0,
            "started_at": "2026-08-15T00:00:00+00:00",
            "updated_at": "2026-08-15T00:00:00+00:00",
            "mode": "full",
            "status": "running",
            "warnings": [],
        })
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
