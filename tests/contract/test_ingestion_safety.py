"""Contract: ingestion safety gates for create-index, full and pilot modes."""

import subprocess
import sys


def _run(cmd: list[str], env: dict | None = None) -> subprocess.CompletedProcess:
    import os

    run_env = os.environ.copy()
    # Strip any real credentials and confirmation vars from the test environment
    for k in [
        "PINECONE_API_KEY",
        "CONFIRM_PINECONE_CREATE",
        "CONFIRM_PINECONE_WRITE",
        "CONFIRM_FULL_INGEST",
    ]:
        run_env.pop(k, None)
    if env:
        run_env.update(env)
    return subprocess.run(cmd, capture_output=True, text=True, env=run_env)


# ── create_pinecone_index.py ──────────────────────────────────────────────────


def test_create_index_no_args_is_dry_run():
    """Without --execute the script prints a plan and exits 0."""
    result = _run([sys.executable, "scripts/create_pinecone_index.py", "--pinecone-index", "test"])
    assert result.returncode == 0
    assert (
        "dry" in result.stderr.lower()
        or "dry_run" in result.stdout.lower()
        or "dry-run" in result.stdout.lower()
    )


def test_create_index_execute_alone_is_dry_run():
    """--execute alone (no env var) must still be dry-run."""
    result = _run(
        [
            sys.executable,
            "scripts/create_pinecone_index.py",
            "--pinecone-index",
            "test",
            "--execute",
        ]
    )
    assert result.returncode == 0
    assert "dry" in result.stderr.lower() or "CONFIRM_PINECONE_CREATE" in result.stderr


def test_create_index_env_alone_is_dry_run():
    """CONFIRM_PINECONE_CREATE=1 alone (no --execute) must still be dry-run."""
    result = _run(
        [sys.executable, "scripts/create_pinecone_index.py", "--pinecone-index", "test"],
        env={"CONFIRM_PINECONE_CREATE": "1"},
    )
    assert result.returncode == 0
    assert "dry" in result.stderr.lower() or "--execute" in result.stderr


def test_create_index_requires_both_guards():
    """Both --execute and CONFIRM_PINECONE_CREATE=1 are required to attempt creation."""
    # Without API key it fails at the API-key check, not the guard check —
    # which confirms both guards passed and the script correctly reached the write path.
    result = _run(
        [
            sys.executable,
            "scripts/create_pinecone_index.py",
            "--pinecone-index",
            "test",
            "--execute",
        ],
        env={"CONFIRM_PINECONE_CREATE": "1"},
    )
    # Should fail at API key check (not dry-run exit)
    assert result.returncode != 0
    assert "PINECONE_API_KEY" in result.stderr or "api_key" in result.stderr.lower()


# ── ingest_all.py ─────────────────────────────────────────────────────────────


def test_full_mode_refuses_without_cli_flag():
    result = _run([sys.executable, "scripts/ingest_all.py", "--mode", "full"])
    assert result.returncode != 0
    assert "confirm-full-ingest" in result.stderr.lower() or "DO NOT RUN" in result.stderr


def test_full_mode_refuses_without_env_var():
    result = _run(
        [sys.executable, "scripts/ingest_all.py", "--mode", "full", "--confirm-full-ingest"]
    )
    assert result.returncode != 0
    assert "CONFIRM_FULL_INGEST" in result.stderr or "DO NOT RUN" in result.stderr


def test_ingest_all_smoke_no_execute_is_dry_run():
    """Smoke mode without --execute exits 0 with dry-run message (no API key needed)."""
    result = _run([sys.executable, "scripts/ingest_all.py", "--mode", "smoke"])
    assert result.returncode == 0
    assert "dry" in result.stderr.lower() or "dry_run" in result.stdout


def test_ingest_all_pilot_no_execute_is_dry_run():
    result = _run([sys.executable, "scripts/ingest_all.py", "--mode", "pilot"])
    assert result.returncode == 0
    assert "dry" in result.stderr.lower() or "dry_run" in result.stdout


# ── ingest_shard.py ───────────────────────────────────────────────────────────


def test_ingest_shard_full_refuses_without_flag():
    result = _run(
        [
            sys.executable,
            "scripts/ingest_shard.py",
            "--config-lang",
            "bn",
            "--split",
            "train",
            "--mode",
            "full",
            "--pinecone-index",
            "msmarco-xi",
            "--pinecone-namespace",
            "full",
        ]
    )
    assert result.returncode != 0
    assert "confirm-full-ingest" in result.stderr.lower() or "DO NOT RUN" in result.stderr


def test_ingest_shard_no_execute_is_dry_run():
    result = _run(
        [
            sys.executable,
            "scripts/ingest_shard.py",
            "--config-lang",
            "bn",
            "--split",
            "train",
            "--mode",
            "pilot",
            "--pinecone-index",
            "msmarco-xi",
            "--pinecone-namespace",
            "pilot_test",
        ]
    )
    assert result.returncode == 0
    assert "dry" in result.stderr.lower()


# ── resume_ingest.py ──────────────────────────────────────────────────────────


def test_resume_full_refuses_without_flag(tmp_path):
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
            }
        )
    )
    result = _run([sys.executable, "scripts/resume_ingest.py", "--checkpoint", str(ckpt_path)])
    assert result.returncode != 0
    assert "confirm-full-ingest" in result.stderr.lower() or "DO NOT RUN" in result.stderr


def test_resume_no_execute_is_dry_run(tmp_path):
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
                "embed_model": "multilingual-e5-large",
                "pinecone_index": "msmarco-xi",
                "pinecone_namespace": "pilot_test",
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
                "mode": "pilot",
                "status": "running",
                "warnings": [],
            }
        )
    )
    result = _run([sys.executable, "scripts/resume_ingest.py", "--checkpoint", str(ckpt_path)])
    assert result.returncode == 0
    assert "dry" in result.stderr.lower()
