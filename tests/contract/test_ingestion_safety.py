"""Contract: ingestion safety gates for create-index, full and pilot modes."""

import subprocess
import sys


def _run(cmd: list[str], env: dict | None = None) -> subprocess.CompletedProcess:
    import os

    run_env = os.environ.copy()
    for k in [
        "PINECONE_API_KEY",
        "CONFIRM_PINECONE_CREATE",
        "CONFIRM_PINECONE_WRITE",
        "CONFIRM_FULL_INGEST",
        "PINECONE_PLAN",
    ]:
        run_env.pop(k, None)
    if env:
        run_env.update(env)
    return subprocess.run(cmd, capture_output=True, text=True, env=run_env)


def _make_pilot_ckpt(path, mode="pilot", namespace="pilot_test"):
    import json

    path.write_text(
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
                "pinecone_namespace": namespace,
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
                "mode": mode,
                "num_workers": 1,
                "status": "running",
                "warnings": [],
            }
        )
    )


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


def test_create_index_execute_alone_exits_2():
    """--execute alone (no env var) must exit 2 (fail-closed, not silently dry-run)."""
    result = _run(
        [
            sys.executable,
            "scripts/create_pinecone_index.py",
            "--pinecone-index",
            "test",
            "--execute",
        ]
    )
    assert result.returncode == 2
    assert "CONFIRM_PINECONE_CREATE" in result.stderr


def test_create_index_env_alone_is_dry_run():
    """CONFIRM_PINECONE_CREATE=1 alone (no --execute) must still be dry-run."""
    result = _run(
        [sys.executable, "scripts/create_pinecone_index.py", "--pinecone-index", "test"],
        env={"CONFIRM_PINECONE_CREATE": "1"},
    )
    assert result.returncode == 0
    assert "dry" in result.stderr.lower() or "--execute" in result.stderr


def test_create_index_requires_both_guards():
    """Both --execute and CONFIRM_PINECONE_CREATE=1 required to attempt creation."""
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
    assert result.returncode != 0
    assert "PINECONE_API_KEY" in result.stderr or "api_key" in result.stderr.lower()


# ── ingest_all.py ─────────────────────────────────────────────────────────────


def test_full_mode_on_starter_blocked_before_confirm_flag():
    """Full mode on Starter plan is blocked regardless of --confirm-full-ingest."""
    result = _run(
        [
            sys.executable,
            "scripts/ingest_all.py",
            "--mode",
            "full",
            "--confirm-full-ingest",
        ],
        env={
            "PINECONE_PLAN": "starter",
            "CONFIRM_FULL_INGEST": "YES_I_APPROVE_FULL_CORPUS",
            "CONFIRM_PINECONE_WRITE": "1",
        },
    )
    assert result.returncode != 0
    # Must fail with Starter plan error, not with API key error
    assert "starter" in result.stderr.lower() or "permanently blocked" in result.stderr.lower()
    assert "PINECONE_API_KEY" not in result.stderr


def test_full_mode_refuses_without_cli_flag():
    """Full mode without --confirm-full-ingest must be refused (non-Starter)."""
    result = _run(
        [sys.executable, "scripts/ingest_all.py", "--mode", "full"],
        env={"PINECONE_PLAN": "standard"},
    )
    assert result.returncode != 0
    assert "confirm-full-ingest" in result.stderr.lower() or "DO NOT RUN" in result.stderr


def test_full_mode_refuses_without_env_var():
    """Full mode without env var must be refused (non-Starter)."""
    result = _run(
        [sys.executable, "scripts/ingest_all.py", "--mode", "full", "--confirm-full-ingest"],
        env={"PINECONE_PLAN": "standard"},
    )
    assert result.returncode != 0
    assert "CONFIRM_FULL_INGEST" in result.stderr or "DO NOT RUN" in result.stderr


def test_ingest_all_smoke_no_execute_is_dry_run():
    """Smoke mode without --execute exits 0 with dry-run message (no API key needed)."""
    result = _run([sys.executable, "scripts/ingest_all.py", "--mode", "smoke"])
    assert result.returncode == 0
    assert "dry" in result.stderr.lower() or "dry_run" in result.stdout


def test_ingest_all_pilot_blocked_redirects_to_ingest_prepared():
    """Pilot mode via ingest_all.py must be blocked and redirect to ingest_prepared.py."""
    result = _run([sys.executable, "scripts/ingest_all.py", "--mode", "pilot"])
    assert result.returncode == 1
    assert "ingest_prepared.py" in result.stderr


def test_ingest_all_batch_size_over_96_rejected():
    """--batch-size > 96 must be rejected (pilot also blocked, so exit != 0)."""
    result = _run(
        [sys.executable, "scripts/ingest_all.py", "--mode", "pilot", "--batch-size", "200"]
    )
    assert result.returncode != 0


# ── ingest_shard.py ───────────────────────────────────────────────────────────


def test_ingest_shard_full_on_starter_blocked():
    """Full mode on Starter blocked before confirm flag."""
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
        ],
        env={
            "PINECONE_PLAN": "starter",
            "CONFIRM_FULL_INGEST": "YES_I_APPROVE_FULL_CORPUS",
            "CONFIRM_PINECONE_WRITE": "1",
        },
    )
    assert result.returncode != 0
    assert "starter" in result.stderr.lower() or "permanently blocked" in result.stderr.lower()
    assert "PINECONE_API_KEY" not in result.stderr


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
        ],
        env={"PINECONE_PLAN": "standard"},
    )
    assert result.returncode != 0
    assert "confirm-full-ingest" in result.stderr.lower() or "DO NOT RUN" in result.stderr


def test_ingest_shard_pilot_blocked_redirects_to_ingest_prepared():
    """Pilot mode via ingest_shard.py must be blocked and redirect to ingest_prepared.py."""
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
    assert result.returncode == 1
    assert "ingest_prepared.py" in result.stderr


def test_ingest_shard_batch_size_over_96_rejected():
    """Pilot mode is blocked before batch-size check; exit must be non-zero."""
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
            "--batch-size",
            "200",
        ]
    )
    assert result.returncode != 0


# ── resume_ingest.py ──────────────────────────────────────────────────────────


def test_resume_full_on_starter_blocked(tmp_path):
    """Full mode checkpoint on Starter plan must be rejected immediately."""
    ckpt_path = tmp_path / "ckpt.json"
    _make_pilot_ckpt(ckpt_path, mode="full", namespace="full")
    result = _run(
        [sys.executable, "scripts/resume_ingest.py", "--checkpoint", str(ckpt_path)],
        env={"PINECONE_PLAN": "starter"},
    )
    assert result.returncode != 0
    # Must fail with Starter plan error before API key check
    assert "starter" in result.stderr.lower() or "permanently blocked" in result.stderr.lower()
    assert "PINECONE_API_KEY" not in result.stderr


def test_resume_full_refuses_without_flag(tmp_path):
    """Full mode checkpoint without --confirm-full-ingest must be refused (non-Starter)."""
    ckpt_path = tmp_path / "ckpt.json"
    _make_pilot_ckpt(ckpt_path, mode="full", namespace="full")
    result = _run(
        [sys.executable, "scripts/resume_ingest.py", "--checkpoint", str(ckpt_path)],
        env={"PINECONE_PLAN": "standard"},
    )
    assert result.returncode != 0
    assert "confirm-full-ingest" in result.stderr.lower() or "DO NOT RUN" in result.stderr


def test_resume_pilot_checkpoint_blocked(tmp_path):
    """Pilot checkpoint via resume_ingest.py must be blocked and redirect to ingest_prepared.py."""
    ckpt_path = tmp_path / "ckpt.json"
    _make_pilot_ckpt(ckpt_path)
    result = _run([sys.executable, "scripts/resume_ingest.py", "--checkpoint", str(ckpt_path)])
    assert result.returncode == 1
    assert "ingest_prepared.py" in result.stderr


def test_resume_old_checkpoint_without_num_workers_fails_closed(tmp_path):
    """Old checkpoint missing num_workers must be rejected — fail closed."""
    import json

    ckpt_path = tmp_path / "old.json"
    ckpt_path.write_text(
        json.dumps(
            {
                "run_id": "old",
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
                # num_workers intentionally absent
                "status": "running",
                "warnings": [],
            }
        )
    )
    result = _run([sys.executable, "scripts/resume_ingest.py", "--checkpoint", str(ckpt_path)])
    assert result.returncode != 0
    assert "num_workers" in result.stderr.lower() or "missing" in result.stderr.lower()
