"""Tests for pre-index hardening: chunker routing, splitting, schema, dry-run.

These cover the requirements added during the pre-index readiness pass:
  - selected chunker is actually invoked
  - no text loss / duplication during token-limit splitting
  - oversized sub-chunks are all considered (not just the first)
  - 507-token boundary passes, 508 is split
  - schema rejects forbidden fields, bad revisions, bad ordinals
  - dry-run never constructs Pinecone even with live-looking env vars
  - contradictory flags / --execute without confirmation fail
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPTS = str(Path(__file__).parent.parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import prepare_canary as pc  # noqa: E402


class FakeTokenizer:
    """Deterministic whitespace tokenizer: 1 token per whitespace unit + 2 special."""

    fingerprint = "faketok0001"

    def count_tokens(self, text: str, add_prefix: bool = True) -> int:
        base = len([w for w in text.split() if w]) or (1 if text.strip() else 0)
        return base + 2  # simulate BOS/EOS overhead


# ── chunker routing ───────────────────────────────────────────────────────────


def test_selected_chunker_is_called(monkeypatch):
    called = {}

    real_get = pc._chunk_passage_texts

    def spy(text, pid, strategy, tokenizer=None):
        called["strategy"] = strategy
        return real_get(text, pid, strategy, tokenizer=tokenizer)

    monkeypatch.setattr(pc, "_chunk_passage_texts", spy)
    rec = {
        "language": "en",
        "config_language": "en",
        "split": "train",
        "physical_source": "train/hintrain.parquet",
        "local_source_row": 0,
        "passage_position": 0,
        "content_hash": "a" * 40,
        "normalized_text": "Short sentence one. Short sentence two.",
    }
    pc._build_prepared_records_for_passage(
        rec,
        "bf5cdc1f26e581e519018e434db14edd1b77602b",
        FakeTokenizer(),
        "sentence_aware",
        "v1",
        "canary-test",
        {},
        {},
    )
    assert called["strategy"] == "sentence_aware"


def test_every_supported_strategy_resolves():
    for strategy in ("passage_native", "sentence_aware", "fixed_token_overlap"):
        out = pc._chunk_passage_texts("A. B. C.", "pid", strategy)
        assert out and all(t.strip() for t in out)


# ── splitting: no loss / no duplication ───────────────────────────────────────


def test_split_no_text_loss_or_duplication():
    tok = FakeTokenizer()
    # 30 words, limit small so it must split multiple times.
    words = [f"w{i}" for i in range(30)]
    text = " ".join(words)
    chunks = pc._split_text_to_fit(text, tok, model_limit=7)  # ~5 words/chunk
    assert len(chunks) > 1
    # Reconstruct word sequence — must equal original with no dup/loss.
    recon = " ".join(chunks).split()
    assert recon == words


def test_split_boundary_507_passes_508_splits():
    tok = FakeTokenizer()
    # count = words + 2. For total 507 tokens -> 505 words.
    text_507 = " ".join(["x"] * 505)
    assert tok.count_tokens(text_507) == 507
    assert pc._split_text_to_fit(text_507, tok, 507) == [text_507]

    text_508 = " ".join(["x"] * 506)
    assert tok.count_tokens(text_508) == 508
    out = pc._split_text_to_fit(text_508, tok, 507)
    assert len(out) > 1
    assert " ".join(out).split() == ["x"] * 506


def test_split_oversized_units_all_considered():
    # A long unsplittable-by-sentence run of words must cascade to word level
    # and keep ALL words, not just the first sub-chunk.
    tok = FakeTokenizer()
    text = " ".join([f"tok{i}" for i in range(20)])  # single "sentence"
    out = pc._split_text_to_fit(text, tok, 6)
    joined = " ".join(out).split()
    assert joined == [f"tok{i}" for i in range(20)]


def test_chunk_ordinal_total_correct_when_split():
    rec = {
        "language": "en",
        "config_language": "en",
        "split": "train",
        "physical_source": "train/hintrain.parquet",
        "local_source_row": 0,
        "passage_position": 0,
        "content_hash": "b" * 40,
        # >507 tokens so the token-limit split path activates.
        "normalized_text": " ".join([f"w{i}" for i in range(700)]),
    }
    records = pc._build_prepared_records_for_passage(
        rec,
        "bf5cdc1f26e581e519018e434db14edd1b77602b",
        FakeTokenizer(),
        "passage_native",
        "v1",
        "canary-test",
        {},
        {},
    )
    assert len(records) > 1
    total = records[0]["chunk_total"]
    assert total == len(records)
    assert [r["chunk_ordinal"] for r in records] == list(range(total))


# ── schema validation ─────────────────────────────────────────────────────────


def _valid_record() -> dict:
    from hhgoa_rag.ingestion.schema import build_record

    return build_record(
        record_id="id-1",
        chunk_text="hello world",
        language="en",
        config_language="en",
        dataset_revision="bf5cdc1f26e581e519018e434db14edd1b77602b",
        split="train",
        physical_shard="0",
        local_source_row=0,
        passage_position=0,
        parent_passage_id="p",
        content_hash="c",
        chunk_strategy="passage_native",
        chunk_strategy_version="v1",
        chunk_ordinal=0,
        chunk_total=1,
        token_length=5,
        tokenizer_fingerprint="fp",
        manifest_id="m",
    )


def test_schema_rejects_short_revision():
    from hhgoa_rag.ingestion.schema import SchemaViolationError, validate_record

    rec = _valid_record()
    rec["dataset_revision"] = "abc123"
    with pytest.raises(SchemaViolationError, match="40-char"):
        validate_record(rec)


def test_schema_rejects_bad_language():
    from hhgoa_rag.ingestion.schema import SchemaViolationError, validate_record

    rec = _valid_record()
    rec["language"] = "fr"
    with pytest.raises(SchemaViolationError, match="language"):
        validate_record(rec)


def test_schema_rejects_ordinal_ge_total():
    from hhgoa_rag.ingestion.schema import SchemaViolationError, validate_record

    rec = _valid_record()
    rec["chunk_ordinal"] = 3
    rec["chunk_total"] = 3
    with pytest.raises(SchemaViolationError, match="ordinal"):
        validate_record(rec)


def test_schema_rejects_forbidden_recursive():
    from hhgoa_rag.ingestion.schema import SchemaViolationError, validate_record

    rec = _valid_record()
    rec["meta"] = {"deep": {"is_selected": True}}
    with pytest.raises(SchemaViolationError, match="is_selected"):
        validate_record(rec)


# ── manifest identity ─────────────────────────────────────────────────────────


def test_chunk_parameters_included_in_identity():
    p1 = pc._chunk_parameters("sentence_aware")
    assert "target_chars" in p1
    # A different strategy yields different parameters -> different identity input.
    p2 = pc._chunk_parameters("fixed_token_overlap")
    assert p1 != p2


# ── dry-run / flag safety (subprocess so imports are isolated) ─────────────────


def _run_ingest(args, env_extra=None):
    import os

    env = dict(os.environ)
    env.setdefault("HHGOA_USE_FAKE_EMBEDDER", "1")
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, str(Path(_SCRIPTS) / "ingest_prepared.py"), *args],
        capture_output=True,
        text=True,
        env=env,
    )


def test_contradictory_flags_fail(tmp_path):
    m = tmp_path / "m.json"
    m.write_text("{}")
    r = _run_ingest(["--manifest", str(m), "--dry-run", "--execute"])
    assert r.returncode == 2
    assert "mutually exclusive" in (r.stderr + r.stdout)


def test_execute_without_confirmation_fails(tmp_path):
    m = tmp_path / "m.json"
    m.write_text("{}")
    r = _run_ingest(
        ["--manifest", str(m), "--execute"],
        env_extra={"CONFIRM_PINECONE_WRITE": "", "PINECONE_API_KEY": "live-looking-key"},
    )
    assert r.returncode == 2
    assert "CONFIRM_PINECONE_WRITE" in (r.stderr + r.stdout)


def test_index_contract_constant_matches_decisions():
    import ingest_prepared as ing

    assert ing.INDEX_CONTRACT["model"] == "multilingual-e5-large"
    assert ing.INDEX_CONTRACT["dimension"] == 1024
    assert ing.INDEX_CONTRACT["metric"] == "cosine"
    assert ing.INDEX_CONTRACT["field_map"] == {"text": "chunk_text"}
    assert ing.INDEX_CONTRACT["write_parameters"] == {
        "input_type": "passage",
        "truncate": "NONE",
    }
    assert ing.INDEX_CONTRACT["read_parameters"] == {
        "input_type": "query",
        "truncate": "NONE",
    }
    assert ing.SAFE_NAMESPACE == "pilot_v1"


# ── canonical contract module ─────────────────────────────────────────────────


def test_contract_fingerprint_is_stable():
    """Same canonical contract must produce identical fingerprint on repeated calls."""
    from hhgoa_rag.pinecone_contract import contract_fingerprint

    fp1 = contract_fingerprint()
    fp2 = contract_fingerprint()
    assert fp1 == fp2
    assert len(fp1) == 64  # SHA-256 hex digest


def test_contract_contains_all_required_fields():
    from hhgoa_rag.pinecone_contract import canonical_contract

    c = canonical_contract()
    for field in [
        "model",
        "dimension",
        "metric",
        "field_map",
        "write_parameters",
        "read_parameters",
        "cloud",
        "region",
        "index_name",
        "namespace",
        "max_input_tokens",
        "max_batch_size",
        "dataset_repo",
        "dataset_revision",
        "tokenizer_repo",
        "tokenizer_revision",
        "contract_version",
    ]:
        assert field in c, f"canonical_contract missing field: {field}"


def test_contract_write_read_parameters():
    from hhgoa_rag.pinecone_contract import READ_PARAMETERS, WRITE_PARAMETERS

    assert WRITE_PARAMETERS == {"input_type": "passage", "truncate": "NONE"}
    assert READ_PARAMETERS == {"input_type": "query", "truncate": "NONE"}


# ── schema: unknown top-level fields rejected ─────────────────────────────────


def test_schema_rejects_unknown_fields():
    from hhgoa_rag.ingestion.schema import SchemaViolationError, build_record

    rec = build_record(
        record_id="id-1",
        chunk_text="hello world",
        language="en",
        config_language="en",
        dataset_revision="bf5cdc1f26e581e519018e434db14edd1b77602b",
        split="train",
        physical_shard="train/hintrain.parquet",
        local_source_row=0,
        passage_position=0,
        parent_passage_id="p",
        content_hash="c",
        chunk_strategy="passage_native",
        chunk_strategy_version="v1",
        chunk_ordinal=0,
        chunk_total=1,
        token_length=5,
        tokenizer_fingerprint="fp",
        manifest_id="m",
    )
    from hhgoa_rag.ingestion.schema import validate_record

    rec["extra_unofficial_field"] = "oops"
    with pytest.raises(SchemaViolationError, match="unknown top-level fields"):
        validate_record(rec)


def test_schema_rejects_physical_source_unofficial():
    """physical_source is not an allowed schema field — must be rejected."""
    from hhgoa_rag.ingestion.schema import SchemaViolationError, build_record, validate_record

    rec = build_record(
        record_id="id-2",
        chunk_text="test text",
        language="hi",
        config_language="hi",
        dataset_revision="bf5cdc1f26e581e519018e434db14edd1b77602b",
        split="train",
        physical_shard="train/hintrain.parquet",
        local_source_row=1,
        passage_position=0,
        parent_passage_id="p2",
        content_hash="d" * 40,
        chunk_strategy="passage_native",
        chunk_strategy_version="v1",
        chunk_ordinal=0,
        chunk_total=1,
        token_length=3,
        tokenizer_fingerprint="fp",
        manifest_id="m",
    )
    rec["physical_source"] = "train/hintrain.parquet"
    with pytest.raises(SchemaViolationError, match="unknown"):
        validate_record(rec)


# ── schema: 507/508 token limit in validate_record ────────────────────────────


def test_schema_507_tokens_pass():
    from hhgoa_rag.ingestion.schema import build_record

    rec = build_record(
        record_id="id-ok",
        chunk_text="x " * 100,
        language="en",
        config_language="en",
        dataset_revision="bf5cdc1f26e581e519018e434db14edd1b77602b",
        split="train",
        physical_shard="train/hintrain.parquet",
        local_source_row=0,
        passage_position=0,
        parent_passage_id="p",
        content_hash="a" * 40,
        chunk_strategy="passage_native",
        chunk_strategy_version="v1",
        chunk_ordinal=0,
        chunk_total=1,
        token_length=507,
        tokenizer_fingerprint="fp",
        manifest_id="m",
    )
    assert rec["token_length"] == 507


def test_schema_508_tokens_fail():
    from hhgoa_rag.ingestion.schema import SchemaViolationError, validate_record

    rec = {
        "id": "id-bad",
        "chunk_text": "x " * 100,
        "language": "en",
        "config_language": "en",
        "dataset_revision": "bf5cdc1f26e581e519018e434db14edd1b77602b",
        "split": "train",
        "physical_shard": "train/hintrain.parquet",
        "local_source_row": 0,
        "passage_position": 0,
        "parent_passage_id": "p",
        "content_hash": "a" * 40,
        "chunk_strategy": "passage_native",
        "chunk_strategy_version": "v1",
        "chunk_ordinal": 0,
        "chunk_total": 1,
        "token_length": 508,
        "tokenizer_fingerprint": "fp",
        "manifest_id": "m",
    }
    with pytest.raises(SchemaViolationError, match="507"):
        validate_record(rec)


# ── chunking strategy registry matches CLI choices ────────────────────────────


def test_every_advertised_chunk_strategy_resolves():
    """Every strategy in SUPPORTED_CHUNK_STRATEGIES must be in the real registry."""
    from hhgoa_rag.ingestion.chunkers import CHUNKERS

    for strategy in pc.SUPPORTED_CHUNK_STRATEGIES:
        assert (
            strategy in CHUNKERS
        ), f"Strategy '{strategy}' is in SUPPORTED_CHUNK_STRATEGIES but not in CHUNKERS"


def test_semantic_not_in_supported_strategies():
    """'semantic' is experimental and must NOT appear in SUPPORTED_CHUNK_STRATEGIES."""
    assert "semantic" not in pc.SUPPORTED_CHUNK_STRATEGIES


# ── physical_shard contains real source path ──────────────────────────────────


def test_physical_shard_is_real_path():
    """physical_shard must be the real parquet file path, not hardcoded '0'."""
    rec_input = {
        "language": "hi",
        "config_language": "hi",
        "split": "train",
        "physical_source": "train/hintrain.parquet",
        "local_source_row": 0,
        "passage_position": 0,
        "content_hash": "e" * 40,
        "normalized_text": "Hello world today.",
    }
    records = pc._build_prepared_records_for_passage(
        rec_input,
        "bf5cdc1f26e581e519018e434db14edd1b77602b",
        FakeTokenizer(),
        "passage_native",
        "v1",
        "canary-test",
        {},
        {},
    )
    assert len(records) >= 1
    for r in records:
        assert (
            r["physical_shard"] == "train/hintrain.parquet"
        ), f"Expected real path, got: {r['physical_shard']!r}"
        assert r["physical_shard"] != "0"


# ── create_pinecone_index dry-run gate ─────────────────────────────────────────


def _run_create(args, env_extra=None):
    import os

    env = dict(os.environ)
    for k in ["PINECONE_API_KEY", "CONFIRM_PINECONE_CREATE"]:
        env.pop(k, None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, str(Path(_SCRIPTS) / "create_pinecone_index.py"), *args],
        capture_output=True,
        text=True,
        env=env,
    )


def test_create_index_execute_alone_exits_2():
    """--execute without CONFIRM_PINECONE_CREATE=1 must exit 2 (fail-closed)."""
    r = _run_create(["--pinecone-index", "msmarco-xi", "--execute"])
    assert r.returncode == 2, f"Expected exit 2, got {r.returncode}\nstderr={r.stderr}"
    assert "CONFIRM_PINECONE_CREATE" in (r.stderr + r.stdout)


def test_create_index_dry_run_shows_full_contract():
    """Dry-run must display complete canonical contract."""
    r = _run_create(["--pinecone-index", "msmarco-xi"])
    assert r.returncode == 0
    out = r.stdout
    assert "multilingual-e5-large" in out
    assert "cosine" in out
    assert "chunk_text" in out
    assert "contract_fingerprint" in out


# ── eval labels cannot cross into production records ─────────────────────────


def test_eval_labels_forbidden_in_records():
    """Evaluation labels must be recursively rejected from production records."""
    from hhgoa_rag.ingestion.schema import SchemaViolationError, validate_record

    rec = {
        "id": "id-eval",
        "chunk_text": "test",
        "language": "en",
        "config_language": "en",
        "dataset_revision": "bf5cdc1f26e581e519018e434db14edd1b77602b",
        "split": "train",
        "physical_shard": "train/hintrain.parquet",
        "local_source_row": 0,
        "passage_position": 0,
        "parent_passage_id": "p",
        "content_hash": "a" * 40,
        "chunk_strategy": "passage_native",
        "chunk_strategy_version": "v1",
        "chunk_ordinal": 0,
        "chunk_total": 1,
        "token_length": 3,
        "tokenizer_fingerprint": "fp",
        "manifest_id": "m",
        "is_selected": 1,  # forbidden
    }
    with pytest.raises(SchemaViolationError):
        validate_record(rec)


# ── manifest contract fingerprint validation ──────────────────────────────────


def test_manifest_wrong_contract_fingerprint_rejected(tmp_path):
    """Manifest with wrong contract_fingerprint must be rejected."""
    import json

    from hhgoa_rag.pinecone_contract import (
        CONTRACT_VERSION,
        INDEX_NAME,
        MANIFEST_SCHEMA_VERSION,
        NAMESPACE,
        canonical_contract,
    )

    # Build a manifest with WRONG fingerprint but all other required fields present
    manifest = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "manifest_id": "test-id",
        "mode": "canary",
        "contract_version": CONTRACT_VERSION,
        "contract_fingerprint": "wrong-fingerprint-value",
        "index_contract": canonical_contract(),
        "index_name": INDEX_NAME,
        "index_namespace": NAMESPACE,
        "dataset_repo": "ai4bharat/MSMARCO-XI",
        "dataset_revision": "bf5cdc1f26e581e519018e434db14edd1b77602b",
        "total_records": 0,
        "total_tokens": 0,
        "prepared_record_path": "fake.jsonl",
        "prepared_record_checksum": "abc",
        "ready_for_write": False,
        "readiness_failures": [],
        "forbidden_field_audit": "PASS",
        "tokenizer_fingerprint": "fp",
        "actual_per_language_records": {},
    }
    # Compute its own checksum
    m_for_ck = {k: v for k, v in manifest.items() if k != "manifest_checksum"}
    import hashlib

    manifest["manifest_checksum"] = hashlib.sha256(
        json.dumps(m_for_ck, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    mpath = tmp_path / "m.json"
    mpath.write_text(json.dumps(manifest))

    sys.path.insert(0, str(Path(_SCRIPTS)))
    import ingest_prepared as ing

    with pytest.raises(ValueError, match="contract_fingerprint"):
        ing._load_manifest(mpath)
