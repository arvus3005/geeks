"""Test deterministic UUID point IDs."""

import uuid

from hhgoa_rag.ingestion.passage_ids import make_content_hash, make_passage_id, make_point_id


def test_point_id_is_valid_uuid():
    pid = make_point_id("rev1", "en", "abc123", "passage_native_v1", 0)
    parsed = uuid.UUID(pid)
    assert parsed.version == 5


def test_point_id_determinism():
    a = make_point_id("rev1", "en", "abc123", "passage_native_v1", 0)
    b = make_point_id("rev1", "en", "abc123", "passage_native_v1", 0)
    assert a == b


def test_point_id_no_cross_language_collision():
    en = make_point_id("rev1", "en", "samehash", "passage_native_v1", 0)
    bn = make_point_id("rev1", "bn", "samehash", "passage_native_v1", 0)
    assert en != bn


def test_point_id_no_cross_strategy_collision():
    a = make_point_id("rev1", "en", "h", "passage_native_v1", 0)
    b = make_point_id("rev1", "en", "h", "fixed_token_v1", 0)
    assert a != b


def test_point_id_no_cross_ordinal_collision():
    a = make_point_id("rev1", "en", "h", "passage_native_v1", 0)
    b = make_point_id("rev1", "en", "h", "passage_native_v1", 1)
    assert a != b


def test_content_hash_deterministic():
    h1 = make_content_hash("hello world")
    h2 = make_content_hash("hello world")
    assert h1 == h2


def test_content_hash_different():
    assert make_content_hash("foo") != make_content_hash("bar")


def test_make_passage_id_legacy():
    id1 = make_passage_id("train", "en", "q001", 0)
    id2 = make_passage_id("train", "en", "q001", 0)
    assert id1 == id2
    assert len(id1) == 24


def test_make_passage_id_namespaced_diff_lang():
    id_en = make_passage_id("train", "en", "q001", 0)
    id_bn = make_passage_id("train", "bn", "q001", 0)
    assert id_en != id_bn
