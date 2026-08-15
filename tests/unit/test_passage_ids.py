from hhgoa_rag.ingestion.passage_ids import make_passage_id, make_content_hash


def test_deterministic():
    id1 = make_passage_id("train", "en", "q001", 0)
    id2 = make_passage_id("train", "en", "q001", 0)
    assert id1 == id2


def test_namespaced_diff_lang():
    id_en = make_passage_id("train", "en", "q001", 0)
    id_bn = make_passage_id("train", "bn", "q001", 0)
    assert id_en != id_bn


def test_content_hash_deterministic():
    h1 = make_content_hash("hello world")
    h2 = make_content_hash("hello world")
    assert h1 == h2


def test_content_hash_different():
    assert make_content_hash("foo") != make_content_hash("bar")
