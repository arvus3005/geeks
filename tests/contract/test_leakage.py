"""Contract: forbidden fields never enter production point payloads."""

from hhgoa_rag.dataset.parser import FORBIDDEN_FIELDS, parse_record

FORBIDDEN = {"query", "Answer", "Eng_Query", "Eng_Answer", "query_type", "is_selected"}


def make_record():
    return {
        "query_id": "q1",
        "query": "What is the capital of India?",
        "Answer": "New Delhi",
        "Eng_Query": "What is the capital of India?",
        "Eng_Answer": "New Delhi",
        "query_type": "description",
        "passages": {
            "is_selected": [1, 0],
            "English_passages": [
                "New Delhi is the capital of India.",
                "Mumbai is in India.",
            ],
            "Translated_passages": [
                "নয়াদিল্লি ভারতের রাজধানী।",
                "মুম্বাই ভারতে আছে।",
            ],
        },
    }


def test_no_forbidden_fields_in_occurrences():
    record = make_record()
    occurrences, _ = parse_record(record, "bn", "train", "shard0", 0)
    for occ in occurrences:
        payload_like = {
            "text": occ.normalized_text,
            "language": occ.passage_language,
            "content_hash": occ.content_hash,
        }
        for field in FORBIDDEN:
            assert field not in payload_like, f"Forbidden field '{field}' in payload"


def test_forbidden_fields_set():
    assert FORBIDDEN_FIELDS == FORBIDDEN


def test_parser_returns_both_languages():
    record = make_record()
    occurrences, rejected = parse_record(record, "bn", "train", "shard0", 0)
    langs = {o.passage_language for o in occurrences}
    assert "en" in langs
    assert "bn" in langs
    assert len(rejected) == 0


def test_parser_rejects_short_passages():
    record = {
        "passages": {
            "English_passages": ["ok"],  # too short
            "Translated_passages": ["ok"],  # too short
        }
    }
    occurrences, rejected = parse_record(record, "hi", "train", "s", 0)
    assert len(occurrences) == 0
    assert len(rejected) >= 2
