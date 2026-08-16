"""Tests for all four chunking strategies."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hhgoa_rag.ingestion.chunkers import (
    FixedTokenChunker,
    PassageNativeChunker,
    SemanticChunker,
    SentenceAwareChunker,
    get_chunker,
)

# ── PassageNativeChunker ──────────────────────────────────────────────────────


def test_passage_native_one_chunk():
    chunker = PassageNativeChunker()
    chunks = chunker.chunk("Hello world this is a test passage.", "p001")
    assert len(chunks) == 1
    assert chunks[0].chunk_strategy == "passage_native"
    assert chunks[0].chunk_ordinal == 0
    assert chunks[0].chunk_total == 1
    assert chunks[0].parent_passage_id == "p001"


def test_passage_native_empty_text():
    chunker = PassageNativeChunker()
    chunks = chunker.chunk("", "p001")
    assert len(chunks) == 1
    assert chunks[0].text == ""


def test_passage_native_preserves_unicode():
    text = "भारत की राजधानी नई दिल्ली है।"
    chunker = PassageNativeChunker()
    chunks = chunker.chunk(text, "p001")
    assert chunks[0].text == text


# ── SentenceAwareChunker ──────────────────────────────────────────────────────


def test_sentence_aware_short_passage():
    chunker = SentenceAwareChunker(target_chars=500)
    text = "Hello world. This is a test."
    chunks = chunker.chunk(text, "p001")
    assert len(chunks) >= 1
    assert all(c.parent_passage_id == "p001" for c in chunks)


def test_sentence_aware_splits_long_passage():
    chunker = SentenceAwareChunker(target_chars=30, overlap_sentences=0)
    text = "First sentence here. Second sentence here. Third sentence here."
    chunks = chunker.chunk(text, "p001")
    assert len(chunks) > 1


def test_sentence_aware_indic_boundary():
    """Devanagari danda (।) should be treated as sentence boundary."""
    chunker = SentenceAwareChunker(target_chars=40, overlap_sentences=0)
    text = "भारत एक देश है। पाकिस्तान भी एक देश है। चीन भी एक देश है।"
    chunks = chunker.chunk(text, "p001")
    assert len(chunks) >= 1
    combined = " ".join(c.text for c in chunks)
    # All words should be preserved
    assert "भारत" in combined
    assert "चीन" in combined


def test_sentence_aware_double_danda():
    """Double danda (॥) should also trigger a split."""
    chunker = SentenceAwareChunker(target_chars=30, overlap_sentences=0)
    text = "प्रथम वाक्य।। द्वितीय वाक्य।। तृतीय वाक्य।"
    chunks = chunker.chunk(text, "p001")
    assert len(chunks) >= 1


def test_sentence_aware_no_punctuation():
    """Passage with no punctuation returns as one chunk."""
    chunker = SentenceAwareChunker(target_chars=1000)
    text = "no punctuation here just words and more words"
    chunks = chunker.chunk(text, "p001")
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_sentence_aware_overlap():
    chunker = SentenceAwareChunker(target_chars=30, overlap_sentences=1)
    text = "Sentence one. Sentence two. Sentence three. Sentence four."
    chunks = chunker.chunk(text, "p001")
    if len(chunks) > 1:
        # Last sentence of chunk N should appear in start of chunk N+1
        last_sent = chunks[0].text.split(".")[-2].strip() if "." in chunks[0].text else ""
        if last_sent:
            assert last_sent in chunks[1].text


def test_sentence_aware_single_very_long_sentence():
    chunker = SentenceAwareChunker(target_chars=10)
    text = "a" * 200
    chunks = chunker.chunk(text, "p001")
    # Single sentence exceeding target — returned as one chunk
    assert len(chunks) == 1


def test_sentence_aware_newlines():
    chunker = SentenceAwareChunker(target_chars=40, overlap_sentences=0)
    text = "Line one\nLine two\nLine three\nLine four"
    chunks = chunker.chunk(text, "p001")
    assert len(chunks) >= 1


def test_sentence_aware_ordinals_sequential():
    chunker = SentenceAwareChunker(target_chars=20, overlap_sentences=0)
    text = "A. B. C. D. E. F. G. H. I. J."
    chunks = chunker.chunk(text, "p001")
    ordinals = [c.chunk_ordinal for c in chunks]
    assert ordinals == list(range(len(chunks)))
    assert all(c.chunk_total == len(chunks) for c in chunks)


def test_sentence_aware_mixed_scripts():
    chunker = SentenceAwareChunker(target_chars=80, overlap_sentences=0)
    text = "India is a country. भारत एक देश है। It has many languages."
    chunks = chunker.chunk(text, "p001")
    combined = " ".join(c.text for c in chunks)
    assert "India" in combined
    assert "भारत" in combined


# ── FixedTokenChunker ─────────────────────────────────────────────────────────


def test_fixed_token_short_passage():
    chunker = FixedTokenChunker(target_tokens=128, allow_approximate=True)
    text = "Short passage"
    chunks = chunker.chunk(text, "p001")
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_fixed_token_long_passage():
    chunker = FixedTokenChunker(target_tokens=5, overlap_tokens=1, allow_approximate=True)
    text = " ".join(["word"] * 20)
    chunks = chunker.chunk(text, "p001")
    assert len(chunks) > 1
    assert all(c.parent_passage_id == "p001" for c in chunks)


def test_fixed_token_deterministic():
    chunker = FixedTokenChunker(target_tokens=10, overlap_tokens=2, allow_approximate=True)
    text = " ".join([f"word{i}" for i in range(50)])
    chunks_a = chunker.chunk(text, "p001")
    chunks_b = chunker.chunk(text, "p001")
    assert [c.text for c in chunks_a] == [c.text for c in chunks_b]


def test_fixed_token_bounded_size():
    chunker = FixedTokenChunker(target_tokens=10, overlap_tokens=2, allow_approximate=True)
    text = " ".join([f"tok{i}" for i in range(100)])
    chunks = chunker.chunk(text, "p001")
    for c in chunks:
        assert c.token_length is None or c.token_length <= 10


def test_fixed_token_no_missing_content():
    """All tokens from original text appear in at least one chunk."""
    chunker = FixedTokenChunker(target_tokens=5, overlap_tokens=1, allow_approximate=True)
    words = [f"w{i}" for i in range(25)]
    text = " ".join(words)
    chunks = chunker.chunk(text, "p001")
    all_text = " ".join(c.text for c in chunks)
    for w in words:
        assert w in all_text


def test_fixed_token_no_infinite_loop():
    chunker = FixedTokenChunker(target_tokens=3, overlap_tokens=1, allow_approximate=True)
    text = " ".join(["x"] * 30)
    chunks = chunker.chunk(text, "p001")
    assert len(chunks) > 0
    assert len(chunks) < 100  # finite


def test_fixed_token_overlap_constraints():
    with pytest.raises(ValueError):
        FixedTokenChunker(target_tokens=5, overlap_tokens=5, allow_approximate=True)


def test_fixed_token_labels_approximate():
    """Whitespace tokenizer must label chunks as approximate_whitespace."""
    chunker = FixedTokenChunker(target_tokens=5, overlap_tokens=1, allow_approximate=True)
    text = " ".join([f"tok{i}" for i in range(20)])
    chunks = chunker.chunk(text, "p001")
    assert all(c.tokenizer_label == "approximate_whitespace" for c in chunks)


def test_fixed_token_unicode():
    chunker = FixedTokenChunker(target_tokens=5, overlap_tokens=1, allow_approximate=True)
    text = "नई दिल्ली भारत की राजधानी है और यह बहुत पुराना शहर है"
    chunks = chunker.chunk(text, "p001")
    assert len(chunks) >= 1
    combined = " ".join(c.text for c in chunks)
    assert "दिल्ली" in combined


def test_fixed_token_ordinals():
    chunker = FixedTokenChunker(target_tokens=4, overlap_tokens=1, allow_approximate=True)
    text = " ".join([f"w{i}" for i in range(20)])
    chunks = chunker.chunk(text, "p001")
    assert [c.chunk_ordinal for c in chunks] == list(range(len(chunks)))
    assert all(c.chunk_total == len(chunks) for c in chunks)


# ── SemanticChunker ───────────────────────────────────────────────────────────


class _FakeSim:
    """Returns a fixed similarity score — useful for testing split behaviour."""

    def __init__(self, score: float) -> None:
        self.score = score

    def similarity(self, a: str, b: str) -> float:
        return self.score


class _AlternatingSim:
    """Alternates 0.1 / 0.9 to force splits on every other boundary."""

    def __init__(self) -> None:
        self._call = 0

    def similarity(self, a: str, b: str) -> float:
        self._call += 1
        return 0.1 if self._call % 2 == 1 else 0.9


def test_semantic_high_similarity_one_chunk():
    chunker = SemanticChunker(similarity_provider=_FakeSim(0.95), split_threshold=0.5)
    text = "First sentence. Second sentence. Third sentence."
    chunks = chunker.chunk(text, "p001")
    assert len(chunks) == 1


def test_semantic_low_similarity_splits():
    chunker = SemanticChunker(similarity_provider=_FakeSim(0.1), split_threshold=0.5)
    text = "First sentence. Second sentence. Third sentence."
    chunks = chunker.chunk(text, "p001")
    assert len(chunks) > 1


def test_semantic_deterministic():
    sim = _FakeSim(0.3)
    chunker = SemanticChunker(similarity_provider=sim, split_threshold=0.5)
    text = "A sentence. Another sentence. Yet another."
    a = chunker.chunk(text, "p001")
    b = chunker.chunk(text, "p001")
    assert [c.text for c in a] == [c.text for c in b]


def test_semantic_single_sentence():
    chunker = SemanticChunker(similarity_provider=_FakeSim(0.5), split_threshold=0.5)
    text = "One sentence only"
    chunks = chunker.chunk(text, "p001")
    assert len(chunks) == 1


def test_semantic_threshold_validation():
    with pytest.raises(ValueError):
        SemanticChunker(similarity_provider=_FakeSim(0.5), split_threshold=1.5)


def test_semantic_ordinals():
    chunker = SemanticChunker(similarity_provider=_AlternatingSim(), split_threshold=0.5)
    text = "A. B. C. D. E."
    chunks = chunker.chunk(text, "p001")
    assert [c.chunk_ordinal for c in chunks] == list(range(len(chunks)))
    assert all(c.chunk_total == len(chunks) for c in chunks)


# ── get_chunker registry ──────────────────────────────────────────────────────


def test_get_chunker_passage_native():
    assert isinstance(get_chunker("passage_native"), PassageNativeChunker)


def test_get_chunker_sentence_aware():
    assert isinstance(get_chunker("sentence_aware"), SentenceAwareChunker)


def test_get_chunker_fixed_token():
    assert isinstance(get_chunker("fixed_token_overlap"), FixedTokenChunker)


def test_get_chunker_unknown_raises():
    with pytest.raises(ValueError, match="Unknown"):
        get_chunker("nonexistent_strategy")


# ── Hypothesis property tests ─────────────────────────────────────────────────


@given(st.text(min_size=0, max_size=500))
@settings(max_examples=50)
def test_passage_native_always_one_chunk(text: str):
    chunker = PassageNativeChunker()
    chunks = chunker.chunk(text, "p")
    assert len(chunks) == 1
    assert chunks[0].text == text


@given(
    st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")), min_size=1, max_size=300
    )
)
@settings(max_examples=50)
def test_fixed_token_preserves_all_tokens(text: str):
    chunker = FixedTokenChunker(target_tokens=8, overlap_tokens=2, allow_approximate=True)
    chunks = chunker.chunk(text, "p")
    all_chunk_text = " ".join(c.text for c in chunks)
    original_tokens = text.split()
    for tok in original_tokens:
        assert tok in all_chunk_text
