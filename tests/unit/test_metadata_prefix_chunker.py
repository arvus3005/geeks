"""Real tests for MetadataPrefixChunker -- a 5th chunking strategy, wraps
another chunker and injects a metadata prefix into the embedding text only,
never the displayable text."""

from hhgoa_rag.ingestion.chunkers import MetadataPrefixChunker, PassageNativeChunker


def test_no_metadata_leaves_chunks_unchanged():
    chunker = MetadataPrefixChunker(base=PassageNativeChunker(), metadata_for={})
    chunks = chunker.chunk("A corporation is a legal entity.", "p001")
    assert len(chunks) == 1
    assert chunks[0].text == "A corporation is a legal entity."
    assert chunks[0].embedding_text is None
    assert chunks[0].chunk_strategy == "passage_native"  # untouched, no prefix applied


def test_prefix_applied_to_embedding_text_only():
    metadata_for = {"p001": {"lang": "hi", "source": "msmarco-xi", "doc_id": "d42"}}
    chunker = MetadataPrefixChunker(base=PassageNativeChunker(), metadata_for=metadata_for)
    chunks = chunker.chunk("A corporation is a legal entity.", "p001")

    assert len(chunks) == 1
    c = chunks[0]
    # Displayable text is untouched -- never show the prefix to a user.
    assert c.text == "A corporation is a legal entity."
    # Embedding text carries the prefix.
    assert c.embedding_text == "[hi|msmarco-xi|d42] A corporation is a legal entity."
    assert c.chunk_strategy == "metadata_prefix"


def test_partial_metadata_omits_missing_keys():
    metadata_for = {"p001": {"lang": "bn"}}  # no source/doc_id
    chunker = MetadataPrefixChunker(base=PassageNativeChunker(), metadata_for=metadata_for)
    chunks = chunker.chunk("Some text.", "p001")
    assert chunks[0].embedding_text == "[bn] Some text."


def test_unknown_passage_id_no_prefix():
    metadata_for = {"p001": {"lang": "hi"}}
    chunker = MetadataPrefixChunker(base=PassageNativeChunker(), metadata_for=metadata_for)
    chunks = chunker.chunk("Some other text.", "p999")  # not in metadata_for
    assert chunks[0].embedding_text is None
    assert chunks[0].text == "Some other text."


def test_applies_to_every_sub_chunk():
    from hhgoa_rag.ingestion.chunkers import SentenceAwareChunker

    metadata_for = {"p001": {"lang": "ta"}}
    chunker = MetadataPrefixChunker(base=SentenceAwareChunker(target_chars=40), metadata_for=metadata_for)
    text = "First sentence here. Second sentence follows. Third one too. " * 5
    chunks = chunker.chunk(text, "p001")
    assert len(chunks) > 1  # confirm this actually produced multiple chunks
    for c in chunks:
        assert c.embedding_text is not None
        assert c.embedding_text.startswith("[ta] ")
        assert not c.text.startswith("[ta] ")  # displayable text never carries the prefix
