from hhgoa_rag.ingestion.chunkers import FixedTokenChunker, PassageNativeChunker


def test_passage_native_one_chunk():
    chunker = PassageNativeChunker()
    chunks = chunker.chunk("Hello world this is a test passage.", "p001")
    assert len(chunks) == 1
    assert chunks[0].chunk_strategy == "passage_native"
    assert chunks[0].chunk_ordinal == 0


def test_fixed_token_short_passage():
    chunker = FixedTokenChunker(target_tokens=256)
    text = "Short passage"
    chunks = chunker.chunk(text, "p001")
    assert len(chunks) == 1


def test_fixed_token_long_passage():
    chunker = FixedTokenChunker(target_tokens=5, overlap_ratio=0.2)
    text = " ".join(["word"] * 20)
    chunks = chunker.chunk(text, "p001")
    assert len(chunks) > 1
    assert all(c.parent_passage_id == "p001" for c in chunks)


def test_chunk_has_parent_id():
    chunker = PassageNativeChunker()
    chunks = chunker.chunk("text", "parent_xyz")
    assert chunks[0].parent_passage_id == "parent_xyz"
