from hhgoa_rag.config.settings import Settings


def test_default_corpus_mode():
    s = Settings()
    assert s.corpus_mode == "smoke"


def test_corpus_mode_from_env(monkeypatch):
    monkeypatch.setenv("CORPUS_MODE", "full")
    s = Settings()
    assert s.corpus_mode == "full"


def test_max_query_chars_default():
    s = Settings()
    assert s.max_query_chars == 512


def test_pinecone_index_default():
    s = Settings()
    assert s.pinecone_index == "msmarco-xi"


def test_pinecone_embed_model_default():
    s = Settings()
    assert s.pinecone_embed_model == "multilingual-e5-large"


def test_pinecone_namespace_default():
    s = Settings()
    assert s.pinecone_namespace == "smoke"


def test_missing_pinecone_api_key_ok():
    # Key is optional at startup; app marks not-ready instead of crashing
    s = Settings()
    assert s.pinecone_api_key is None


def test_missing_sarvam_key_ok():
    s = Settings()
    assert s.sarvam_api_key is None
