from hhgoa_rag.config.settings import Settings


def test_default_corpus_mode():
    # "full" here means the self-hosted local hybrid index (6/14 MSMARCO-XI
    # language configs, 54.25M passages) that's the live serving backend as
    # of 2026-08-22 -- not the old Pinecone smoke default.
    s = Settings(_env_file=None)
    assert s.corpus_mode == "full"


def test_corpus_mode_from_env(monkeypatch):
    monkeypatch.setenv("CORPUS_MODE", "full")
    s = Settings(_env_file=None)
    assert s.corpus_mode == "full"


def test_max_query_chars_default():
    s = Settings(_env_file=None)
    assert s.max_query_chars == 512


def test_local_index_root_default():
    s = Settings(_env_file=None)
    assert s.local_index_root == "artifacts/full_local_index"


def test_missing_sarvam_key_ok(monkeypatch):
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    s = Settings(_env_file=None)
    assert not s.sarvam_api_key
