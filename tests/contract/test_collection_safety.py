"""Contract: destructive ops refuse production/alias names."""

from unittest.mock import MagicMock

import pytest

from hhgoa_rag.qdrant_lifecycle import SERVING_ALIAS, _is_destructive_safe, create_collection


def test_smoke_is_safe():
    assert _is_destructive_safe("msmarco_xi_passages_smoke_v001")


def test_pilot_is_safe():
    assert _is_destructive_safe("msmarco_xi_passages_pilot_v001")


def test_production_is_not_safe():
    assert not _is_destructive_safe("msmarco_xi_passages_v001")


def test_alias_is_not_safe():
    assert not _is_destructive_safe(SERVING_ALIAS)


def test_force_refuses_production():
    client = MagicMock()
    client.get_collections.return_value.collections = []
    # Should not raise for smoke
    create_collection(client, "msmarco_xi_passages_smoke_v001", force=True)
    # Should raise for production
    with pytest.raises(ValueError, match="--force refused"):
        create_collection(client, "msmarco_xi_passages_v001", force=True)


def test_test_prefix_is_safe():
    assert _is_destructive_safe("test_collection_abc")
