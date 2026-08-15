"""Contract: pilot/smoke manifests cannot activate production alias."""

from unittest.mock import MagicMock

import pytest

from hhgoa_rag.qdrant_lifecycle import switch_alias


def test_smoke_collection_refused_for_production_alias():
    client = MagicMock()
    with pytest.raises(ValueError, match="smoke/pilot"):
        switch_alias(
            client,
            "msmarco_xi_passages_smoke_v001",
            alias="msmarco_xi_passages_current",
            smoke_ok=False,
        )


def test_pilot_collection_refused_for_production_alias():
    client = MagicMock()
    with pytest.raises(ValueError, match="smoke/pilot"):
        switch_alias(
            client,
            "msmarco_xi_passages_pilot_v001",
            alias="msmarco_xi_passages_current",
            smoke_ok=False,
        )
