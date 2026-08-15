"""Application-level resource container.

PineconeStore is loaded once during FastAPI lifespan and reused across all
requests. Readiness stays False until the store is verified and a lightweight
describe_index_stats probe passes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from hhgoa_rag.pinecone_store import PineconeStore

logger = logging.getLogger(__name__)


@dataclass
class AppResources:
    pinecone_store: PineconeStore | None = None
    ready: bool = False
    readiness_detail: dict = field(default_factory=dict)

    def mark_ready(self) -> None:
        self.ready = True

    def mark_not_ready(self, reason: str) -> None:
        self.ready = False
        self.readiness_detail["reason"] = reason


# Module-level singleton — set during lifespan startup.
_resources: AppResources | None = None


def get_resources() -> AppResources:
    global _resources
    if _resources is None:
        _resources = AppResources()
    return _resources


def _reset_resources() -> None:
    """For testing only — reset singleton."""
    global _resources
    _resources = None
