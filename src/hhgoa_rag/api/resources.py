"""Application-level resource container.

The local hybrid index's query embedder and shards are loaded once during
FastAPI lifespan startup (see api/app.py) and reused across all requests;
sharded_local_hybrid_store.py owns its own module-level shard cache
directly rather than storing a client handle here. Readiness stays False
until at least one shard is discovered and warmed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class AppResources:
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
