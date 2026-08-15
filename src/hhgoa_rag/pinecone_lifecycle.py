"""Pinecone index lifecycle management.

Rules:
- Never delete or recreate an existing index automatically.
- create_index_idempotent returns the existing index model if it already exists.
- validate_index checks model, field_map, cloud, region and returns structured errors.
- Clear actionable errors for incompatible existing indexes.
"""

from __future__ import annotations

import logging
from typing import Any

from pinecone import IndexEmbed

from .pinecone_store import FIELD_MAP, TEXT_RECORD_FIELD

logger = logging.getLogger(__name__)

DEFAULT_EMBED_MODEL = "multilingual-e5-large"
DEFAULT_METRIC = "cosine"


def create_index_idempotent(
    pc: Any,  # pinecone.Pinecone
    name: str,
    cloud: str,
    region: str,
    embed_model: str = DEFAULT_EMBED_MODEL,
    tags: dict[str, str] | None = None,
) -> Any:  # IndexModel
    """Create an integrated-embedding index, or return the existing one.

    Never deletes or recreates. Validates that an existing index is compatible.
    Raises ValueError with a clear message if incompatible.
    """
    existing = {idx.name for idx in pc.list_indexes().indexes}
    if name in existing:
        logger.info("Index '%s' already exists — validating compatibility", name)
        errors = validate_index(pc, name, embed_model=embed_model, cloud=cloud, region=region)
        if errors:
            msg = (
                f"Existing index '{name}' is incompatible with requested config:\n"
                + "\n".join(f"  • {e}" for e in errors)
                + "\nRename the index or reconcile the config before proceeding."
            )
            raise ValueError(msg)
        return pc.describe_index(name)

    logger.info(
        "Creating integrated-embedding index '%s' (model=%s, cloud=%s, region=%s)",
        name,
        embed_model,
        cloud,
        region,
    )
    return pc.create_index_for_model(
        name=name,
        cloud=cloud,
        region=region,
        embed=IndexEmbed(
            model=embed_model,
            field_map=FIELD_MAP,
            metric=DEFAULT_METRIC,
        ),
        tags=tags or {},
    )


def validate_index(
    pc: Any,
    name: str,
    embed_model: str = DEFAULT_EMBED_MODEL,
    cloud: str | None = None,
    region: str | None = None,
) -> list[str]:
    """Validate an existing index against expected config. Returns list of error strings."""
    try:
        info = pc.describe_index(name)
    except Exception as e:
        return [f"Could not describe index '{name}': {e}"]

    errors: list[str] = []

    # Check embed model
    spec = getattr(info, "spec", None)
    embed_cfg = getattr(spec, "embed", None) if spec else None
    if embed_cfg is None:
        # Try integrated spec
        integrated = getattr(spec, "integrated", None) if spec else None
        embed_cfg = getattr(integrated, "embed", None) if integrated else None

    if embed_cfg is not None:
        actual_model = getattr(embed_cfg, "model", None)
        if actual_model and actual_model != embed_model:
            errors.append(
                f"embed model mismatch: expected '{embed_model}', got '{actual_model}'"
            )
        actual_field_map = getattr(embed_cfg, "field_map", None)
        if actual_field_map and actual_field_map != FIELD_MAP:
            errors.append(
                f"field_map mismatch: expected {FIELD_MAP!r}, got {actual_field_map!r}"
            )
    else:
        errors.append(
            "Could not verify embed config — index may not be an integrated-embedding index"
        )

    # Check cloud/region if provided
    if cloud or region:
        serverless = getattr(spec, "serverless", None) if spec else None
        if serverless:
            actual_cloud = str(getattr(serverless, "cloud", "")).lower()
            actual_region = str(getattr(serverless, "region", "")).lower()
            if cloud and actual_cloud != cloud.lower():
                errors.append(f"cloud mismatch: expected '{cloud}', got '{actual_cloud}'")
            if region and actual_region != region.lower():
                errors.append(f"region mismatch: expected '{region}', got '{actual_region}'")

    return errors


def get_index_info(pc: Any, name: str) -> dict[str, Any]:
    """Return a structured summary of the index for display/logging."""
    try:
        info = pc.describe_index(name)
    except Exception as e:
        return {"error": str(e), "name": name}

    spec = getattr(info, "spec", None)
    embed_cfg = getattr(spec, "embed", None) if spec else None
    if embed_cfg is None:
        integrated = getattr(spec, "integrated", None) if spec else None
        embed_cfg = getattr(integrated, "embed", None) if integrated else None

    return {
        "name": name,
        "status": getattr(info, "status", {}).get("state") if hasattr(info, "status") else "unknown",
        "embed_model": getattr(embed_cfg, "model", "unknown") if embed_cfg else "unknown",
        "field_map": getattr(embed_cfg, "field_map", {}) if embed_cfg else {},
        "text_record_field": TEXT_RECORD_FIELD,
    }
