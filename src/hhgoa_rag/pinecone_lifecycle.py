"""Pinecone index lifecycle management.

Rules:
- Never delete or recreate an existing index automatically.
- create_index_idempotent returns the existing index model if it already exists.
- validate_index checks ALL canonical contract fields and returns structured errors.
- Any field that cannot be verified produces an "unverifiable contract field" error.
- Clear actionable errors for incompatible existing indexes.

All contract constants are imported from pinecone_contract — there are no
hand-written contract dicts in this module.
"""

from __future__ import annotations

import logging
from typing import Any

from pinecone import IndexEmbed

from .pinecone_contract import (
    CLOUD,
    DIMENSION,
    FIELD_MAP,
    METRIC,
    MODEL,
    READ_PARAMETERS,
    REGION,
    WRITE_PARAMETERS,
    canonical_contract,
)
from .pinecone_store import TEXT_RECORD_FIELD

logger = logging.getLogger(__name__)


def create_index_idempotent(
    pc: Any,  # pinecone.Pinecone
    name: str,
    cloud: str = CLOUD,
    region: str = REGION,
    embed_model: str = MODEL,
    tags: dict[str, str] | None = None,
) -> Any:  # IndexModel
    """Create an integrated-embedding index, or return the existing one.

    Never deletes or recreates. Validates that an existing index is compatible
    against ALL canonical contract fields.
    Raises ValueError with a clear message if incompatible.
    """
    existing = {idx.name for idx in pc.list_indexes().indexes}
    if name in existing:
        logger.info("Index '%s' already exists — validating compatibility", name)
        errors = validate_index(pc, name)
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
            metric=METRIC,
            write_parameters=WRITE_PARAMETERS,
            read_parameters=READ_PARAMETERS,
        ),
        tags=tags or {},
    )


def validate_index(pc: Any, name: str) -> list[str]:
    """Validate an existing index against ALL canonical contract fields.

    Returns list of error strings.  Any field that cannot be verified produces
    an explicit "unverifiable contract field" error rather than a silent pass.
    """
    try:
        info = pc.describe_index(name)
    except Exception as e:
        return [f"Could not describe index '{name}': {e}"]

    errors: list[str] = []

    spec = getattr(info, "spec", None)
    embed_cfg = getattr(spec, "embed", None) if spec else None
    if embed_cfg is None:
        # Try integrated spec layout
        integrated = getattr(spec, "integrated", None) if spec else None
        embed_cfg = getattr(integrated, "embed", None) if integrated else None

    if embed_cfg is not None:
        # Model
        actual_model = getattr(embed_cfg, "model", None)
        if actual_model is None:
            errors.append("unverifiable contract field: embed model (not returned by API)")
        elif actual_model != MODEL:
            errors.append(f"embed model mismatch: expected '{MODEL}', got '{actual_model}'")

        # field_map
        actual_field_map = getattr(embed_cfg, "field_map", None)
        if actual_field_map is None:
            errors.append("unverifiable contract field: field_map (not returned by API)")
        elif actual_field_map != FIELD_MAP:
            errors.append(f"field_map mismatch: expected {FIELD_MAP!r}, got {actual_field_map!r}")

        # metric
        actual_metric = getattr(embed_cfg, "metric", None)
        if actual_metric is None:
            errors.append("unverifiable contract field: metric (not returned by API)")
        elif actual_metric != METRIC:
            errors.append(f"metric mismatch: expected '{METRIC}', got '{actual_metric}'")

        # write_parameters
        actual_write = getattr(embed_cfg, "write_parameters", None)
        if actual_write is None:
            errors.append("unverifiable contract field: write_parameters (not returned by API)")
        else:
            _wp = dict(actual_write) if not isinstance(actual_write, dict) else actual_write
            if _wp != WRITE_PARAMETERS:
                errors.append(
                    f"write_parameters mismatch: expected {WRITE_PARAMETERS!r}, got {_wp!r}"
                )

        # read_parameters
        actual_read = getattr(embed_cfg, "read_parameters", None)
        if actual_read is None:
            errors.append("unverifiable contract field: read_parameters (not returned by API)")
        else:
            _rp = dict(actual_read) if not isinstance(actual_read, dict) else actual_read
            if _rp != READ_PARAMETERS:
                errors.append(
                    f"read_parameters mismatch: expected {READ_PARAMETERS!r}, got {_rp!r}"
                )
    else:
        errors.append(
            "unverifiable contract field: embed config — index may not be an "
            "integrated-embedding index"
        )

    # dimension
    actual_dim = getattr(info, "dimension", None)
    if actual_dim is None:
        errors.append("unverifiable contract field: dimension (not returned by API)")
    elif actual_dim != DIMENSION:
        errors.append(f"dimension mismatch: expected {DIMENSION}, got {actual_dim}")

    # cloud / region
    serverless = getattr(spec, "serverless", None) if spec else None
    if serverless is not None:
        actual_cloud = str(getattr(serverless, "cloud", "")).lower()
        actual_region = str(getattr(serverless, "region", "")).lower()
        if actual_cloud != CLOUD.lower():
            errors.append(f"cloud mismatch: expected '{CLOUD}', got '{actual_cloud}'")
        if actual_region != REGION.lower():
            errors.append(f"region mismatch: expected '{REGION}', got '{actual_region}'")
    else:
        errors.append(
            "unverifiable contract field: cloud/region (serverless spec not returned by API)"
        )

    return errors


def get_index_info(pc: Any, name: str) -> dict[str, Any]:
    """Return a structured summary of the index including full normalized contract."""
    try:
        info = pc.describe_index(name)
    except Exception as e:
        return {"error": str(e), "name": name}

    spec = getattr(info, "spec", None)
    embed_cfg = getattr(spec, "embed", None) if spec else None
    if embed_cfg is None:
        integrated = getattr(spec, "integrated", None) if spec else None
        embed_cfg = getattr(integrated, "embed", None) if integrated else None

    serverless = getattr(spec, "serverless", None) if spec else None

    actual_write = getattr(embed_cfg, "write_parameters", None) if embed_cfg else None
    actual_read = getattr(embed_cfg, "read_parameters", None) if embed_cfg else None

    return {
        "name": name,
        "status": getattr(info, "status", {}).get("state")
        if hasattr(info, "status")
        else "unknown",
        "embed_model": getattr(embed_cfg, "model", "unknown") if embed_cfg else "unknown",
        "dimension": getattr(info, "dimension", "unknown"),
        "metric": getattr(embed_cfg, "metric", "unknown") if embed_cfg else "unknown",
        "field_map": getattr(embed_cfg, "field_map", {}) if embed_cfg else {},
        "write_parameters": dict(actual_write) if actual_write is not None else "unknown",
        "read_parameters": dict(actual_read) if actual_read is not None else "unknown",
        "cloud": str(getattr(serverless, "cloud", "unknown")).lower() if serverless else "unknown",
        "region": str(getattr(serverless, "region", "unknown")).lower()
        if serverless
        else "unknown",
        "text_record_field": TEXT_RECORD_FIELD,
        "canonical_contract": canonical_contract(),
    }
