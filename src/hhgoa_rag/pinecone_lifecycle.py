"""Pinecone index lifecycle management.

Rules:
- Never delete or recreate an existing index automatically.
- create_index_idempotent returns the existing index model if it already exists.
- validate_index checks ALL canonical contract fields and returns structured errors.
- Any field that cannot be verified produces an "unverifiable contract field" error.
- Clear actionable errors for incompatible existing indexes.

All contract constants are imported from pinecone_contract — there are no
hand-written contract dicts in this module.

Pinecone 9.1.0 SDK layout (from describe_index / IndexModel):
    info.embed         — ModelIndexEmbed (or dict) with model/field_map/metric/read_parameters/write_parameters
    info.dimension     — int (top-level)
    info.metric        — str (top-level)
    info.spec          — IndexSpec with .serverless (ServerlessSpecInfo) for cloud/region
    info.status        — dict {"ready": bool, "state": str}

    The old info.spec.embed path does NOT exist in 9.1.0.
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


# ---------------------------------------------------------------------------
# Normalization helpers — handle both SDK struct objects and dict-shaped fixtures
# ---------------------------------------------------------------------------


def _attr_or_key(obj: Any, name: str, default: Any = None) -> Any:
    """Read `name` from obj as attribute (SDK struct) or dict key."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _to_plain_dict(value: Any) -> dict | None:
    """Coerce a mapping-like (SDK struct or dict) to a plain dict, or return None."""
    if value is None:
        return None
    if isinstance(value, dict):
        return dict(value)
    # SDK struct objects may expose .items() or can be iterated
    if hasattr(value, "items"):
        return dict(value.items())
    # Last resort: try __dict__
    try:
        return {k: v for k, v in vars(value).items() if not k.startswith("_")}
    except TypeError:
        return None


def _normalize_embed(info: Any) -> dict | None:
    """Extract the embed config from an IndexModel response.

    Pinecone 9.1.0: embedded config is at info.embed (top level), NOT info.spec.embed.
    Supports both SDK ModelIndexEmbed objects and plain-dict test fixtures.

    Returns a plain dict with keys: model, field_map, metric, write_parameters, read_parameters.
    Returns None if no embed config is present.
    """
    # Primary path: info.embed (Pinecone 9.1.0 layout)
    embed = _attr_or_key(info, "embed")
    if embed is None:
        return None
    if isinstance(embed, dict):
        return embed
    # SDK struct object — coerce each field to plain Python type
    return {
        "model": _attr_or_key(embed, "model"),
        "field_map": _to_plain_dict(_attr_or_key(embed, "field_map")),
        "metric": _attr_or_key(embed, "metric"),
        "write_parameters": _to_plain_dict(_attr_or_key(embed, "write_parameters")),
        "read_parameters": _to_plain_dict(_attr_or_key(embed, "read_parameters")),
    }


def _normalize_serverless(info: Any) -> dict | None:
    """Extract cloud/region from info.spec.serverless.

    Returns {"cloud": str, "region": str} or None if unavailable.
    """
    spec = _attr_or_key(info, "spec")
    if spec is None:
        return None
    serverless = _attr_or_key(spec, "serverless")
    if serverless is None:
        return None
    return {
        "cloud": str(_attr_or_key(serverless, "cloud") or "").lower(),
        "region": str(_attr_or_key(serverless, "region") or "").lower(),
    }


def _normalize_status_ready(info: Any) -> bool | None:
    """Return the index ready flag from info.status, or None if unreadable."""
    status = _attr_or_key(info, "status")
    if status is None:
        return None
    ready = _attr_or_key(status, "ready")
    if ready is None:
        return None
    return bool(ready)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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

    Parameters that differ from canonical contract values are rejected rather
    than silently accepted, to prevent accidental noncanonical index creation.
    """
    # Reject noncanonical parameters before any API call.
    if cloud != CLOUD:
        raise ValueError(
            f"cloud {cloud!r} differs from canonical {CLOUD!r}. "
            "Pass canonical values only to prevent noncanonical index creation."
        )
    if region != REGION:
        raise ValueError(f"region {region!r} differs from canonical {REGION!r}.")
    if embed_model != MODEL:
        raise ValueError(f"embed_model {embed_model!r} differs from canonical {MODEL!r}.")

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
            field_map=dict(FIELD_MAP),
            metric=METRIC,
            write_parameters=dict(WRITE_PARAMETERS),
            read_parameters=dict(READ_PARAMETERS),
        ),
        tags=tags or {},
    )


def validate_index(pc: Any, name: str) -> list[str]:
    """Validate an existing index against ALL canonical contract fields.

    Reads the Pinecone 9.1.0 SDK response layout:
        info.embed         — embed config (top level, not info.spec.embed)
        info.dimension     — int (top level)
        info.metric        — str (top level)
        info.spec.serverless — cloud/region

    Returns list of error strings.  Any field that cannot be verified produces
    an explicit "unverifiable contract field" error rather than a silent pass.
    A correct, fully verified index returns an empty list.
    """
    try:
        info = pc.describe_index(name)
    except Exception as e:
        return [f"Could not describe index '{name}': {e}"]

    errors: list[str] = []

    # ── Embed configuration (top-level info.embed in Pinecone 9.1.0) ────────────
    embed = _normalize_embed(info)
    if embed is None:
        errors.append(
            "unverifiable contract field: embed config — index may not be an "
            "integrated-embedding index (info.embed is absent)"
        )
    else:
        # Model
        actual_model = embed.get("model")
        if actual_model is None:
            errors.append("unverifiable contract field: embed model (not returned by API)")
        elif actual_model != MODEL:
            errors.append(f"embed model mismatch: expected '{MODEL}', got '{actual_model}'")

        # field_map
        actual_field_map = embed.get("field_map")
        if actual_field_map is None:
            errors.append("unverifiable contract field: field_map (not returned by API)")
        elif actual_field_map != dict(FIELD_MAP):
            errors.append(
                f"field_map mismatch: expected {dict(FIELD_MAP)!r}, got {actual_field_map!r}"
            )

        # metric (also available at top level; use embed's copy for integrated embed check)
        actual_embed_metric = embed.get("metric")
        if actual_embed_metric is None:
            # Fall back to top-level metric
            actual_embed_metric = _attr_or_key(info, "metric")
        if actual_embed_metric is None:
            errors.append("unverifiable contract field: metric (not returned by API)")
        elif actual_embed_metric != METRIC:
            errors.append(f"metric mismatch: expected '{METRIC}', got '{actual_embed_metric}'")

        # write_parameters
        actual_write = embed.get("write_parameters")
        if actual_write is None:
            errors.append("unverifiable contract field: write_parameters (not returned by API)")
        else:
            _wp = dict(actual_write) if not isinstance(actual_write, dict) else actual_write
            if _wp != dict(WRITE_PARAMETERS):
                errors.append(
                    f"write_parameters mismatch: expected {dict(WRITE_PARAMETERS)!r}, got {_wp!r}"
                )

        # read_parameters
        actual_read = embed.get("read_parameters")
        if actual_read is None:
            errors.append("unverifiable contract field: read_parameters (not returned by API)")
        else:
            _rp = dict(actual_read) if not isinstance(actual_read, dict) else actual_read
            if _rp != dict(READ_PARAMETERS):
                errors.append(
                    f"read_parameters mismatch: expected {dict(READ_PARAMETERS)!r}, got {_rp!r}"
                )

    # ── Dimension (top-level) ────────────────────────────────────────────────────
    actual_dim = _attr_or_key(info, "dimension")
    if actual_dim is None:
        errors.append("unverifiable contract field: dimension (not returned by API)")
    elif actual_dim != DIMENSION:
        errors.append(f"dimension mismatch: expected {DIMENSION}, got {actual_dim}")

    # ── Cloud / region (info.spec.serverless) ────────────────────────────────────
    serverless = _normalize_serverless(info)
    if serverless is None:
        errors.append(
            "unverifiable contract field: cloud/region (serverless spec not returned by API)"
        )
    else:
        if serverless["cloud"] != CLOUD.lower():
            errors.append(f"cloud mismatch: expected '{CLOUD}', got '{serverless['cloud']}'")
        if serverless["region"] != REGION.lower():
            errors.append(f"region mismatch: expected '{REGION}', got '{serverless['region']}'")

    # ── Index readiness ──────────────────────────────────────────────────────────
    ready = _normalize_status_ready(info)
    if ready is None:
        errors.append("unverifiable contract field: index readiness (status not returned by API)")
    elif not ready:
        errors.append("index is not ready (status.ready is False)")

    return errors


def get_index_info(pc: Any, name: str) -> dict[str, Any]:
    """Return a structured summary of the index including full normalized contract."""
    try:
        info = pc.describe_index(name)
    except Exception as e:
        return {"error": str(e), "name": name}

    embed = _normalize_embed(info)
    serverless = _normalize_serverless(info)

    status = _attr_or_key(info, "status") or {}
    if isinstance(status, dict):
        status_state = status.get("state", "unknown")
    else:
        status_state = getattr(status, "state", "unknown")

    return {
        "name": name,
        "status": status_state,
        "embed_model": (embed or {}).get("model", "unknown"),
        "dimension": _attr_or_key(info, "dimension", "unknown"),
        "metric": _attr_or_key(info, "metric") or (embed or {}).get("metric", "unknown"),
        "field_map": (embed or {}).get("field_map", {}),
        "write_parameters": (embed or {}).get("write_parameters", "unknown"),
        "read_parameters": (embed or {}).get("read_parameters", "unknown"),
        "cloud": (serverless or {}).get("cloud", "unknown"),
        "region": (serverless or {}).get("region", "unknown"),
        "text_record_field": TEXT_RECORD_FIELD,
        "canonical_contract": canonical_contract(),
    }
