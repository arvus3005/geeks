"""Discover ai4bharat/MSMARCO-XI dataset structure."""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from .models import INDIC_LANGUAGE_CODES, DiscoveryManifest

DATASET_REPO = "ai4bharat/MSMARCO-XI"
EXPECTED_SPLITS = ["train", "validation"]


def discover_dataset(
    revision: str | None = None,
    manifest_dir: Path = Path("artifacts/manifests"),
    streaming: bool = True,
) -> DiscoveryManifest:
    """Discover MSMARCO-XI configs, splits, and schema. Writes manifest to disk."""
    try:
        from datasets import get_dataset_config_names, load_dataset
    except ImportError:
        print("ERROR: datasets package not installed", file=sys.stderr)
        sys.exit(1)

    manifest_dir.mkdir(parents=True, exist_ok=True)

    try:
        all_configs = get_dataset_config_names(DATASET_REPO, revision=revision)
    except Exception as e:
        print(f"ERROR discovering configs: {e}", file=sys.stderr)
        sys.exit(1)

    observed = sorted(all_configs)
    missing = [c for c in INDIC_LANGUAGE_CODES if c not in observed]

    splits_per_config: dict[str, list[str]] = {}
    row_counts: dict[str, dict[str, int]] = {}
    schema_fields: list[str] = []

    for config in observed[:1]:  # Check schema from first config
        for split in EXPECTED_SPLITS:
            try:
                ds = load_dataset(
                    DATASET_REPO, config, split=split, streaming=True, revision=revision
                )
                if not schema_fields:
                    schema_fields = list(next(iter(ds)).keys())
                break
            except Exception:
                continue

    for config in observed:
        splits_per_config[config] = []
        row_counts[config] = {}
        for split in EXPECTED_SPLITS:
            try:
                load_dataset(DATASET_REPO, config, split=split, streaming=True, revision=revision)
                splits_per_config[config].append(split)
                row_counts[config][split] = -1  # -1 = unknown for streaming
            except Exception:
                pass

    now = datetime.now(UTC).isoformat()
    manifest_path = str(manifest_dir / f"discovery_{now[:10].replace('-', '')}.json")

    manifest = DiscoveryManifest(
        dataset_repo=DATASET_REPO,
        dataset_revision=revision,
        observed_configs=observed,
        missing_configs=missing,
        splits_per_config=splits_per_config,
        row_counts=row_counts,
        schema_fields=schema_fields,
        manifest_path=manifest_path,
        discovered_at=now,
    )

    import dataclasses

    tmp = manifest_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(dataclasses.asdict(manifest), f, indent=2)
    Path(tmp).rename(manifest_path)

    return manifest
