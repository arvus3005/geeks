"""Typed models for MSMARCO-XI dataset records."""

from dataclasses import dataclass

INDIC_LANGUAGE_CODES = [
    "as",
    "bn",
    "gu",
    "hi",
    "kn",
    "ml",
    "mr",
    "ne",
    "or",
    "pa",
    "sa",
    "ta",
    "te",
    "ur",
]
ALL_LANGUAGE_CODES = INDIC_LANGUAGE_CODES + ["en"]


@dataclass
class PassageOccurrence:
    """A single passage occurrence from MSMARCO-XI, ready for ingestion."""

    dataset_revision: str
    config_language: str  # one of INDIC_LANGUAGE_CODES
    split: str  # "train" or "validation"
    source_shard: str
    source_row: int
    passage_position: int
    passage_language: str  # "en" or config_language
    raw_text: str
    normalized_text: str
    content_hash: str  # hash of normalized_text
    is_original_english: bool


@dataclass
class RejectedRecord:
    """A rejected passage occurrence with structured reason."""

    config_language: str
    split: str
    source_shard: str
    source_row: int
    passage_position: int
    field: str
    reason: str
    raw_value: str | None = None


@dataclass
class DiscoveryManifest:
    """Records what was discovered about the dataset."""

    dataset_repo: str
    dataset_revision: str | None
    observed_configs: list[str]
    missing_configs: list[str]
    splits_per_config: dict[str, list[str]]
    row_counts: dict[str, dict[str, int]]  # config -> split -> count
    schema_fields: list[str]
    manifest_path: str
    discovered_at: str
