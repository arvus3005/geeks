"""Parse and validate MSMARCO-XI records into PassageOccurrences."""

from typing import Any

from ..ingestion.normalizer import content_hash, is_valid_passage, normalize_text
from .models import PassageOccurrence, RejectedRecord

# Fields that must NEVER enter production payload/embeddings
FORBIDDEN_FIELDS = {"query", "Answer", "Eng_Query", "Eng_Answer", "query_type", "is_selected"}


def parse_record(
    record: dict[str, Any],
    config_language: str,
    split: str,
    source_shard: str,
    source_row: int,
    dataset_revision: str = "unknown",
) -> tuple[list[PassageOccurrence], list[RejectedRecord]]:
    """Parse one MSMARCO-XI record into PassageOccurrences. Never includes forbidden fields."""
    occurrences: list[PassageOccurrence] = []
    rejected: list[RejectedRecord] = []

    passages_field = record.get("passages")
    if not isinstance(passages_field, dict):
        rejected.append(
            RejectedRecord(
                config_language=config_language,
                split=split,
                source_shard=source_shard,
                source_row=source_row,
                passage_position=-1,
                field="passages",
                reason="missing_or_wrong_type",
            )
        )
        return occurrences, rejected

    eng_passages = passages_field.get("English_passages")
    trans_passages = passages_field.get("Translated_passages")

    if not isinstance(eng_passages, list) or not isinstance(trans_passages, list):
        rejected.append(
            RejectedRecord(
                config_language=config_language,
                split=split,
                source_shard=source_shard,
                source_row=source_row,
                passage_position=-1,
                field="passages.lists",
                reason="missing_passage_lists",
            )
        )
        return occurrences, rejected

    if len(eng_passages) != len(trans_passages):
        rejected.append(
            RejectedRecord(
                config_language=config_language,
                split=split,
                source_shard=source_shard,
                source_row=source_row,
                passage_position=-1,
                field="passages.length_mismatch",
                reason=f"eng={len(eng_passages)} trans={len(trans_passages)}",
            )
        )
        return occurrences, rejected

    for idx, (eng_p, trans_p) in enumerate(zip(eng_passages, trans_passages)):
        # English passage
        if not isinstance(eng_p, str):
            rejected.append(
                RejectedRecord(
                    config_language=config_language,
                    split=split,
                    source_shard=source_shard,
                    source_row=source_row,
                    passage_position=idx,
                    field="passages.English_passages",
                    reason="null_or_wrong_type",
                    raw_value=str(eng_p)[:100],
                )
            )
        else:
            norm = normalize_text(eng_p)
            if len(norm) < 10:
                rejected.append(
                    RejectedRecord(
                        config_language=config_language,
                        split=split,
                        source_shard=source_shard,
                        source_row=source_row,
                        passage_position=idx,
                        field="passages.English_passages",
                        reason="too_short",
                        raw_value=eng_p[:100],
                    )
                )
            else:
                occurrences.append(
                    PassageOccurrence(
                        dataset_revision=dataset_revision,
                        config_language=config_language,
                        split=split,
                        source_shard=source_shard,
                        source_row=source_row,
                        passage_position=idx,
                        passage_language="en",
                        raw_text=eng_p,
                        normalized_text=norm,
                        content_hash=content_hash(norm),
                        is_original_english=True,
                    )
                )

        # Translated passage
        if not isinstance(trans_p, str):
            rejected.append(
                RejectedRecord(
                    config_language=config_language,
                    split=split,
                    source_shard=source_shard,
                    source_row=source_row,
                    passage_position=idx,
                    field="passages.Translated_passages",
                    reason="null_or_wrong_type",
                    raw_value=str(trans_p)[:100],
                )
            )
        else:
            norm = normalize_text(trans_p)
            if len(norm) < 10:
                rejected.append(
                    RejectedRecord(
                        config_language=config_language,
                        split=split,
                        source_shard=source_shard,
                        source_row=source_row,
                        passage_position=idx,
                        field="passages.Translated_passages",
                        reason="too_short",
                        raw_value=trans_p[:100],
                    )
                )
            else:
                occurrences.append(
                    PassageOccurrence(
                        dataset_revision=dataset_revision,
                        config_language=config_language,
                        split=split,
                        source_shard=source_shard,
                        source_row=source_row,
                        passage_position=idx,
                        passage_language=config_language,
                        raw_text=trans_p,
                        normalized_text=norm,
                        content_hash=content_hash(norm),
                        is_original_english=False,
                    )
                )

    return occurrences, rejected
