"""Build a small, offline-eval-only benchmark query set from real MSMARCO-XI queries.

Pulls `query` (native hi/bn) and `Eng_Query` (en) text from the HuggingFace
*validation* split — disjoint from the `train` split that was indexed into
Pinecone — so benchmark queries never overlap the indexed corpus.

Per CLAUDE.md, `query` / `Eng_Query` must never enter production vector store
payloads, dense/sparse vector inputs, or reranking features. They are used
here ONLY as benchmark input text (i.e. what a user would type/say), which is
the explicit "offline evaluation" carve-out — nothing from this file is ever
written back into Pinecone.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DATASET_REVISION = "bf5cdc1f26e581e519018e434db14edd1b77602b"
_VALIDATION_PARQUET = {
    "hi": "validation/hinval.parquet",
    "bn": "validation/benval.parquet",
}
_CACHE_DIR = ".cache/huggingface"
DEFAULT_OUTPUT = Path("artifacts/eval/bench_queries.jsonl")
DEFAULT_PER_LANGUAGE = 20


def _fetch_language_queries(config_lang: str, n: int) -> list[dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    from hhgoa_rag.pinecone_contract import DATASET_REPO

    local_path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=_VALIDATION_PARQUET[config_lang],
        revision=DATASET_REVISION,
        repo_type="dataset",
        cache_dir=_CACHE_DIR,
    )
    pf = pq.ParquetFile(local_path)
    rows: list[dict] = []
    for batch in pf.iter_batches(batch_size=min(n * 2, 200)):
        d = batch.to_pydict()
        for i in range(batch.num_rows):
            if len(rows) >= n:
                break
            native_query = (d["query"][i] or "").strip()
            eng_query = (d["Eng_Query"][i] or "").strip()
            if not native_query or not eng_query:
                continue
            rows.append(
                {
                    "query_id": d["query_id"][i],
                    "language": config_lang,
                    "text": native_query,
                }
            )
            rows.append(
                {
                    "query_id": d["query_id"][i],
                    "language": "en",
                    "text": eng_query,
                }
            )
        if len(rows) >= n:
            break
    return rows[: n * 2]


def build_query_set(
    per_language: int = DEFAULT_PER_LANGUAGE,
    output_path: Path = DEFAULT_OUTPUT,
) -> list[dict]:
    """Fetch (or reuse a cached) benchmark query set and write it to output_path."""
    if output_path.exists():
        logger.info("Reusing cached benchmark query set at %s", output_path)
        with output_path.open() as f:
            return [json.loads(line) for line in f if line.strip()]

    queries: list[dict] = []
    for config_lang in ("hi", "bn"):
        queries.extend(_fetch_language_queries(config_lang, per_language))

    if not queries:
        raise RuntimeError("No benchmark queries could be built from the validation split")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    logger.info("Wrote %d benchmark queries to %s", len(queries), output_path)
    return queries


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    qs = build_query_set()
    by_lang: dict[str, int] = {}
    for q in qs:
        by_lang[q["language"]] = by_lang.get(q["language"], 0) + 1
    print(f"Built {len(qs)} queries: {by_lang}")
