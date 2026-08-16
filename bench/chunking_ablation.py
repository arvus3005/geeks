"""Offline chunking strategy ablation benchmark.

Evaluates passage_native, fixed_token_overlap, sentence_aware, and a simple
parent-child strategy using a pinned validation-split sample of MSMARCO-XI.

Evaluation labels (is_selected, query_type, etc.) are kept in a separate
eval-only artifact directory and are NEVER imported by production code.

Usage:
    uv run python bench/chunking_ablation.py \\
        --dataset-revision bf5cdc1f26e581e519018e434db14edd1b77602b \\
        --tokenizer-revision 3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3 \\
        --config-lang hi \\
        --max-rows 200 \\
        --seed 42

Results are written to:
    artifacts/reports/chunking_ablation_<timestamp>.json
    artifacts/reports/chunking_ablation_<timestamp>.md

NOTE: MRR/Recall are OFFLINE PROXY metrics computed by checking whether the
gold-selected passage (is_selected=1) appears among the top-k chunks returned
by BM25-style token overlap ranking.  These are NOT live Pinecone metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

STRATEGIES = ["passage_native", "sentence_aware", "fixed_token_overlap", "parent_child"]
MODEL_INPUT_LIMIT = 507


def _get_tokenizer(revision: str | None):
    try:
        from hhgoa_rag.ingestion.tokenizer import get_tokenizer

        return get_tokenizer(revision=revision)
    except Exception as e:
        logger.warning("Could not load real tokenizer (%s); using whitespace approx.", e)
        return None


class _FallbackTokenizer:
    fingerprint = "whitespace-approx"

    def count_tokens(self, text: str, add_prefix: bool = True) -> int:
        return len(text.split()) + 2


def _count_tokens(tok, text: str) -> int:
    if tok is None:
        return len(text.split()) + 2
    try:
        return tok.count_tokens(text)
    except Exception:
        return len(text.split()) + 2


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:40]


def _chunk_passage(text: str, strategy: str, passage_id: str) -> list[str]:
    """Chunk text with given strategy. Returns list of chunk texts."""
    if strategy == "parent_child":
        # Simple parent-child: return full passage + sentence-aware children
        from hhgoa_rag.ingestion.chunkers import SentenceAwareChunker

        children = SentenceAwareChunker().chunk(text, passage_id)
        return [text] + [c.text for c in children if c.text.strip()]

    from hhgoa_rag.ingestion.chunkers import get_chunker

    chunker = get_chunker(strategy)
    chunks = chunker.chunk(text, passage_id)
    return [c.text for c in chunks if c.text.strip()]


def _token_overlap_score(query_text: str, chunk_text: str) -> float:
    """Simple BM25-inspired token overlap score for offline proxy ranking."""
    q_toks = set(query_text.lower().split())
    c_toks = chunk_text.lower().split()
    if not q_toks or not c_toks:
        return 0.0
    hits = sum(1 for t in c_toks if t in q_toks)
    return hits / (math.log(len(c_toks) + 1) + 1)


def _evaluate_strategy(
    strategy: str,
    rows: list[dict],
    tok,
    config_lang: str,
) -> dict:
    """Evaluate one strategy on the sampled rows. Returns per-strategy metrics."""
    t0 = time.perf_counter()

    total_passages = 0
    total_chunks = 0
    rejected_passages = 0
    dup_ids: set[str] = set()
    dup_id_count = 0
    dup_text_hashes: dict[str, int] = {}
    dup_text_count = 0
    token_lengths: list[int] = []
    meta_bytes: list[int] = []
    planned_requests = 0

    # Evaluation metrics (offline proxy using is_selected labels)
    reciprocal_ranks: list[float] = []
    recall_at_5: list[float] = []
    recall_at_10: list[float] = []
    recall_at_20: list[float] = []

    per_lang_counts: dict[str, int] = {}

    for row in rows:
        passages = row.get("passages", {})
        passage_texts = passages.get("passage_text", []) if isinstance(passages, dict) else []
        is_selected_list = passages.get("is_selected", []) if isinstance(passages, dict) else []
        query_text = row.get("query", "")  # eval-only label
        lang = row.get("_lang", config_lang)
        per_lang_counts[lang] = per_lang_counts.get(lang, 0) + 1

        for pi, ptext in enumerate(passage_texts):
            if not isinstance(ptext, str) or not ptext.strip():
                continue
            total_passages += 1
            pid = _content_hash(ptext)

            try:
                chunks = _chunk_passage(ptext, strategy, pid)
            except Exception as e:
                logger.debug("Chunk error for strategy %s: %s", strategy, e)
                rejected_passages += 1
                continue

            # Filter oversized chunks
            valid_chunks = []
            for c in chunks:
                tl = _count_tokens(tok, c)
                if tl <= MODEL_INPUT_LIMIT:
                    valid_chunks.append((c, tl))
                else:
                    rejected_passages += 1

            if not valid_chunks:
                rejected_passages += 1
                continue

            for chunk_text, tl in valid_chunks:
                total_chunks += 1
                token_lengths.append(tl)
                chunk_id = f"{pid}_{_content_hash(chunk_text)[:8]}"
                if chunk_id in dup_ids:
                    dup_id_count += 1
                dup_ids.add(chunk_id)

                th = _content_hash(chunk_text)
                if th in dup_text_hashes:
                    dup_text_count += 1
                dup_text_hashes[th] = dup_text_hashes.get(th, 0) + 1

                # Rough metadata bytes estimate
                meta = {"language": lang, "chunk_text": chunk_text[:100]}
                meta_bytes.append(len(json.dumps(meta, ensure_ascii=False).encode("utf-8")))

            # Offline proxy evaluation: score chunks against query, rank
            if query_text and is_selected_list and pi < len(is_selected_list):
                gold_is_selected = int(is_selected_list[pi]) == 1
                if gold_is_selected:
                    # All chunks for this passage vs. chunks from other passages
                    all_scored: list[tuple[float, bool]] = []
                    for chunk_text, _ in valid_chunks:
                        score = _token_overlap_score(query_text, chunk_text)
                        all_scored.append((score, True))

                    # Add negative distractor chunks from other passages
                    for opi, optext in enumerate(passage_texts):
                        if opi == pi or not isinstance(optext, str):
                            continue
                        neg_chunks = _chunk_passage(optext, strategy, _content_hash(optext))
                        for nc in neg_chunks[:3]:
                            all_scored.append((_token_overlap_score(query_text, nc), False))

                    all_scored.sort(key=lambda x: x[0], reverse=True)
                    gold_rank = None
                    for rank, (_, is_gold) in enumerate(all_scored, 1):
                        if is_gold:
                            gold_rank = rank
                            break

                    if gold_rank is not None:
                        reciprocal_ranks.append(1.0 / gold_rank)
                        recall_at_5.append(1.0 if gold_rank <= 5 else 0.0)
                        recall_at_10.append(1.0 if gold_rank <= 10 else 0.0)
                        recall_at_20.append(1.0 if gold_rank <= 20 else 0.0)

        # Each row = one planned Pinecone request (worst case)
        planned_requests += max(1, total_chunks // 96)

    build_time_s = time.perf_counter() - t0

    def _pct(vals: list[int | float], p: float) -> float:
        if not vals:
            return 0.0
        s = sorted(vals)
        return s[int(len(s) * p)]

    def _mean(vals: list[int | float]) -> float:
        return statistics.mean(vals) if vals else 0.0

    expansion_factor = total_chunks / max(total_passages, 1)
    projected_storage_bytes = total_chunks * 1024 * 4  # rough: 1024-dim float32

    mrr10 = _mean(reciprocal_ranks[:10]) if reciprocal_ranks else None
    r5 = _mean(recall_at_5) if recall_at_5 else None
    r10 = _mean(recall_at_10) if recall_at_10 else None
    r20 = _mean(recall_at_20) if recall_at_20 else None

    return {
        "strategy": strategy,
        "passages_evaluated": total_passages,
        "chunks_produced": total_chunks,
        "expansion_factor": round(expansion_factor, 3),
        "rejected_passages": rejected_passages,
        "duplicate_ids": dup_id_count,
        "duplicate_texts": dup_text_count,
        "token_p50": _pct(token_lengths, 0.50),
        "token_p70": _pct(token_lengths, 0.70),
        "token_p95": _pct(token_lengths, 0.95),
        "token_p100": max(token_lengths) if token_lengths else 0,
        "token_mean": round(_mean(token_lengths), 1),
        "meta_bytes_mean": round(_mean(meta_bytes), 1),
        "meta_bytes_max": max(meta_bytes) if meta_bytes else 0,
        "planned_pinecone_requests": planned_requests,
        "projected_storage_bytes": projected_storage_bytes,
        "build_time_s": round(build_time_s, 3),
        "mrr_at_10_proxy": round(mrr10, 4) if mrr10 is not None else None,
        "recall_at_5_proxy": round(r5, 4) if r5 is not None else None,
        "recall_at_10_proxy": round(r10, 4) if r10 is not None else None,
        "recall_at_20_proxy": round(r20, 4) if r20 is not None else None,
        "per_language_counts": per_lang_counts,
        "proxy_label": "OFFLINE_TOKEN_OVERLAP_PROXY",
    }


def _load_eval_sample(
    config_lang: str,
    dataset_revision: str | None,
    tokenizer_revision: str | None,
    max_rows: int,
    seed: int,
    split: str,
) -> list[dict]:
    """Load a pinned sample from the validation split.

    Evaluation labels (query, is_selected, etc.) are loaded here for offline
    proxy evaluation ONLY.  This function MUST NOT be called from production code.
    """
    logger.info(
        "Loading eval sample: config_lang=%s split=%s max_rows=%d", config_lang, split, max_rows
    )
    try:
        from datasets import load_dataset

        ds = load_dataset(
            "ai4bharat/MSMARCO-XI",
            config_lang,
            split=split,
            revision=dataset_revision,
            trust_remote_code=False,
        )
        # Stable sample using seed
        import random

        rng = random.Random(seed)
        indices = rng.sample(range(len(ds)), min(max_rows, len(ds)))
        rows = [dict(ds[i]) for i in sorted(indices)]
        for row in rows:
            row["_lang"] = config_lang
        logger.info("Loaded %d rows", len(rows))
        return rows
    except Exception as e:
        logger.warning("Could not load HF dataset: %s — using empty sample", e)
        return []


def _write_markdown_report(results: list[dict], path: Path) -> None:
    lines = [
        "# Chunking Strategy Ablation Report",
        "",
        f"_Generated: {datetime.now(UTC).isoformat()}_",
        "",
        "> **Note**: MRR/Recall figures are OFFLINE PROXY metrics using token-overlap ranking,",
        "> not live Pinecone vector search results.",
        "",
        "## Summary",
        "",
        "| Strategy | Passages | Chunks | Expansion | Token P95 | MRR@10 (proxy) | R@10 (proxy) | Build(s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        mrr = f"{r['mrr_at_10_proxy']:.4f}" if r["mrr_at_10_proxy"] is not None else "N/A"
        r10 = f"{r['recall_at_10_proxy']:.4f}" if r["recall_at_10_proxy"] is not None else "N/A"
        lines.append(
            f"| {r['strategy']} | {r['passages_evaluated']} | {r['chunks_produced']} "
            f"| {r['expansion_factor']} | {r['token_p95']} | {mrr} | {r10} | {r['build_time_s']} |"
        )

    lines += [
        "",
        "## Detailed Metrics",
        "",
    ]
    for r in results:
        lines.append(f"### {r['strategy']}")
        lines.append("")
        for k, v in r.items():
            if k not in ("strategy", "proxy_label", "per_language_counts"):
                lines.append(f"- **{k}**: {v}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Offline chunking strategy ablation")
    p.add_argument("--dataset-revision", default=None)
    p.add_argument("--tokenizer-revision", default=None)
    p.add_argument("--config-lang", default="hi", choices=["hi", "bn", "en"])
    p.add_argument("--split", default="validation", choices=["train", "validation"])
    p.add_argument("--max-rows", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/reports"))
    args = p.parse_args()

    tok = _get_tokenizer(args.tokenizer_revision)
    if tok is None:
        tok = _FallbackTokenizer()
        logger.warning("Using whitespace approximation tokenizer (real tokenizer unavailable)")

    rows = _load_eval_sample(
        config_lang=args.config_lang,
        dataset_revision=args.dataset_revision,
        tokenizer_revision=args.tokenizer_revision,
        max_rows=args.max_rows,
        seed=args.seed,
        split=args.split,
    )

    if not rows:
        logger.warning("No eval rows loaded — generating stub report with empty metrics.")

    results = []
    for strategy in STRATEGIES:
        logger.info("Evaluating strategy: %s", strategy)
        result = _evaluate_strategy(strategy, rows, tok, config_lang=args.config_lang)
        results.append(result)
        logger.info(
            "  %s: %d passages → %d chunks (expansion=%.2f)",
            strategy,
            result["passages_evaluated"],
            result["chunks_produced"],
            result["expansion_factor"],
        )

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"chunking_ablation_{timestamp}.json"
    md_path = args.output_dir / f"chunking_ablation_{timestamp}.md"

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "config_lang": args.config_lang,
        "split": args.split,
        "max_rows": args.max_rows,
        "seed": args.seed,
        "dataset_revision": args.dataset_revision,
        "tokenizer_revision": args.tokenizer_revision,
        "tokenizer_fingerprint": getattr(tok, "fingerprint", "unknown"),
        "strategies_evaluated": STRATEGIES,
        "results": results,
        "proxy_note": (
            "MRR/Recall are OFFLINE PROXY metrics using BM25-style token overlap ranking. "
            "They are NOT live Pinecone vector search results."
        ),
    }

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown_report(results, md_path)

    logger.info("Reports written:")
    logger.info("  JSON : %s", json_path)
    logger.info("  MD   : %s", md_path)

    # Print summary table
    print("\nChunking Ablation Summary:")
    print(
        f"{'Strategy':<25} {'Passages':>8} {'Chunks':>7} {'Expansion':>10} "
        f"{'P95tok':>7} {'MRR@10':>8} {'R@10':>7}"
    )
    print("-" * 75)
    for r in results:
        mrr = f"{r['mrr_at_10_proxy']:.4f}" if r["mrr_at_10_proxy"] is not None else "N/A"
        r10 = f"{r['recall_at_10_proxy']:.4f}" if r["recall_at_10_proxy"] is not None else "N/A"
        print(
            f"{r['strategy']:<25} {r['passages_evaluated']:>8} {r['chunks_produced']:>7} "
            f"{r['expansion_factor']:>10.3f} {r['token_p95']:>7} {mrr:>8} {r10:>7}"
        )
    print("\n[OFFLINE PROXY — token-overlap ranking, not live Pinecone vector search]\n")


if __name__ == "__main__":
    main()
