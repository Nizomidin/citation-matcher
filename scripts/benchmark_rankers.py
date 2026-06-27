from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from citation_matcher.benchmark.rankers import (
    build_comparison_table,
    build_metric_pivot_table,
    default_cache_path,
    load_random_articles,
    run_ranker_benchmark,
    summarize_ranker_benchmark,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Crossref, CyberLeninka and Matcher (/match) "
            "on a random article sample, broken down by error type."
        )
    )
    parser.add_argument("-n", "--count", type=int, default=20, help="Random articles.")
    parser.add_argument("--rows", type=int, default=10, help="Candidates per search.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--cache",
        type=str,
        default=str(default_cache_path()),
        help="CSV for raw per-query results.",
    )
    parser.add_argument("--refresh", action="store_true", help="Re-run live searches.")
    parser.add_argument("--output", type=str, default="", help="Save summary CSV.")
    args = parser.parse_args(argv)

    cache_path = Path(args.cache)

    if cache_path.exists() and not args.refresh:
        logging.info("Loading cached results from %s", cache_path)
        raw = pd.read_csv(cache_path)
    else:
        articles = load_random_articles(
            args.count,
            **({"seed": args.seed} if args.seed is not None else {}),
        )
        if not articles:
            print("Failed to load articles.", file=sys.stderr)
            return 1
        logging.info(
            "Benchmarking %d random articles x 5 error types x 3 rankers",
            len(articles),
        )
        raw = run_ranker_benchmark(articles, rows=args.rows)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        raw.to_csv(cache_path, index=False)
        logging.info("Saved raw results to %s", cache_path)

    summary = summarize_ranker_benchmark(raw)
    top1_table = build_metric_pivot_table(summary, "top1", metric_label="Top-1")
    top3_table = build_metric_pivot_table(summary, "top3", metric_label="Top-3")
    full_table = build_comparison_table(summary)

    print("\n=== Top-1 (доля запросов, где правильная статья на 1-м месте) ===\n")
    print(top1_table.to_string(index=False))
    print("\n=== Top-3 (правильная статья в первых трёх) ===\n")
    print(top3_table.to_string(index=False))
    print("\n=== Полная таблица (Top-1 / Top-3 / Recall@10) ===\n")
    print(full_table.to_string(index=False))
    print()

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out, index=False)
        logging.info("Saved summary to %s", out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
