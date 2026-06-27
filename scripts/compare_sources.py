from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from citation_matcher.config import DEFAULT_ELIBRARY_SEEDS_CSV
from citation_matcher.benchmark.sources import (
    BENCHMARK_SOURCES,
    build_language_leaderboard,
    build_native_vs_model_table,
    default_cache_path,
    load_seed_articles,
    run_unified_benchmark,
    summarize_unified_benchmark,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

LANG_LABELS = {"en": "Англоязычные", "ru": "Русскоязычные", "all": "Все"}
RANKER_LABELS = {
    "crossref": "Crossref",
    "openalex": "OpenAlex",
    "cyberleninka": "CyberLeninka",
    "elibrary": "eLibrary",
    "multi": "Multi (слияние)",
    "matcher": "Matcher (multi+ML)",
    "model:crossref": "ML @ Crossref",
    "model:openalex": "ML @ OpenAlex",
    "model:cyberleninka": "ML @ CyberLeninka",
    "model:elibrary": "ML @ eLibrary",
}


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def _label_ranker(ranker: str) -> str:
    return RANKER_LABELS.get(ranker, ranker)


def _print_table(table: pd.DataFrame, title: str) -> None:
    if table.empty:
        return
    print(f"\n{title}\n")
    print(table.to_string(index=False))
    print()


def print_language_leaderboard(summary: pd.DataFrame, lang: str) -> None:
    board = build_language_leaderboard(summary, lang)
    if board.empty:
        return
    table = board.copy()
    table["ranker"] = table["ranker"].map(_label_ranker)
    table["top1"] = table["top1"].map(_pct)
    table["top3"] = table["top3"].map(_pct)
    table["recall@10"] = table["recall@10"].map(_pct)
    table["mrr"] = table["mrr"].map(lambda v: f"{v:.3f}")
    table = table.rename(
        columns={
            "place": "#",
            "ranker": "Ранжировщик",
            "top1": "Top-1",
            "top3": "Top-3",
            "recall@10": "Recall@10",
            "mrr": "MRR",
            "found": "Найдено",
            "count": "Запросов",
        }
    )
    _print_table(table, f"=== {LANG_LABELS.get(lang, lang)}: кто лучше ===")


def print_native_vs_model(summary: pd.DataFrame, lang: str) -> None:
    table = build_native_vs_model_table(summary, lang)
    if table.empty:
        return
    if "Источник" in table.columns:
        table["Источник"] = table["Источник"].map(
            lambda value: _label_ranker(str(value))
            if value != "matcher (multi+ML)"
            else "Matcher (multi+ML)"
        )
    _print_table(table, f"=== {LANG_LABELS.get(lang, lang)}: native vs модель ===")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare bibliographic sources and the ML ranker on a shared article pool."
        )
    )
    parser.add_argument(
        "--articles",
        choices=("bilingual", "crossref", "cyberleninka", "elibrary", "mixed"),
        default="bilingual",
        help="Seed pool (default: bilingual EN+RU).",
    )
    parser.add_argument(
        "--csv",
        type=str,
        help="CSV with Russian articles (default: data/seeds/elibrary_data_fixed.csv).",
    )
    parser.add_argument("--limit", type=int, default=30, help="Total seed articles.")
    parser.add_argument("--english", type=int, default=None, help="EN article count.")
    parser.add_argument("--russian", type=int, default=None, help="RU article count.")
    parser.add_argument("--rows", type=int, default=10, help="Candidates per source.")
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=BENCHMARK_SOURCES,
        default=list(BENCHMARK_SOURCES),
        help="Sources to compare.",
    )
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="Skip ML ranker evaluation.",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default=str(default_cache_path()),
        help="CSV path for raw benchmark results.",
    )
    parser.add_argument("--refresh", action="store_true", help="Re-run live searches.")
    parser.add_argument("--output-summary", type=str, default="", help="Save summary CSV.")
    args = parser.parse_args(argv)

    cache_path = Path(args.cache)
    sources = tuple(args.sources)
    include_model = not args.no_model

    if cache_path.exists() and not args.refresh:
        logging.info("Loading cached benchmark from %s", cache_path)
        raw = pd.read_csv(cache_path)
    else:
        csv_path = Path(args.csv) if args.csv else None
        if csv_path is None and args.articles == "elibrary":
            csv_path = DEFAULT_ELIBRARY_SEEDS_CSV
        articles = load_seed_articles(
            args.articles,
            args.limit,
            csv_path=csv_path,
            english_count=args.english,
            russian_count=args.russian,
        )
        if not articles:
            print("No seed articles loaded.", file=sys.stderr)
            return 1
        en_count = sum(1 for a in articles if a.get("lang") == "en")
        ru_count = sum(1 for a in articles if a.get("lang") == "ru")
        logging.info(
            "Benchmark: %d articles (%d EN, %d RU) x 5 errors x %d sources%s",
            len(articles),
            en_count,
            ru_count,
            len(sources),
            " + model" if include_model else "",
        )
        raw = run_unified_benchmark(
            articles,
            sources=sources,
            rows=args.rows,
            include_model=include_model,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        raw.to_csv(cache_path, index=False)
        logging.info("Saved raw results to %s", cache_path)

    summary = summarize_unified_benchmark(
        raw, sources=sources, include_model=include_model
    )

    print("\n=== Общий пул: одинаковые запросы, все источники + модель ===")
    for lang in ("all", "en", "ru"):
        print_language_leaderboard(summary, lang)
        if include_model:
            print_native_vs_model(summary, lang)

    if args.output_summary:
        out = Path(args.output_summary)
        out.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out, index=False)
        logging.info("Saved summary to %s", out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
