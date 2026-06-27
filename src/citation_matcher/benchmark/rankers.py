from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Any

import pandas as pd

from citation_matcher.config import DATASET_SEED, DEFAULT_ROWS, REPORTS_DIR
from citation_matcher.dataset import ERROR_TYPE_STRINGS, ErrorType
from citation_matcher.matcher import match_citation
from citation_matcher.search import (
    crossref_search,
    cyberleninka_search,
    get_articles_crossref,
    get_articles_cyberleninka,
)
from citation_matcher.benchmark.sources import (
    BENCHMARK_ERROR_TYPES,
    _metrics_from_ranks,
    _summary_to_candidate,
    candidate_matches_seed,
    find_match_rank,
    generate_unified_query,
    tag_article,
)

logger = logging.getLogger(__name__)

RANKER_COLUMNS = (
    ("crossref", "crossref_rank"),
    ("cyberleninka", "cyberleninka_rank"),
    ("matcher", "matcher_rank"),
)

ERROR_TYPE_ORDER = [
    "No errors",
    "Typo in title error",
    "No author error",
    "No year error",
    "Title with dropped word error",
]

ERROR_LABELS_RU = {
    "No errors": "Без ошибок",
    "Typo in title error": "Опечатка в заголовке",
    "No author error": "Без автора",
    "No year error": "Без года",
    "Title with dropped word error": "Пропущенное слово",
    "all": "Все",
}

RANKER_LABELS = {
    "crossref": "Crossref",
    "cyberleninka": "CyberLeninka",
    "matcher": "Matcher (/match)",
}


def load_random_articles(count: int, *, seed: int = DATASET_SEED) -> list[dict[str, Any]]:
    """Random sample from Crossref + CyberLeninka seed pools."""
    pool_size = max(count * 3, 30)
    pool = [tag_article(a) for a in get_articles_crossref(pool_size)]
    pool.extend(tag_article(a) for a in get_articles_cyberleninka(pool_size))
    random.seed(seed)
    random.shuffle(pool)
    seen_titles: set[str] = set()
    selected: list[dict[str, Any]] = []
    for article in pool:
        key = str(article.get("title", "")).lower()
        if not key or key in seen_titles:
            continue
        seen_titles.add(key)
        selected.append(article)
        if len(selected) >= count:
            break
    return selected


def find_matcher_rank(query: str, seed: dict[str, Any], *, rows: int) -> int:
    result = match_citation(query, rows=rows)
    if result.error:
        return -1
    ranked = [result.best_match] + (result.alternatives or [])
    for rank, summary in enumerate(ranked, start=1):
        if summary and candidate_matches_seed(_summary_to_candidate(summary), seed):
            return rank
    return -1


def run_ranker_benchmark(
    articles: list[dict[str, Any]],
    *,
    rows: int = DEFAULT_ROWS,
    error_types: tuple[ErrorType, ...] = BENCHMARK_ERROR_TYPES,
) -> pd.DataFrame:
    random.seed(DATASET_SEED)
    rows_out: list[dict[str, Any]] = []

    for error_type in error_types:
        for article in articles:
            query = generate_unified_query(article, error_type)
            crossref_candidates = crossref_search(query, rows=rows)
            cyberleninka_candidates = cyberleninka_search(query, rows=rows)
            time.sleep(0.15)

            rows_out.append(
                {
                    "article_id": article["id"],
                    "article_source": article["source"],
                    "lang": article.get("lang"),
                    "title": article["title"],
                    "query": query,
                    "error_type": ERROR_TYPE_STRINGS[error_type],
                    "crossref_rank": find_match_rank(crossref_candidates, article),
                    "cyberleninka_rank": find_match_rank(
                        cyberleninka_candidates, article
                    ),
                    "matcher_rank": find_matcher_rank(query, article, rows=rows),
                }
            )

    return pd.DataFrame(rows_out)


def summarize_ranker_benchmark(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    error_types = [e for e in ERROR_TYPE_ORDER if e in set(frame["error_type"])] + [
        e
        for e in frame["error_type"].unique()
        if e not in ERROR_TYPE_ORDER
    ]
    groups = error_types + ["all"]

    for error_type in groups:
        group = frame if error_type == "all" else frame[frame["error_type"] == error_type]
        if group.empty:
            continue
        for ranker_name, column in RANKER_COLUMNS:
            if column not in group.columns:
                continue
            rows.append(
                {
                    "error_type": error_type,
                    "ranker": ranker_name,
                    **_metrics_from_ranks(group[column].tolist()),
                }
            )
    return pd.DataFrame(rows)


def build_metric_pivot_table(
    summary: pd.DataFrame, metric: str, *, metric_label: str
) -> pd.DataFrame:
    """Compact table: rows = error types, columns = rankers."""
    rows: list[dict[str, object]] = []
    error_types = [e for e in ERROR_TYPE_ORDER if e in set(summary["error_type"])]
    if "all" in summary["error_type"].values:
        error_types.append("all")

    for error_type in error_types:
        block = summary[summary["error_type"] == error_type]
        row: dict[str, object] = {
            "Ошибка": ERROR_LABELS_RU.get(error_type, error_type),
        }
        count = 0
        for ranker_name, _ in RANKER_COLUMNS:
            part = block[block["ranker"] == ranker_name]
            if part.empty:
                continue
            metrics = part.iloc[0]
            count = int(metrics["count"])
            row[RANKER_LABELS[ranker_name]] = f"{100 * metrics[metric]:.1f}%"
        row["n"] = count
        rows.append(row)
    return pd.DataFrame(rows)


def build_comparison_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Wide table: one row per error type, columns per ranker."""
    rows: list[dict[str, object]] = []
    error_types = [e for e in ERROR_TYPE_ORDER if e in set(summary["error_type"])]
    if "all" in summary["error_type"].values:
        error_types.append("all")

    for error_type in error_types:
        block = summary[summary["error_type"] == error_type]
        row: dict[str, object] = {
            "Ошибка": ERROR_LABELS_RU.get(error_type, error_type),
        }
        count = 0
        for ranker_name, _ in RANKER_COLUMNS:
            part = block[block["ranker"] == ranker_name]
            if part.empty:
                continue
            metrics = part.iloc[0]
            count = int(metrics["count"])
            label = RANKER_LABELS[ranker_name]
            row[f"{label} Top-1"] = f"{100 * metrics['top1']:.1f}%"
            row[f"{label} Top-3"] = f"{100 * metrics['top3']:.1f}%"
            row[f"{label} Recall@10"] = f"{100 * metrics['recall@10']:.1f}%"
        row["n"] = count
        rows.append(row)
    return pd.DataFrame(rows)


def default_cache_path() -> Path:
    return REPORTS_DIR / "ranker_benchmark.csv"
