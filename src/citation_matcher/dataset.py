from __future__ import annotations

import logging
import random
import string
import time
from collections import defaultdict
from enum import Enum
from typing import Any

import pandas as pd

from citation_matcher.config import DATASET_SEED
from citation_matcher.matcher import add_features
from citation_matcher.search import (
    get_articles_crossref,
    get_articles_cyberleninka,
    get_articles_elibrary,
    search_for_dataset,
)
from citation_matcher.util.parsing_utils import clean_html

logger = logging.getLogger(__name__)

TITLE_ONLY_SOURCES = frozenset({"cyberleninka", "elibrary"})
RATE_LIMITED_SOURCES = frozenset({"cyberleninka", "elibrary"})


class ErrorType(Enum):
    clean = 0
    no_author = 1
    typo_in_title = 2
    no_year = 3
    title_with_drop_word = 4


ERROR_TYPE_STRINGS = {
    ErrorType.no_author: "No author error",
    ErrorType.title_with_drop_word: "Title with dropped word error",
    ErrorType.no_year: "No year error",
    ErrorType.typo_in_title: "Typo in title error",
    ErrorType.clean: "No errors",
}


def _generate_typo(text: str) -> str:
    if not text:
        return text
    line = text
    alphabet = string.ascii_lowercase + "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    for _ in range(random.randint(1, 4)):
        if not line:
            break
        idx = random.randint(0, len(line) - 1)
        if random.randint(0, 1) == 0:
            chars = list(line)
            chars[idx] = random.choice(alphabet)
            line = "".join(chars)
        else:
            line = line[:idx] + line[idx + 1 :]
    return line


def _drop_word(text: str) -> str:
    words = str(text).split()
    if len(words) <= 1:
        return text
    words.pop(random.randint(0, len(words) - 1))
    return " ".join(words)


def generate_query(citation: dict[str, Any], error_type: ErrorType) -> str:
    title = citation["title"]
    author = citation.get("first_author") or ""
    year = citation.get("year") or ""

    if citation["source"] == "crossref":
        if error_type == ErrorType.no_year:
            return f"{title} {author}".strip()
        if error_type == ErrorType.no_author:
            return f"{title} {year}".strip()
        if error_type == ErrorType.title_with_drop_word:
            return f"{_drop_word(title)} {author} {year}".strip()
        if error_type == ErrorType.typo_in_title:
            return f"{_generate_typo(title)} {author} {year}".strip()
        return f"{title} {author} {year}".strip()

    if error_type == ErrorType.typo_in_title:
        return _generate_typo(title)
    if error_type == ErrorType.title_with_drop_word:
        return _drop_word(title)
    return title


def _normalize_candidate(
    source: str, match: dict[str, Any], rank: int, query: str, citation: dict[str, Any]
) -> dict[str, Any]:
    if source == "crossref":
        candidate_id = match.get("DOI")
        authors = match.get("author") or []
        first = authors[0] if authors else None
        candidate_author = (
            f"{first.get('given', '')} {first.get('family', '')}".strip()
            if isinstance(first, dict)
            else str(first or "").strip()
        ) or None
        year = None
        for field in ("published-print", "published-online", "issued"):
            if field in match:
                year = match[field]["date-parts"][0][0]
                break
        return {
            "query": query,
            "source": source,
            "candidate_id": candidate_id,
            "label": int(candidate_id == citation["id"]),
            "candidate_year": year,
            "candidate_title": clean_html((match.get("title") or [""])[0]),
            "true_id": citation["id"],
            "candidate_rank": rank,
            "candidate_author": candidate_author,
            "query_author": citation.get("first_author"),
            "journal": clean_html((match.get("container-title") or [""])[0]),
        }

    candidate_id = match.get("link")
    authors = match.get("authors") or []
    title = match.get("name") or match.get("title") or ""
    return {
        "query": query,
        "source": source,
        "candidate_id": candidate_id,
        "label": int(candidate_id == citation["id"]),
        "candidate_year": match.get("year"),
        "candidate_title": clean_html(title),
        "true_id": citation["id"],
        "candidate_rank": rank,
        "candidate_author": authors[0] if authors else None,
        "query_author": citation.get("first_author"),
        "journal": clean_html(match.get("journal")),
    }


def _check_match(
    source: str, query: str, citation: dict[str, Any]
) -> tuple[int, list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    match_rank = -1
    for rank, match in enumerate(search_for_dataset(source, query), start=1):
        row = _normalize_candidate(source, match, rank, query, citation)
        if row["label"] == 1:
            match_rank = rank
        candidates.append(row)
    if source in RATE_LIMITED_SOURCES:
        time.sleep(0.2)
    return match_rank, candidates


def build_dataset(
    crossref_count: int = 50,
    cyberleninka_count: int = 50,
    elibrary_count: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    random.seed(DATASET_SEED)

    articles = (
        get_articles_cyberleninka(cyberleninka_count)
        + get_articles_crossref(crossref_count)
        + get_articles_elibrary(elibrary_count)
    )
    logger.info("Collected %d seed articles for dataset building", len(articles))

    all_results: list[dict[str, Any]] = []
    all_ranking_rows: list[dict[str, Any]] = []

    for error_type in (
        ErrorType.clean,
        ErrorType.typo_in_title,
        ErrorType.no_author,
        ErrorType.no_year,
        ErrorType.title_with_drop_word,
    ):
        matches = defaultdict(int)
        for article in articles:
            if article["source"] in TITLE_ONLY_SOURCES and error_type in {
                ErrorType.no_author,
                ErrorType.no_year,
            }:
                continue

            query = generate_query(article, error_type)
            match_rank, candidates = _check_match(article["source"], query, article)
            all_results.append(
                {
                    "Original Citation": article,
                    "Query": query,
                    "Source": article["source"],
                    "Type of Error": ERROR_TYPE_STRINGS[error_type],
                    "Match Number": match_rank,
                    "Candidates": candidates,
                }
            )
            all_ranking_rows.extend(candidates)
            matches[match_rank] += 1

        logger.info("%s: top-1=%d", ERROR_TYPE_STRINGS[error_type], matches[1])

    results_df = pd.DataFrame(all_results)
    if not results_df.empty:
        results_df["top1"] = results_df["Match Number"] == 1
        results_df["top3"] = results_df["Match Number"].between(1, 3)
        results_df["top10"] = results_df["Match Number"].between(1, 10)
        summary_df = results_df.groupby(["Source", "Type of Error"]).agg(
            top1=("top1", "mean"),
            top3=("top3", "mean"),
            top10=("top10", "mean"),
            count=("Match Number", "count"),
        )
    else:
        summary_df = pd.DataFrame()

    ranking_df = pd.DataFrame(all_ranking_rows)
    return results_df, summary_df, ranking_df


def build_feature_dataset(
    crossref_count: int = 50,
    cyberleninka_count: int = 50,
    elibrary_count: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results_df, summary_df, ranking_df = build_dataset(
        crossref_count, cyberleninka_count, elibrary_count
    )
    return results_df, summary_df, add_features(ranking_df)
