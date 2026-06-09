from __future__ import annotations

import re
from typing import Any

from rapidfuzz import fuzz

from citation_matcher.text import clean_html

FEATURE_COLUMNS = [
    "title_similarity",
    "first_author_similarity",
    "crossref_score",
    "candidate_rank",
    "year_difference",
    "title_token_set_similarity",
]


def extract_first_author(item: dict[str, Any]) -> str | None:
    authors = item.get("author")
    if not authors:
        return None

    first = authors[0]
    given = first.get("given", "")
    family = first.get("family", "")
    full_name = f"{given} {family}".strip()
    return full_name or None


def extract_year_from_query(query: str) -> int | None:
    match = re.search(r"\b(19|20)\d{2}\b", str(query))
    if match:
        return int(match.group())
    return None


def extract_year(item: dict[str, Any]) -> int | None:
    for field in ("published-print", "published-online", "issued"):
        if field in item:
            return item[field]["date-parts"][0][0]
    year = item.get("year")
    if year is not None:
        return int(year)
    return None


def author_similarity(query: str, candidate_author: str | None) -> float:
    if not candidate_author:
        return 0.0
    return max(
        fuzz.token_sort_ratio(query, candidate_author),
        fuzz.partial_ratio(candidate_author, query),
    )


def build_features(query: str, candidate: dict[str, Any]) -> dict[str, float | int]:
    candidate_title = clean_html(candidate.get("title", [""])[0])
    candidate_year = extract_year(candidate)
    query_year = extract_year_from_query(query)
    candidate_author = extract_first_author(candidate)

    return {
        "title_similarity": fuzz.token_sort_ratio(query, candidate_title),
        "title_token_set_similarity": fuzz.token_set_ratio(query, candidate_title),
        "first_author_similarity": author_similarity(query, candidate_author),
        "crossref_score": float(candidate.get("score", 0) or 0),
        "candidate_rank": int(candidate["rank"]),
        "year_difference": abs(query_year - candidate_year)
        if query_year and candidate_year
        else 999,
    }
