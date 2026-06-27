from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from rapidfuzz import fuzz

from citation_matcher.config import DEFAULT_MODEL_PATH, DEFAULT_ROWS
from citation_matcher.search import multi_search
from citation_matcher.util.parsing_utils import (
    clean_html,
    compute_script_overlap,
    extract_year_from_query,
    query_word_count,
)

FEATURE_COLUMNS = [
    "title_similarity",
    "title_token_set_similarity",
    "first_author_similarity",
    "year_difference",
    "journal_similarity",
    "word_count",
    "script_overlap",
]

EXACT_MATCH_THRESHOLD = 95.0
YEAR_DIFFERENCE_MISSING = 999
SOURCE_RANK_MISSING = 999


def prepare_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = _enrich_missing_features(df)
    frame["source_rank"] = frame["candidate_rank"]
    if "source" in frame.columns:
        frame["source_crossref"] = (frame["source"] == "crossref").astype(int)
        frame["source_openalex"] = (frame["source"] == "openalex").astype(int)
        frame["source_cyberleninka"] = (frame["source"] == "cyberleninka").astype(int)
        frame["source_elibrary"] = (frame["source"] == "elibrary").astype(int)
    else:
        frame["source_crossref"] = 1
        frame["source_openalex"] = 0
        frame["source_cyberleninka"] = 0
        frame["source_elibrary"] = 0
    return frame


def extract_first_author(item: dict[str, Any]) -> str | None:
    authors = item.get("author")
    if not authors:
        return None
    first = authors[0]
    if isinstance(first, str):
        return first.strip() or None
    given = first.get("given", "")
    family = first.get("family", "")
    full_name = f"{given} {family}".strip()
    return full_name or None


def extract_year(item: dict[str, Any]) -> int | None:
    for field_name in ("published-print", "published-online", "issued"):
        if field_name in item:
            return item[field_name]["date-parts"][0][0]
    year = item.get("year")
    return int(year) if year is not None else None


def author_similarity(query: str, candidate_author: str | None) -> float:
    if not candidate_author:
        return 0.0
    return max(
        fuzz.token_sort_ratio(query, candidate_author),
        fuzz.partial_ratio(candidate_author, query),
    )


def compute_author_feature(
    query: str,
    candidate_author: str | None,
    *,
    query_author: str | None = None,
) -> float:
    if not candidate_author:
        return 0.0
    anchor = (query_author or "").strip() or query
    return author_similarity(anchor, candidate_author)


def compute_year_difference(
    query_year: int | None, candidate_year: int | None
) -> int:
    if query_year is None or candidate_year is None:
        return YEAR_DIFFERENCE_MISSING
    return abs(int(query_year) - int(candidate_year))


def compute_journal_similarity(query: str, journal: str | None) -> float:
    if not journal:
        return 0.0
    return float(fuzz.token_set_ratio(str(query), clean_html(journal)))


def _enrich_missing_features(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    if "journal_similarity" not in frame.columns and "journal" in frame.columns:
        frame["journal_similarity"] = frame.apply(
            lambda row: compute_journal_similarity(
                str(row.get("query", "")), str(row.get("journal") or "") or None
            ),
            axis=1,
        )
    if "word_count" not in frame.columns and "query" in frame.columns:
        frame["word_count"] = frame["query"].apply(query_word_count)
    if "script_overlap" not in frame.columns and "candidate_title" in frame.columns:
        frame["script_overlap"] = frame.apply(
            lambda row: compute_script_overlap(
                str(row.get("query", "")), str(row.get("candidate_title", ""))
            ),
            axis=1,
        )
    return frame


def build_features(query: str, candidate: dict[str, Any]) -> dict[str, float | int]:
    candidate_title = clean_html(candidate.get("title", [""])[0])
    candidate_journal = (
        clean_html(candidate.get("container-title", [""])[0])
        if candidate.get("container-title")
        else None
    )
    candidate_year = extract_year(candidate)
    query_year = extract_year_from_query(query)
    source = candidate.get("source", "crossref")
    source_rank = candidate.get("source_rank", candidate.get("rank", SOURCE_RANK_MISSING))

    return {
        "title_similarity": fuzz.token_sort_ratio(query, candidate_title),
        "title_token_set_similarity": fuzz.token_set_ratio(query, candidate_title),
        "first_author_similarity": compute_author_feature(
            query, extract_first_author(candidate)
        ),
        "year_difference": compute_year_difference(query_year, candidate_year),
        "journal_similarity": compute_journal_similarity(query, candidate_journal),
        "word_count": query_word_count(query),
        "script_overlap": compute_script_overlap(query, candidate_title),
        "source_rank": int(source_rank),
        "source_crossref": int(source == "crossref"),
        "source_openalex": int(source == "openalex"),
        "source_cyberleninka": int(source == "cyberleninka"),
        "source_elibrary": int(source == "elibrary"),
    }


def normalize_candidate_probabilities(
    raw_probs: list[float],
    title_sims: list[float],
) -> list[float]:
    """Turn per-candidate scores into a distribution that sums to 1 within one query."""
    if not raw_probs:
        return []

    total = sum(raw_probs)
    if total > 1e-12:
        return [prob / total for prob in raw_probs]

    sim_total = sum(title_sims)
    if sim_total > 0:
        return [sim / sim_total for sim in title_sims]

    uniform = 1.0 / len(raw_probs)
    return [uniform] * len(raw_probs)


def calibrated_probability(raw_probability: float, token_similarity: float) -> float:
    """Map model output to a user-facing probability."""
    probability = float(raw_probability)
    if token_similarity >= EXACT_MATCH_THRESHOLD:
        probability = max(probability, token_similarity / 100.0)
    return round(min(max(probability, 0.0), 0.99), 4)


def format_authors(item: dict[str, Any]) -> str:
    authors = item.get("author") or []
    if not authors:
        return ""
    formatted: list[str] = []
    for author in authors[:10]:
        if isinstance(author, str):
            formatted.append(author)
            continue
        family = author.get("family", "")
        given = author.get("given", "")
        if family and given:
            initials = ". ".join(
                part[0] for part in given.replace(".", " ").split() if part
            )
            formatted.append(f"{family} {initials}.")
        elif family:
            formatted.append(family)
        elif given:
            formatted.append(given)
    if len(authors) > 10:
        formatted.append("et al.")
    return ", ".join(formatted)


def _format_source_url(item: dict[str, Any]) -> str | None:
    link = item.get("link")
    if not link:
        return None
    if str(link).startswith("http"):
        return str(link)
    source = item.get("source", "crossref")
    if source == "elibrary":
        return f"https://elibrary.ru{link}"
    if source == "cyberleninka":
        return f"https://cyberleninka.ru{link}"
    return str(link)


def format_citation(item: dict[str, Any]) -> str:
    title = clean_html(item.get("title", [""])[0])
    journal = (
        clean_html(item.get("container-title", [""])[0])
        if item.get("container-title")
        else ""
    )
    year = extract_year(item)
    volume = item.get("volume")
    issue = item.get("issue")
    page = item.get("page") or (
        f"Art. {item['article-number']}" if "article-number" in item else None
    )
    doi = item.get("DOI")

    parts: list[str] = []
    authors = format_authors(item)
    if authors:
        parts.append(authors)
    if title:
        parts.append(title + ".")
    if journal:
        parts.append(journal + ".")
    if year:
        parts.append(str(year) + ".")
    if volume:
        vol = f"Vol. {volume}"
        if issue:
            vol += f" ({issue})"
        parts.append(vol + ".")
    if page:
        parts.append(f"P. {page}.")
    if doi:
        parts.append(f"DOI: {doi}")
    elif url := _format_source_url(item):
        parts.append(f"URL: {url}")
    return " ".join(parts)


def candidate_summary(item: dict[str, Any], *, probability: float) -> dict[str, Any]:
    return {
        "title": clean_html(item.get("title", [""])[0]),
        "doi": item.get("DOI"),
        "link": item.get("link"),
        "year": extract_year(item),
        "authors": format_authors(item),
        "first_author": extract_first_author(item),
        "journal": clean_html(item.get("container-title", [""])[0])
        if item.get("container-title")
        else None,
        "probability": probability,
        "formatted_citation": format_citation(item),
        "source": item.get("source", "crossref"),
    }


class Ranker:
    def __init__(self, model_path: Path | None = None):
        path = model_path or DEFAULT_MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found at {path}. Run: python -m citation_matcher.train"
            )
        self.model = joblib.load(path)

    def summarize(
        self, query: str, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        scored: list[dict[str, Any]] = []
        raw_probs: list[float] = []
        title_sims: list[float] = []

        for candidate in candidates:
            features = build_features(query, candidate)
            frame = pd.DataFrame([features])[FEATURE_COLUMNS]
            raw_probs.append(float(self.model.predict_proba(frame)[0][1]))
            title_sims.append(float(features["title_token_set_similarity"]))
            scored.append({**candidate, "features": features})

        normalized = normalize_candidate_probabilities(raw_probs, title_sims)
        for candidate, raw_probability, token_similarity, probability in zip(
            scored,
            raw_probs,
            title_sims,
            normalized,
            strict=True,
        ):
            candidate["ml_score"] = calibrated_probability(probability, token_similarity)
            candidate["raw_probability"] = raw_probability

        scored.sort(key=lambda item: item["ml_score"], reverse=True)
        return [
            candidate_summary(candidate, probability=candidate["ml_score"])
            for candidate in scored
        ]


@lru_cache(maxsize=2)
def _get_ranker(model_path: str) -> Ranker:
    return Ranker(Path(model_path))


@dataclass
class MatchResult:
    query: str
    best_match: dict[str, Any] | None
    alternatives: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query": self.query,
            "best_match": self.best_match,
            "alternatives": self.alternatives,
        }
        if self.error:
            payload["error"] = self.error
        return payload


def match_citation(
    query: str,
    *,
    model_path: Path | None = None,
    rows: int = DEFAULT_ROWS,
) -> MatchResult:
    query = query.strip()
    if not query:
        return MatchResult(query=query, best_match=None, error="Empty query")

    candidates = multi_search(query, rows=rows)
    if not candidates:
        return MatchResult(
            query=query,
            best_match=None,
            error="No candidates found",
        )

    path = model_path or DEFAULT_MODEL_PATH
    try:
        ranker = _get_ranker(str(path.resolve()))
    except FileNotFoundError as exc:
        return MatchResult(query=query, best_match=None, error=str(exc))

    ranked = ranker.summarize(query, candidates)
    return MatchResult(
        query=query,
        best_match=ranked[0],
        alternatives=ranked[1:],
    )


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["title_similarity"] = df.apply(
        lambda row: fuzz.token_sort_ratio(
            str(row.get("query", "")), str(row.get("candidate_title", ""))
        ),
        axis=1,
    )
    df["title_token_set_similarity"] = df.apply(
        lambda row: fuzz.token_set_ratio(
            str(row.get("query", "")), str(row.get("candidate_title", ""))
        ),
        axis=1,
    )
    df["query_year"] = df["query"].apply(extract_year_from_query)
    df["year_difference"] = df.apply(
        lambda row: compute_year_difference(
            int(row["query_year"]) if pd.notna(row.get("query_year")) else None,
            int(row["candidate_year"]) if pd.notna(row.get("candidate_year")) else None,
        ),
        axis=1,
    )
    df["first_author_similarity"] = df.apply(
        lambda row: compute_author_feature(
            str(row.get("query", "")),
            str(row.get("candidate_author") or "") or None,
            query_author=str(row.get("query_author") or "") or None,
        ),
        axis=1,
    )
    df["journal_similarity"] = df.apply(
        lambda row: compute_journal_similarity(
            str(row.get("query", "")), str(row.get("journal") or "") or None
        ),
        axis=1,
    )
    df["word_count"] = df["query"].apply(query_word_count)
    df["script_overlap"] = df.apply(
        lambda row: compute_script_overlap(
            str(row.get("query", "")), str(row.get("candidate_title", ""))
        ),
        axis=1,
    )
    return prepare_training_frame(df)
