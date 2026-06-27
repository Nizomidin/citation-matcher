from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Any

import pandas as pd
from rapidfuzz import fuzz

from citation_matcher.config import DATASET_SEED, DEFAULT_MODEL_PATH, DEFAULT_ROWS, REPORTS_DIR
from citation_matcher.dataset import ERROR_TYPE_STRINGS, ErrorType, generate_query
from citation_matcher.matcher import Ranker, extract_first_author, extract_year
from citation_matcher.search import (
    crossref_search,
    cyberleninka_search,
    get_articles_crossref,
    get_articles_cyberleninka,
    get_articles_elibrary,
    multi_search,
    openalex_search,
)
from citation_matcher.elibrary import elibrary_search
from citation_matcher.util.parsing_utils import _CYRILLIC_RE, clean_html

logger = logging.getLogger(__name__)

BENCHMARK_SOURCES = ("crossref", "openalex", "cyberleninka", "elibrary")
SEARCH_FUNCTIONS = {
    "crossref": crossref_search,
    "openalex": openalex_search,
    "cyberleninka": cyberleninka_search,
    "elibrary": elibrary_search,
}
RATE_LIMITED_SOURCES = frozenset({"cyberleninka", "elibrary"})
TITLE_MATCH_THRESHOLD = 92.0
EN_SOURCES = frozenset({"crossref", "openalex"})
RU_SOURCES = frozenset({"cyberleninka", "elibrary"})

BENCHMARK_ERROR_TYPES = (
    ErrorType.clean,
    ErrorType.typo_in_title,
    ErrorType.no_author,
    ErrorType.no_year,
    ErrorType.title_with_drop_word,
)


def infer_article_lang(article: dict[str, Any]) -> str:
    if article.get("lang") in {"en", "ru"}:
        return str(article["lang"])
    if article.get("source") == "crossref":
        return "en"
    if article.get("source") in {"cyberleninka", "elibrary"}:
        return "ru"
    title = str(article.get("title", ""))
    if _CYRILLIC_RE.search(title):
        return "ru"
    return "en"


def tag_article(article: dict[str, Any]) -> dict[str, Any]:
    tagged = dict(article)
    tagged["lang"] = infer_article_lang(tagged)
    return tagged


def load_bilingual_pool(
    english_count: int = 15,
    russian_count: int = 15,
    *,
    russian_csv: Path | None = None,
) -> list[dict[str, Any]]:
    """Mixed EN (Crossref) + RU (CyberLeninka or CSV) articles."""
    articles = [tag_article(a) for a in get_articles_crossref(english_count)]
    if russian_csv:
        articles.extend(_load_articles_from_csv(russian_csv, russian_count))
    else:
        articles.extend(
            tag_article(a) for a in get_articles_cyberleninka(russian_count)
        )
    return articles


def load_seed_articles(
    pool: str = "bilingual",
    count: int = 30,
    *,
    csv_path: Path | None = None,
    english_count: int | None = None,
    russian_count: int | None = None,
) -> list[dict[str, Any]]:
    if pool == "bilingual":
        en = english_count if english_count is not None else count // 2
        ru = russian_count if russian_count is not None else count - en
        return load_bilingual_pool(en, ru, russian_csv=csv_path)
    if csv_path:
        return _load_articles_from_csv(csv_path, count)
    if pool == "crossref":
        return [tag_article(a) for a in get_articles_crossref(count)]
    if pool == "cyberleninka":
        return [tag_article(a) for a in get_articles_cyberleninka(count)]
    if pool == "elibrary":
        return [tag_article(a) for a in get_articles_elibrary(count)]
    if pool == "mixed":
        third = max(count // 3, 1)
        return [
            tag_article(a)
            for a in (
                get_articles_crossref(third)
                + get_articles_cyberleninka(third)
                + get_articles_elibrary(count - 2 * third)
            )
        ]
    raise ValueError(f"Unknown article pool: {pool}")


def _load_articles_from_csv(path: Path, count: int) -> list[dict[str, Any]]:
    frame = pd.read_csv(path).head(count)
    articles: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        title = clean_html(getattr(row, "title", "") or "")
        year = getattr(row, "year", None)
        authors = getattr(row, "authors", None) or getattr(row, "first_author", None)
        first_author = str(authors).split(",")[0].strip() if authors else ""
        article_id = getattr(row, "id", None) or getattr(row, "url", None)
        if not title or not year or not first_author or not article_id:
            continue
        articles.append(
            tag_article(
                {
                    "id": str(article_id),
                    "source": "elibrary",
                    "title": title,
                    "year": int(year),
                    "first_author": first_author,
                    "journal": clean_html(
                        getattr(row, "source", "") or getattr(row, "journal", "") or ""
                    ),
                }
            )
        )
    return articles


def generate_unified_query(article: dict[str, Any], error_type: ErrorType) -> str:
    """Use the same bibliographic query format for every source."""
    return generate_query({**article, "source": "crossref"}, error_type)


def candidate_matches_seed(candidate: dict[str, Any], seed: dict[str, Any]) -> bool:
    source = candidate.get("source")
    seed_source = seed.get("source")
    seed_id = str(seed.get("id", ""))

    candidate_doi = candidate.get("DOI")
    if seed_source == "crossref" and candidate_doi and candidate_doi == seed_id:
        return True

    if source == seed_source:
        if source == "crossref" and candidate.get("DOI") == seed_id:
            return True
        candidate_ref = candidate.get("link") or candidate.get("id")
        if candidate_ref and str(candidate_ref) == seed_id:
            return True
        if source == "elibrary" and seed_id.isdigit():
            if f"id={seed_id}" in str(candidate_ref or ""):
                return True

    title_raw = candidate.get("title")
    if isinstance(title_raw, list):
        title = clean_html(title_raw[0] if title_raw else "")
    else:
        title = clean_html(str(title_raw or ""))
    if fuzz.token_set_ratio(title, seed["title"]) < TITLE_MATCH_THRESHOLD:
        return False

    seed_year = seed.get("year")
    candidate_year = extract_year(candidate)
    if seed_year and candidate_year and int(seed_year) != int(candidate_year):
        return False

    seed_author = (seed.get("first_author") or "").strip()
    candidate_author = (extract_first_author(candidate) or "").strip()
    if source == seed_source and seed_author and candidate_author:
        if max(
            fuzz.partial_ratio(seed_author, candidate_author),
            fuzz.partial_ratio(candidate_author, seed_author),
        ) < 70:
            return False
    return True


def _summary_to_candidate(summary: dict[str, Any]) -> dict[str, Any]:
    first_author = summary.get("first_author") or ""
    family = first_author.rsplit(" ", 1)[-1] if first_author else ""
    given = first_author.rsplit(" ", 1)[0] if " " in first_author else ""
    item: dict[str, Any] = {
        "title": [summary.get("title") or ""],
        "source": summary.get("source"),
        "author": [{"given": given, "family": family}] if first_author else [],
        "DOI": summary.get("doi"),
        "link": summary.get("link"),
    }
    year = summary.get("year")
    if year:
        item["issued"] = {"date-parts": [[int(year)]]}
    return item


def find_match_rank(candidates: list[dict[str, Any]], seed: dict[str, Any]) -> int:
    for rank, candidate in enumerate(candidates, start=1):
        if candidate_matches_seed(candidate, seed):
            return rank
    return -1


def find_model_rank(
    query: str,
    candidates: list[dict[str, Any]],
    seed: dict[str, Any],
    ranker: Ranker,
) -> int:
    if not candidates:
        return -1
    summaries = ranker.summarize(query, candidates)
    for rank, summary in enumerate(summaries, start=1):
        if candidate_matches_seed(_summary_to_candidate(summary), seed):
            return rank
    return -1


def _search_source(source: str, query: str, rows: int) -> list[dict[str, Any]]:
    search_fn = SEARCH_FUNCTIONS.get(source)
    if not search_fn:
        return []
    try:
        return search_fn(query, rows=rows)
    except Exception:
        logger.exception("%s search failed for query=%r", source, query[:80])
        return []


def _metrics_from_ranks(ranks: list[int]) -> dict[str, float | int]:
    found = [rank for rank in ranks if rank > 0]
    total = len(ranks)
    return {
        "recall@10": len(found) / total if total else 0.0,
        "top1": sum(rank == 1 for rank in found) / total if total else 0.0,
        "top3": sum(rank <= 3 for rank in found) / total if total else 0.0,
        "mrr": sum(1.0 / rank for rank in found) / total if total else 0.0,
        "found": len(found),
        "count": total,
    }


def run_unified_benchmark(
    articles: list[dict[str, Any]],
    *,
    sources: tuple[str, ...] = BENCHMARK_SOURCES,
    rows: int = DEFAULT_ROWS,
    error_types: tuple[ErrorType, ...] = BENCHMARK_ERROR_TYPES,
    include_model: bool = True,
    ranker: Ranker | None = None,
) -> pd.DataFrame:
    random.seed(DATASET_SEED)
    if include_model and ranker is None:
        ranker = Ranker(DEFAULT_MODEL_PATH)

    rows_out: list[dict[str, Any]] = []

    for error_type in error_types:
        for article in articles:
            query = generate_unified_query(article, error_type)
            row: dict[str, Any] = {
                "article_id": article["id"],
                "article_source": article["source"],
                "lang": article.get("lang", infer_article_lang(article)),
                "title": article["title"],
                "year": article["year"],
                "first_author": article.get("first_author"),
                "query": query,
                "error_type": ERROR_TYPE_STRINGS[error_type],
            }
            for source in sources:
                candidates = _search_source(source, query, rows)
                row[f"{source}_rank"] = find_match_rank(candidates, article)
                if include_model and ranker is not None:
                    row[f"model_{source}_rank"] = find_model_rank(
                        query, candidates, article, ranker
                    )
                if source in RATE_LIMITED_SOURCES:
                    time.sleep(0.15)

            merged = multi_search(query, rows=rows, sources=sources)
            row["multi_rank"] = find_match_rank(merged, article)
            if include_model and ranker is not None:
                row["matcher_rank"] = find_model_rank(query, merged, article, ranker)

            rows_out.append(row)

    return pd.DataFrame(rows_out)


def _rank_columns(
    sources: tuple[str, ...], *, include_model: bool = True
) -> list[tuple[str, str]]:
    columns: list[tuple[str, str]] = []
    for source in sources:
        columns.append((source, f"{source}_rank"))
        if include_model:
            columns.append((f"model:{source}", f"model_{source}_rank"))
    columns.append(("multi", "multi_rank"))
    if include_model:
        columns.append(("matcher", "matcher_rank"))
    return columns


def summarize_unified_benchmark(
    frame: pd.DataFrame,
    *,
    sources: tuple[str, ...] = BENCHMARK_SOURCES,
    include_model: bool = True,
) -> pd.DataFrame:
    rank_columns = _rank_columns(sources, include_model=include_model)
    rows: list[dict[str, Any]] = []

    lang_values = ["all"]
    if "lang" in frame.columns:
        lang_values.extend(sorted(lang for lang in frame["lang"].dropna().unique()))

    error_values = list(frame["error_type"].unique()) + ["all"]

    for lang in lang_values:
        lang_frame = frame if lang == "all" else frame[frame["lang"] == lang]
        if lang_frame.empty:
            continue
        for error_type in error_values:
            group = (
                lang_frame
                if error_type == "all"
                else lang_frame[lang_frame["error_type"] == error_type]
            )
            if group.empty:
                continue
            for ranker_name, column in rank_columns:
                if column not in group.columns:
                    continue
                metrics = _metrics_from_ranks(group[column].tolist())
                rows.append(
                    {
                        "lang": lang,
                        "error_type": error_type,
                        "ranker": ranker_name,
                        **metrics,
                    }
                )

    return pd.DataFrame(rows)


def build_language_leaderboard(
    summary: pd.DataFrame, lang: str, *, include_model: bool = True
) -> pd.DataFrame:
    block = summary[(summary["lang"] == lang) & (summary["error_type"] == "all")].copy()
    if block.empty:
        return block
    block = block.sort_values("top1", ascending=False).reset_index(drop=True)
    block.insert(0, "place", range(1, len(block) + 1))
    return block[
        ["place", "ranker", "top1", "top3", "recall@10", "mrr", "found", "count"]
    ]


def build_native_vs_model_table(
    summary: pd.DataFrame, lang: str
) -> pd.DataFrame:
    block = summary[(summary["lang"] == lang) & (summary["error_type"] == "all")]
    rows: list[dict[str, object]] = []
    rankers = block["ranker"].tolist()
    for ranker in sorted(set(rankers)):
        if ranker.startswith("model:") or ranker in {"matcher", "multi"}:
            continue
        native = block[block["ranker"] == ranker]
        model_key = f"model:{ranker}"
        model = block[block["ranker"] == model_key]
        if native.empty:
            continue
        native_row = native.iloc[0]
        row = {
            "Источник": ranker,
            "Native Top-1": f"{100 * native_row['top1']:.1f}%",
            "Native Recall@10": f"{100 * native_row['recall@10']:.1f}%",
            "Native MRR": f"{native_row['mrr']:.3f}",
        }
        if not model.empty:
            model_row = model.iloc[0]
            row["Model Top-1"] = f"{100 * model_row['top1']:.1f}%"
            row["Model Recall@10"] = f"{100 * model_row['recall@10']:.1f}%"
            row["Δ Top-1"] = f"{100 * (model_row['top1'] - native_row['top1']):+.1f}%"
        rows.append(row)

    if "matcher" in rankers and "multi" in rankers:
        multi = block[block["ranker"] == "multi"].iloc[0]
        matcher = block[block["ranker"] == "matcher"].iloc[0]
        rows.append(
            {
                "Источник": "matcher (multi+ML)",
                "Native Top-1": f"{100 * multi['top1']:.1f}%",
                "Native Recall@10": f"{100 * multi['recall@10']:.1f}%",
                "Native MRR": f"{multi['mrr']:.3f}",
                "Model Top-1": f"{100 * matcher['top1']:.1f}%",
                "Model Recall@10": f"{100 * matcher['recall@10']:.1f}%",
                "Δ Top-1": f"{100 * (matcher['top1'] - multi['top1']):+.1f}%",
            }
        )
    return pd.DataFrame(rows)


def build_source_leaderboard(summary: pd.DataFrame) -> pd.DataFrame:
    return build_language_leaderboard(summary, "all")


def build_error_comparison_table(summary: pd.DataFrame) -> pd.DataFrame:
    by_error = summary[
        (summary["error_type"] != "all") & (summary["lang"] == "all")
    ].copy()
    if by_error.empty:
        return by_error

    rows: list[dict[str, object]] = []
    for error_type, group in by_error.groupby("error_type", sort=False):
        for _, row in group.sort_values("top1", ascending=False).iterrows():
            rows.append(
                {
                    "Ошибка": error_type,
                    "Ранжировщик": row["ranker"],
                    "Top-1": f"{100 * row['top1']:.1f}%",
                    "Top-3": f"{100 * row['top3']:.1f}%",
                    "Recall@10": f"{100 * row['recall@10']:.1f}%",
                    "MRR": f"{row['mrr']:.3f}",
                    "Найдено": f"{int(row['found'])}/{int(row['count'])}",
                }
            )
    return pd.DataFrame(rows)


def default_cache_path() -> Path:
    return REPORTS_DIR / "unified_bilingual_benchmark.csv"
