from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from citation_matcher.config import (
    CROSSREF_API,
    CYBERLENINKA_API,
    DEFAULT_MULTI_SEARCH_SOURCES,
    DEFAULT_ROWS,
    OPENALEX_API,
    USER_AGENT,
)
from citation_matcher.elibrary import (
    elibrary_dataset_results,
    elibrary_search,
    get_articles_elibrary,
)
from citation_matcher.util.network_utils import request_json
from citation_matcher.util.parsing_utils import (
    _candidate_key,
    _extract_year,
    _parse_authors,
    clean_html,
)

logger = logging.getLogger(__name__)

__all__ = [
    "crossref_search",
    "openalex_search",
    "cyberleninka_search",
    "elibrary_search",
    "multi_search",
    "get_articles_crossref",
    "get_articles_cyberleninka",
    "get_articles_elibrary",
    "search_for_dataset",
]


def crossref_search(query: str, rows: int = DEFAULT_ROWS) -> list[dict[str, Any]]:
    data = request_json(
        "GET",
        CROSSREF_API,
        params={"query.bibliographic": query, "rows": rows},
        headers={"User-Agent": USER_AGENT},
    )
    if not data:
        return []

    items = data.get("message", {}).get("items", [])[:rows]
    return [
        {**dict(item), "source_rank": rank, "rank": rank, "source": "crossref"}
        for rank, item in enumerate(items, start=1)
    ]


def openalex_search(query: str, rows: int = DEFAULT_ROWS) -> list[dict[str, Any]]:
    data = request_json(
        "GET",
        OPENALEX_API,
        params={
            "search": query,
            "per-page": rows,
            "select": "title,authorships,primary_location,publication_year,doi,relevance_score",
        },
        headers={"User-Agent": USER_AGENT},
    )
    if not data:
        return []

    candidates: list[dict[str, Any]] = []
    for rank, work in enumerate((data.get("results") or [])[:rows], start=1):
        authors: list[dict[str, str]] = []
        for authorship in work.get("authorships") or []:
            display = authorship.get("author", {}).get("display_name") or ""
            parts = display.rsplit(" ", 1)
            authors.append(
                {
                    "given": parts[0] if len(parts) > 1 else "",
                    "family": parts[-1],
                }
            )

        venue = (work.get("primary_location") or {}).get("source") or {}
        journal = venue.get("display_name") or ""
        year = work.get("publication_year")
        doi = (work.get("doi") or "").replace("https://doi.org/", "")

        item: dict[str, Any] = {
            "title": [clean_html(work.get("title") or "")],
            "author": authors,
            "container-title": [journal] if journal else [],
            "DOI": doi or None,
            "score": float(work.get("relevance_score") or 0),
            "source_rank": rank,
            "rank": rank,
            "source": "openalex",
        }
        if year:
            item["issued"] = {"date-parts": [[year]]}
        candidates.append(item)
    return candidates


def cyberleninka_search(query: str, rows: int = DEFAULT_ROWS) -> list[dict[str, Any]]:
    data = request_json(
        "POST",
        CYBERLENINKA_API,
        json={"mode": "articles", "q": query, "size": rows, "from": 0},
        headers={
            "content-type": "application/json",
            "origin": "https://cyberleninka.ru",
            "referer": "https://cyberleninka.ru/",
            "user-agent": USER_AGENT,
        },
    )
    if not data:
        return []

    candidates: list[dict[str, Any]] = []
    for rank, article in enumerate((data.get("articles") or [])[:rows], start=1):
        year = article.get("year")
        link = article.get("link")
        item: dict[str, Any] = {
            "title": [clean_html(article.get("name"))],
            "container-title": [clean_html(article.get("journal"))]
            if article.get("journal")
            else [],
            "author": _parse_authors(article.get("authors")),
            "DOI": None,
            "link": link,
            "score": float(rows - rank + 1),
            "source_rank": rank,
            "rank": rank,
            "source": "cyberleninka",
        }
        if year:
            item["year"] = year
            item["issued"] = {"date-parts": [[int(year)]]}
        candidates.append(item)
    return candidates


def multi_search(
    query: str,
    rows: int = DEFAULT_ROWS,
    *,
    sources: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    all_sources = {
        "crossref": crossref_search,
        "openalex": openalex_search,
        "cyberleninka": cyberleninka_search,
        "elibrary": elibrary_search,
    }
    if sources is None:
        source_map = {
            name: all_sources[name]
            for name in DEFAULT_MULTI_SEARCH_SOURCES
            if name in all_sources
        }
    else:
        source_map = {name: all_sources[name] for name in sources if name in all_sources}
    results: dict[str, list[dict[str, Any]]] = {}

    with ThreadPoolExecutor(max_workers=max(len(source_map), 1)) as pool:
        futures = {
            pool.submit(search_fn, query, rows): name
            for name, search_fn in source_map.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception:
                logger.exception("%s search failed", name)
                results[name] = []

    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    for source in source_map:
        for candidate in results.get(source) or []:
            key = _candidate_key(candidate)
            if key in seen:
                continue
            seen.add(key)
            merged.append(candidate)

    for idx, candidate in enumerate(merged, start=1):
        candidate["rank"] = idx
    return merged


def get_articles_crossref(count: int) -> list[dict[str, Any]]:
    if count <= 0:
        return []

    data = request_json(
        "GET",
        CROSSREF_API,
        params={
            "filter": "type:journal-article,has-references:true",
            "rows": count,
            "select": "DOI,title,author,container-title,issued,published-print,published-online",
        },
        headers={"User-Agent": USER_AGENT},
    )
    if not data:
        return []

    cleaned: list[dict[str, Any]] = []
    for item in data.get("message", {}).get("items", []):
        title_list = item.get("title") or []
        authors = item.get("author") or []
        first = authors[0] if authors else None
        first_name = (
            f"{first.get('given', '')} {first.get('family', '')}".strip()
            if isinstance(first, dict)
            else str(first or "").strip()
        )
        year = _extract_year(item)
        doi = item.get("DOI")
        if not doi or not title_list or not first_name or not year:
            continue
        if title_list == ["In Response"]:
            continue
        cleaned.append(
            {
                "id": doi,
                "source": "crossref",
                "title": clean_html(title_list[0]),
                "year": year,
                "first_author": first_name,
                "journal": clean_html((item.get("container-title") or [""])[0]),
            }
        )
    return cleaned


def get_articles_cyberleninka(count: int) -> list[dict[str, Any]]:
    if count <= 0:
        return []

    data = request_json(
        "POST",
        CYBERLENINKA_API,
        json={"mode": "articles", "q": "", "size": count, "from": 0},
        headers={"User-Agent": USER_AGENT},
    )
    if not data:
        return []

    cleaned: list[dict[str, Any]] = []
    for item in data.get("articles") or []:
        authors = item.get("authors") or []
        link = item.get("link")
        title = item.get("name")
        year = item.get("year")
        if not link or not title or not authors or not year:
            continue
        cleaned.append(
            {
                "id": link,
                "source": "cyberleninka",
                "title": clean_html(title),
                "year": year,
                "first_author": authors[0],
                "journal": clean_html(item.get("journal")),
            }
        )
    return cleaned


def search_for_dataset(
    source: str, query: str, rows: int = DEFAULT_ROWS
) -> list[dict[str, Any]]:
    if source == "crossref":
        data = request_json(
            "GET",
            CROSSREF_API,
            params={"query.bibliographic": query, "rows": rows},
            headers={"User-Agent": USER_AGENT},
        )
        return (data or {}).get("message", {}).get("items", [])
    if source == "cyberleninka":
        data = request_json(
            "POST",
            CYBERLENINKA_API,
            json={"mode": "articles", "q": query, "size": rows, "from": 0},
            headers={"User-Agent": USER_AGENT},
        )
        if not data or data.get("found", 0) == 0:
            return []
        return data.get("articles") or []
    if source == "elibrary":
        return elibrary_dataset_results(query, rows=rows)
    raise ValueError(f"Unknown source: {source}")
