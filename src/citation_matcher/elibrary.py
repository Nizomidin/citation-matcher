from __future__ import annotations

import asyncio
import logging
from typing import Any

from curl_cffi.requests import AsyncSession
from curl_cffi.requests.exceptions import RequestException
from rapidfuzz import fuzz

from citation_matcher.config import (
    ELIBRARY_HEADERS,
    ELIBRARY_HOME_URL,
    ELIBRARY_SEARCH_URL,
    ELIBRARY_SEED_QUERIES,
)
from citation_matcher.util.network_utils import get_proxy, load_cookies, save_cookies
from citation_matcher.util.parsing_utils import (
    _parse_authors,
    clean_html,
    parse_elibrary_candidates,
)

logger = logging.getLogger(__name__)

SEARCH_FORM: dict[str, str] = {
    "where_fulltext": "on",
    "where_name": "on",
    "where_abstract": "on",
    "where_keywords": "on",
    "type_article": "on",
    "type_disser": "on",
    "type_book": "on",
    "type_report": "on",
    "type_conf": "on",
    "type_patent": "on",
    "type_preprint": "on",
    "type_grant": "on",
    "type_dataset": "on",
    "search_morph": "on",
    "issues": "all",
    "orderby": "rank",
    "order": "rev",
    "changed": "1",
}


def _is_access_denied(html: str) -> bool:
    return "403 - Forbidden" in html or "Access is denied" in html


def _is_captcha(url: str, html: str) -> bool:
    return "page_captcha.asp" in url or "нарушения правил пользования" in html


def _is_empty(html: str) -> bool:
    return "Не найдено публикаций, соответствующих запросу" in html


def _results_match_query(query: str, parsed: list[dict[str, Any]]) -> bool:
    """Detect stale eLibrary sessions that ignore the current query."""
    if not parsed:
        return False
    top_title = str(parsed[0].get("title", ""))
    return fuzz.token_set_ratio(query, top_title) >= 60


def _first_author_name(authors: str) -> str | None:
    for part in authors.split(","):
        name = part.strip()
        if name:
            return name
    return None


def _to_search_candidate(item: dict[str, Any], rank: int, rows: int) -> dict[str, Any]:
    author_names = [
        name.strip() for name in str(item.get("authors", "")).split(",") if name.strip()
    ]
    link = f"/item.asp?id={item['id']}"
    journal = clean_html(str(item.get("source", "")).split(".")[0])
    year = item.get("year")
    normalized: dict[str, Any] = {
        "title": [clean_html(item["title"])],
        "author": _parse_authors(author_names),
        "container-title": [journal] if journal else [],
        "DOI": None,
        "link": link,
        "id": link,
        "score": float(rows - rank + 1),
        "source_rank": rank,
        "rank": rank,
        "source": "elibrary",
    }
    if year:
        normalized["year"] = int(year)
        normalized["issued"] = {"date-parts": [[int(year)]]}
    return normalized


def _to_dataset_article(item: dict[str, Any]) -> dict[str, Any] | None:
    article_id = f"/item.asp?id={item['id']}"
    title = clean_html(item.get("title"))
    year = item.get("year")
    first_author = _first_author_name(str(item.get("authors", "")))
    if not title or not year or not first_author:
        return None
    journal = clean_html(str(item.get("source", "")).split(".")[0]) or None
    return {
        "id": article_id,
        "source": "elibrary",
        "title": title,
        "year": int(year),
        "first_author": first_author,
        "journal": journal,
    }


async def _fetch_html(
    query: str,
    cookies: dict[str, str] | None,
    *,
    page: int = 1,
    proxy: str | None = None,
    refresh_session: bool = False,
) -> tuple[str | None, bool, dict[str, str]]:
    """Fetch search results; optionally open homepage in the same session first."""
    data = {**SEARCH_FORM, "ftext": query}
    try:
        async with AsyncSession(
            impersonate="chrome", headers=ELIBRARY_HEADERS, proxy=proxy, 
        ) as session:
            if refresh_session or not cookies:
                home = await session.get(ELIBRARY_HOME_URL)
                if _is_captcha(str(home.url), home.text):
                    return None, False, {}
            elif cookies:
                session.cookies.update(cookies)

            response = await session.post(
                ELIBRARY_SEARCH_URL,
                params={"pagenum": str(max(page, 1))},
                data=data,
            )
            html = response.text
            updated_cookies = session.cookies.get_dict()
            if _is_captcha(str(response.url), html):
                return None, False, updated_cookies
            if _is_access_denied(html):
                logger.warning("eLibrary returned 403 Forbidden (proxy/cookies may be required)")
                return html, False, updated_cookies
            if _is_empty(html) or not response.ok:
                return html, False, updated_cookies
            return html, True, updated_cookies
    except RequestException as exc:
        logger.warning("eLibrary request failed: %s", exc)
        return None, False, cookies or {}


async def _search_async(query: str, rows: int) -> list[dict[str, Any]]:
    proxy = get_proxy()
    cookies = load_cookies()

    html, ok, cookies = await _fetch_html(query, cookies, proxy=proxy)
    parsed = parse_elibrary_candidates(html or "") if html else []

    if ok and parsed and not _results_match_query(query, parsed):
        logger.warning(
            "eLibrary returned unrelated results (stale session), opening fresh session"
        )
        html, ok, cookies = await _fetch_html(
            query, None, proxy=proxy, refresh_session=True
        )
        parsed = parse_elibrary_candidates(html or "") if html else []

    if not ok:
        html, ok, cookies = await _fetch_html(
            query, None, proxy=proxy, refresh_session=True
        )
        parsed = parse_elibrary_candidates(html or "") if html else []

    if cookies:
        save_cookies(cookies)

    if not ok or not html:
        return []

    return [
        _to_search_candidate(item, rank, rows)
        for rank, item in enumerate(parsed[:rows], start=1)
    ]


def elibrary_search(query: str, rows: int = 10) -> list[dict[str, Any]]:
    """Sync wrapper for multi_search thread pool."""
    try:
        return asyncio.run(_search_async(query, rows))
    except Exception:
        logger.exception("eLibrary search failed for query=%r", query)
        return []


def get_articles_elibrary(count: int) -> list[dict[str, Any]]:
    if count <= 0:
        return []

    collected: dict[str, dict[str, Any]] = {}
    for keyword in ELIBRARY_SEED_QUERIES:
        html_items = asyncio.run(_collect_raw_items(keyword))
        for item in html_items:
            article = _to_dataset_article(item)
            if not article:
                continue
            collected[article["id"]] = article
            if len(collected) >= count:
                return list(collected.values())
    return list(collected.values())


async def _collect_raw_items(query: str) -> list[dict[str, Any]]:
    candidates = await _search_async(query, rows=25)
    if not candidates:
        return []
    return [
        {
            "id": item["link"].split("=")[-1],
            "title": (item.get("title") or [""])[0],
            "authors": ", ".join(
                f"{a.get('given', '')} {a.get('family', '')}".strip()
                for a in (item.get("author") or [])
            ),
            "source": (item.get("container-title") or [""])[0],
            "year": item.get("year"),
        }
        for item in candidates
        if item.get("link")
    ]


def elibrary_dataset_results(query: str, rows: int = 10) -> list[dict[str, Any]]:
    """Return cyberleninka-compatible dicts for dataset building."""
    results: list[dict[str, Any]] = []
    for candidate in elibrary_search(query, rows=rows):
        authors = [
            f"{author.get('given', '')} {author.get('family', '')}".strip()
            for author in (candidate.get("author") or [])
        ]
        results.append(
            {
                "link": candidate.get("link"),
                "name": (candidate.get("title") or [""])[0],
                "authors": authors,
                "year": candidate.get("year"),
                "journal": (candidate.get("container-title") or [""])[0] or None,
            }
        )
    return results


def refresh_elibrary_cookies() -> bool:
    async def _run() -> bool:
        proxy = get_proxy()
        _, ok, cookies = await _fetch_html(
            "экономика", None, proxy=proxy, refresh_session=True
        )
        if ok and cookies:
            save_cookies(cookies)
        return ok and bool(cookies)

    return asyncio.run(_run())
