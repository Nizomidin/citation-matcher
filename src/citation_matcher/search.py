from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests

from citation_matcher.text import clean_html

USER_AGENT = "citation-matcher/0.1 (mailto:citation-matcher@example.com)"

CROSSREF_API = "https://api.crossref.org/works"
OPENALEX_API = "https://api.openalex.org/works"


def crossref_search(query: str, rows: int = 10) -> list[dict[str, Any]]:
    """Search Crossref for bibliographic candidates."""
    try:
        response = requests.get(
            CROSSREF_API,
            params={"query.bibliographic": query, "rows": rows},
            timeout=20,
            headers={"User-Agent": USER_AGENT},
        )
    except requests.RequestException:
        return []

    if response.status_code != 200:
        return []

    items = response.json().get("message", {}).get("items", [])
    candidates: list[dict[str, Any]] = []
    for rank, item in enumerate(items, start=1):
        item = dict(item)
        item["rank"] = rank
        item["source"] = "crossref"
        candidates.append(item)
    return candidates


def _openalex_to_crossref(work: dict[str, Any], rank: int) -> dict[str, Any]:
    """Normalize an OpenAlex work to a Crossref-like structure."""
    # title
    title = clean_html(work.get("title") or "")

    # authors
    authors: list[dict[str, str]] = []
    for authorship in work.get("authorships") or []:
        display = authorship.get("author", {}).get("display_name") or ""
        parts = display.rsplit(" ", 1)
        family = parts[-1]
        given = parts[0] if len(parts) > 1 else ""
        authors.append({"given": given, "family": family})

    # journal
    venue = work.get("primary_location") or {}
    source = venue.get("source") or {}
    journal = source.get("display_name") or ""

    # year
    year = work.get("publication_year")

    # doi
    doi = (work.get("doi") or "").replace("https://doi.org/", "")

    # crossref score proxy: relevance_score from OpenAlex
    oa_score = float(work.get("relevance_score") or 0)

    normalized: dict[str, Any] = {
        "title": [title],
        "author": authors,
        "container-title": [journal] if journal else [],
        "DOI": doi or None,
        "score": oa_score,
        "rank": rank,
        "source": "openalex",
    }
    if year:
        normalized["issued"] = {"date-parts": [[year]]}
    return normalized


def openalex_search(query: str, rows: int = 10) -> list[dict[str, Any]]:
    """Search OpenAlex for bibliographic candidates."""
    try:
        response = requests.get(
            OPENALEX_API,
            params={
                "search": query,
                "per-page": rows,
                "select": "title,authorships,primary_location,publication_year,doi,relevance_score",
            },
            timeout=20,
            headers={"User-Agent": USER_AGENT},
        )
    except requests.RequestException:
        return []

    if response.status_code != 200:
        return []

    results = response.json().get("results") or []
    return [
        _openalex_to_crossref(work, rank) for rank, work in enumerate(results, start=1)
    ]


def cyberleninka_search(query: str, rows: int = 10):
    headers = {
        "accept": "*/*",
        "accept-language": "en-US,en;q=0.9,ru;q=0.8",
        "content-type": "application/json",
        "origin": "https://cyberleninka.ru",
        "priority": "u=1, i",
        "referer": "https://cyberleninka.ru/search?q=%D0%A1%D0%9E%D0%92%D0%95%D0%A0%D0%A8%D0%95%D0%9D%D0%A1%D0%A2%D0%92%D0%9E%D0%92%D0%90%D0%9D%D0%98%D0%95%20%D0%A1%D0%A2%D0%A0%D0%A3%D0%9A%D0%A2%D0%A3%D0%A0%D0%AB%20%D0%A1%D0%A3%D0%91%D0%AA%D0%95%D0%9A%D0%A2%D0%9E%D0%92%20%D0%9F%D0%9E%D0%A2%D0%A0%D0%95%D0%91%D0%98%D0%A2%D0%95%D0%9B%D0%AC%D0%A1%D0%9A%D0%9E%D0%93%D0%9E%20%D0%A0%D0%AB%D0%9D%D0%9A%D0%90%20%D0%9D%D0%90%20%D0%9E%D0%A1%D0%9D%D0%9E%D0%92%D0%95%20%D0%98%D0%A1%D0%9F%D0%9E%D0%9B%D0%AC%D0%97%D0%9E%D0%92%D0%90%D0%9D%D0%98%D0%AF%20%D0%A4%D0%98%D0%97%D0%98%D0%A7%D0%95%D0%A1%D0%9A%D0%98%D0%A5%20%D0%90%D0%9D%D0%90%D0%9B%D0%9E%D0%93%D0%98%D0%99&page=1",
        "sec-ch-ua": '"Chromium";v="148", "Google Chrome";v="148", "Not/A)Brand";v="99"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"macOS"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36",
        # 'cookie': '_ga=GA1.2.763198204.1780901579; _gid=GA1.2.96971.1780901579; _ym_uid=1780901579460253252; _ym_d=1780901579; _ym_isad=2; adrcid=AumZitIzjeCSA6IBMqe3CXQ; acs_3=%7B%22hash%22%3A%221aa3f9523ee6c2690cb34fc702d4143056487c0d%22%2C%22nst%22%3A1780987988364%2C%22sl%22%3A%7B%22224%22%3A1780901588364%2C%221228%22%3A1780901588364%7D%7D; _gat=1; _gcl_au=1.2.391471826.1780952831; _ga_4GZ9YCR2VB=GS2.2.s1780952828$o2$g1$t1780952831$j57$l0$h0; adrdel=1780952831420',
    }

    json_data = {
        "mode": "articles",
        "q": query,
        "size": 10,
        "from": 0,
    }

    try:
        response = requests.post(
            "https://cyberleninka.ru/api/search", headers=headers, json=json_data
        )
    except requests.RequestException:
        return []

    if response.status_code != 200:
        return []
    articles = response.json().get("articles") or []
    print(articles)
    result: list[dict[str, Any]] = []
    for rank, article in enumerate(articles[:rows], start=1):
        year = article.get("year")
        print(article.get("authors"))
        normalized: dict[str, Any] = {
            "title": [clean_html(article.get("name"))],
            "container-title": [clean_html(article.get("journal"))]
            if article.get("journal")
            else [],
            "author": {"given": article.get("authors").get(0), "family": ""},
            "DOI": None,
            "score": (100 / rows) * rank,
            "rank": rank,
            "source": "cyberleninka",
        }
        if year:
            normalized["year"] = year
            normalized["issued"] = {"date-parts": [[int(year)]]}
        result.append(normalized)
    return result


def multi_search(query: str, rows: int = 10) -> list[dict[str, Any]]:
    """Search both Crossref and OpenAlex in parallel and merge candidates."""
    results: dict[str, list[dict[str, Any]]] = {}

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(crossref_search, query, rows): "crossref",
            pool.submit(openalex_search, query, rows): "openalex",
            pool.submit(cyberleninka_search, query, rows): "cyberleninka",
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception:
                results[name] = []

    crossref = results.get("crossref") or []
    openalex = results.get("openalex") or []
    cyberleninka = results.get("cyberleninka") or []
    # deduplicate by DOI: prefer crossref entry when both have the same DOI
    seen_dois: set[str] = set()
    merged: list[dict[str, Any]] = []

    for candidate in crossref:
        doi = (candidate.get("DOI") or "").lower()
        if doi:
            seen_dois.add(doi)
        merged.append(candidate)

    for candidate in openalex:
        doi = (candidate.get("DOI") or "").lower()
        if doi and doi in seen_dois:
            continue
        merged.append(candidate)

    for candidate in cyberleninka:
        merged.append(candidate)
    # re-rank sequentially after merge
    for idx, candidate in enumerate(merged, start=1):
        candidate["rank"] = idx

    return merged
