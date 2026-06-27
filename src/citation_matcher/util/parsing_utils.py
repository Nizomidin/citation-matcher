from html import unescape
import re
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from citation_matcher.config import _TAG_RE

_LATIN_RE = re.compile(r"[A-Za-z]")
_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")


def clean_html(text: str | None) -> str:
    if not text:
        return ""
    text = unescape(str(text))
    text = _TAG_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _parse_authors(names: list[str] | None) -> list[dict[str, str]]:
    authors: list[dict[str, str]] = []
    for name in (names or [])[:10]:
        parts = str(name).rsplit(" ", 1)
        if len(parts) == 2:
            authors.append({"given": parts[0], "family": parts[1]})
        else:
            authors.append({"given": "", "family": parts[0]})
    return authors


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _extract_year(item: dict[str, Any]) -> int | None:
    for field in ("published-print", "published-online", "issued"):
        if field in item:
            try:
                return item[field]["date-parts"][0][0]
            except (KeyError, IndexError, TypeError):
                continue
    year = item.get("year")
    return int(year) if year is not None else None


def _candidate_key(candidate: dict[str, Any]) -> str:
    doi = (candidate.get("DOI") or "").lower()
    if doi:
        return f"doi:{doi}"
    link = (candidate.get("link") or candidate.get("id") or "").lower()
    if link:
        return f"link:{link}"
    title = clean_html((candidate.get("title") or [""])[0]).lower()
    return f"title:{title}"


def extract_year_from_query(query: str) -> int | None:
    match = re.search(r"\b(19|20)\d{2}\b", str(query))
    return int(match.group()) if match else None


def detect_scripts(text: str) -> set[str]:
    """Letter scripts present in text (multilingual strings may have several)."""
    scripts: set[str] = set()
    if _LATIN_RE.search(text):
        scripts.add("latin")
    if _CYRILLIC_RE.search(text):
        scripts.add("cyrillic")
    if _ARABIC_RE.search(text):
        scripts.add("arabic")
    return scripts


def compute_script_overlap(query: str, candidate_title: str) -> float:
    """Jaccard overlap of scripts between query and title (0..100).

    Works better than a binary same_script flag for mixed-language citations.
    """
    query_scripts = detect_scripts(query)
    title_scripts = detect_scripts(candidate_title)
    if not query_scripts and not title_scripts:
        return 100.0
    if not query_scripts or not title_scripts:
        return 0.0
    union = query_scripts | title_scripts
    return 100.0 * len(query_scripts & title_scripts) / len(union)


def query_word_count(query: str) -> int:
    return len(str(query).split())


def parse_elibrary_candidates(html: str, base_url: str = "https://elibrary.ru"):
    soup = BeautifulSoup(html, "html.parser")
    candidates = []

    table = soup.find("table", id="restab")
    if not table:
        return candidates

    rows = table.find_all("tr", id=re.compile(r"^a\d+"))

    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 3:
            continue

        left_cell = cells[0]
        info_cell = cells[1]
        cite_cell = cells[2]

        row_id = row.get("id", "")
        item_id = row_id[1:] if row_id.startswith("a") else None

        number_tag = left_cell.find("b")
        number = int(clean_text(number_tag.get_text())) if number_tag else None

        title_link = info_cell.find("a", href=re.compile(r"/item\.asp\?id=\d+"))
        if not title_link:
            continue

        title = clean_text(title_link.get_text())
        url = urljoin(base_url, title_link.get("href"))

        authors_tag = info_cell.find("i")
        authors = clean_text(authors_tag.get_text()) if authors_tag else ""

        info_copy = BeautifulSoup(str(info_cell), "html.parser")

        for tag in info_copy.find_all(["a", "i"]):
            if tag.name == "a" and not re.search(
                r"/item\.asp\?id=\d+", tag.get("href", "")
            ):
                tag.unwrap()
            else:
                tag.decompose()

        source_text = clean_text(info_copy.get_text())

        year_match = re.search(r"\b(19|20)\d{2}\b", source_text)
        year = int(year_match.group(0)) if year_match else None

        pages_match = re.search(r"С\.\s*([\d\-–—]+)", source_text)
        pages = pages_match.group(1) if pages_match else None

        citations_text = clean_text(cite_cell.get_text())
        citations = int(citations_text) if citations_text.isdigit() else 0

        candidates.append(
            {
                "id": item_id,
                "number": number,
                "title": title,
                "authors": authors,
                "source": source_text,
                "year": year,
                "pages": pages,
                "citations": citations,
                "url": url,
            }
        )

    return candidates
