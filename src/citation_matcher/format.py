from __future__ import annotations

from typing import Any

from citation_matcher.features import extract_first_author, extract_year
from citation_matcher.text import clean_html


def format_authors(item: dict[str, Any]) -> str:
    authors = item.get("author") or []
    if not authors:
        return ""

    formatted: list[str] = []
    for author in authors[:10]:
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


def format_pages(item: dict[str, Any]) -> str | None:
    page = item.get("page")
    if page:
        return str(page)

    if "article-number" in item:
        return f"Art. {item['article-number']}"
    return None


def format_citation(item: dict[str, Any]) -> str:
    """Build a human-readable bibliographic reference from Crossref metadata."""
    authors = format_authors(item)
    title = clean_html(item.get("title", [""])[0])
    journal = clean_html(item.get("container-title", [""])[0]) if item.get("container-title") else ""
    year = extract_year(item)
    volume = item.get("volume")
    issue = item.get("issue")
    pages = format_pages(item)
    doi = item.get("DOI")

    parts: list[str] = []
    if authors:
        parts.append(authors)
    if title:
        parts.append(title + ".")
    if journal:
        parts.append(journal + ".")
    if year:
        parts.append(str(year) + ".")
    if volume:
        vol_part = f"Vol. {volume}"
        if issue:
            vol_part += f" ({issue})"
        parts.append(vol_part + ".")
    if pages:
        parts.append(f"P. {pages}.")
    if doi:
        parts.append(f"DOI: {doi}")

    return " ".join(parts)


def candidate_summary(
    item: dict[str, Any],
    *,
    probability: float,
    confidence: float,
) -> dict[str, Any]:
    return {
        "title": clean_html(item.get("title", [""])[0]),
        "doi": item.get("DOI"),
        "year": extract_year(item),
        "authors": format_authors(item),
        "first_author": extract_first_author(item),
        "journal": clean_html(item.get("container-title", [""])[0])
        if item.get("container-title")
        else None,
        "probability": round(probability, 4),
        "confidence": round(confidence, 4),
        "formatted_citation": format_citation(item),
        "source": item.get("source", "crossref"),
    }
