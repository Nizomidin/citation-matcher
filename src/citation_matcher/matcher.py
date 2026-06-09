from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from citation_matcher.ranker import Ranker
from citation_matcher.search import multi_search


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
    rows: int = 10,
) -> MatchResult:
    """Find the most likely article for a bibliographic description."""
    query = query.strip()
    if not query:
        return MatchResult(query=query, best_match=None, error="Empty query")

    candidates = multi_search(query, rows=rows)
    if not candidates:
        return MatchResult(
            query=query,
            best_match=None,
            error="No candidates found (Crossref and OpenAlex returned nothing)",
        )

    try:
        ranker = Ranker(model_path=model_path)
    except FileNotFoundError as exc:
        return MatchResult(query=query, best_match=None, error=str(exc))

    ranked = ranker.summarize(query, candidates)
    return MatchResult(
        query=query,
        best_match=ranked[0],
        alternatives=ranked[1:],
    )
