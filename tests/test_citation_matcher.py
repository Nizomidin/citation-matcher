from __future__ import annotations

from citation_matcher.confidence import relative_confidence
from citation_matcher.features import (
    author_similarity,
    build_features,
    extract_year_from_query,
)
from citation_matcher.text import clean_html
from citation_matcher.format import format_citation
from citation_matcher.matcher import match_citation


def test_extract_year_from_query():
    assert extract_year_from_query("Some paper Smith 2020") == 2020
    assert extract_year_from_query("No year here") is None


def test_author_similarity_partial_match():
    query = "Deep learning Nature Smith J 2020"
    assert author_similarity(query, "John Smith") > 50


def test_build_features():
    query = "Attention is all you need Vaswani 2017"
    candidate = {
        "title": ["Attention Is All You Need"],
        "author": [{"given": "Ashish", "family": "Vaswani"}],
        "score": 100.0,
        "rank": 1,
        "issued": {"date-parts": [[2017]]},
    }
    features = build_features(query, candidate)
    assert features["title_similarity"] > 0
    assert features["candidate_rank"] == 1
    assert features["year_difference"] == 0


def test_format_citation():
    item = {
        "author": [{"given": "Ashish", "family": "Vaswani"}],
        "title": ["Attention Is All You Need"],
        "container-title": ["Neural Information Processing Systems"],
        "issued": {"date-parts": [[2017]]},
        "DOI": "10.1234/example",
    }
    citation = format_citation(item)
    assert "Vaswani" in citation
    assert "Attention Is All You Need" in citation
    assert "2017" in citation
    assert "10.1234/example" in citation


def test_relative_confidence_sums_to_one():
    scores = [0.39, 0.006, 0.001, 0.0001]
    confidences = relative_confidence(scores)
    assert len(confidences) == len(scores)
    assert abs(sum(confidences) - 1.0) < 1e-9
    assert confidences[0] > confidences[1] > confidences[2]


def test_relative_confidence_zero_scores():
    confidences = relative_confidence([0.0, 0.0, 0.0])
    assert abs(sum(confidences) - 1.0) < 1e-9


def test_clean_html_strips_tags():
    raw = "<b>Применение</b> <b>современных</b> технологий"
    assert clean_html(raw) == "Применение современных технологий"


def test_clean_html_decodes_entities():
    assert clean_html("Deep learning &amp; AI") == "Deep learning & AI"


def test_match_citation_empty_query():
    result = match_citation("   ")
    assert result.error == "Empty query"
    assert result.best_match is None
