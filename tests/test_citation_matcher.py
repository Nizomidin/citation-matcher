from __future__ import annotations

from citation_matcher.config import FIXTURES_DIR
from citation_matcher.matcher import (
    FEATURE_COLUMNS,
    author_similarity,
    build_features,
    calibrated_probability,
    compute_journal_similarity,
    compute_year_difference,
    format_citation,
    match_citation,
    normalize_candidate_probabilities,
    prepare_training_frame,
)
from citation_matcher.util.parsing_utils import (
    clean_html,
    compute_script_overlap,
    extract_year_from_query,
    parse_elibrary_candidates,
    query_word_count,
)


def test_extract_year_from_query():
    assert extract_year_from_query("Some paper Smith 2020") == 2020
    assert extract_year_from_query("No year here") is None


def test_author_similarity_partial_match():
    query = "Deep learning Nature Smith J 2020"
    assert author_similarity(query, "John Smith") > 50


def test_build_features_uses_source_rank():
    query = "Attention is all you need Vaswani 2017"
    candidate = {
        "title": ["Attention Is All You Need"],
        "author": [{"given": "Ashish", "family": "Vaswani"}],
        "source_rank": 2,
        "rank": 10,
        "source": "cyberleninka",
        "issued": {"date-parts": [[2017]]},
    }
    features = build_features(query, candidate)
    assert features["source_rank"] == 2
    assert features["source_cyberleninka"] == 1
    assert features["source_elibrary"] == 0


def test_build_features_elibrary_source():
    query = "PEMANFAATAN GOOGLE DATA STUDIO"
    candidate = {
        "title": ["PEMANFAATAN GOOGLE DATA STUDIO UNTUK VISUALISASI DATA"],
        "author": [{"given": "T.A.", "family": "Saputri"}],
        "source_rank": 1,
        "rank": 1,
        "source": "elibrary",
        "year": 2022,
    }
    features = build_features(query, candidate)
    assert features["source_elibrary"] == 1


def test_build_features_new_similarity_fields():
    query = "Deep learning Nature Smith 2015"
    candidate = {
        "title": ["Deep Learning"],
        "author": [{"given": "Yann", "family": "LeCun"}],
        "container-title": ["Nature"],
        "issued": {"date-parts": [[2015]]},
        "source": "crossref",
    }
    features = build_features(query, candidate)
    assert features["journal_similarity"] > 0
    assert features["word_count"] == 5
    assert features["script_overlap"] == 100.0


def test_compute_script_overlap_multilingual():
    # Latin title in query, Cyrillic in title -> partial overlap is not forced to 0/1
    assert compute_script_overlap("Machine learning 2020", "Машинное обучение") == 0.0
    assert compute_script_overlap("AI и ML", "AI и ML") == 100.0
    assert compute_script_overlap("AI and ML", "AI и ML") == 50.0


def test_query_word_count():
    assert query_word_count("one two three") == 3


def test_compute_journal_similarity():
    assert compute_journal_similarity("Deep learning Nature 2015", "Nature") > 50
    assert compute_journal_similarity("title only", None) == 0.0


def test_prepare_training_frame_backfills_new_features():
    import pandas as pd

    frame = prepare_training_frame(
        pd.DataFrame(
            [
                {
                    "query": "Deep learning Nature 2015",
                    "candidate_title": "Deep Learning",
                    "journal": "Nature",
                    "candidate_rank": 1,
                    "source": "crossref",
                }
            ]
        )
    )
    assert frame["journal_similarity"].iloc[0] > 0
    assert frame["word_count"].iloc[0] == 4
    assert frame["script_overlap"].iloc[0] == 100.0


def test_compute_year_difference_missing_year():
    assert compute_year_difference(None, 2020) == 999
    assert compute_year_difference(2020, None) == 999


def test_prepare_training_frame_has_elibrary():
    import pandas as pd

    frame = prepare_training_frame(
        pd.DataFrame([{"source": "elibrary", "candidate_rank": 1}])
    )
    assert "source_elibrary" in frame.columns
    assert frame["source_elibrary"].iloc[0] == 1


def test_calibrated_probability_exact_match():
    assert calibrated_probability(0.001, 100.0) == 0.99
    assert calibrated_probability(0.4, 50.0) == 0.4


def test_normalize_candidate_probabilities_uses_model_scores():
    probs = normalize_candidate_probabilities([0.2, 0.02, 0.0], [10.0, 20.0, 30.0])
    assert abs(sum(probs) - 1.0) < 1e-9
    assert probs[0] > probs[1] > probs[2]


def test_normalize_candidate_probabilities_falls_back_to_title_similarity():
    probs = normalize_candidate_probabilities([0.0, 0.0, 0.0], [10.0, 20.0, 30.0])
    assert abs(sum(probs) - 1.0) < 1e-9
    assert probs[0] < probs[1] < probs[2]


def test_format_citation_elibrary_link():
    item = {
        "author": [{"given": "T.A.", "family": "Saputri"}],
        "title": ["Sample Article"],
        "container-title": ["Journal"],
        "year": 2022,
        "source": "elibrary",
        "link": "/item.asp?id=123",
    }
    citation = format_citation(item)
    assert "https://elibrary.ru/item.asp?id=123" in citation


def test_format_citation_crossref():
    item = {
        "author": [{"given": "Ashish", "family": "Vaswani"}],
        "title": ["Attention Is All You Need"],
        "container-title": ["Neural Information Processing Systems"],
        "issued": {"date-parts": [[2017]]},
        "DOI": "10.1234/example",
    }
    citation = format_citation(item)
    assert "Vaswani" in citation
    assert "2017" in citation


def test_clean_html_strips_tags():
    raw = "<b>Применение</b> <b>современных</b> технологий"
    assert clean_html(raw) == "Применение современных технологий"


def test_parse_elibrary_candidates_from_fixture():
    html = (FIXTURES_DIR / "response.html").read_text(encoding="utf-8")
    results = parse_elibrary_candidates(html)
    assert results
    assert results[0]["id"]
    assert results[0]["title"]


def test_elibrary_results_match_query():
    from citation_matcher.elibrary import _results_match_query

    query = "РУШДИ ИНДУСТРИАЛӢ-АГРАРИИ ИҚТИСОДИЁТИ МИЛЛӢ"
    assert _results_match_query(
        query,
        [{"title": "РУШДИ ИНДУСТРИАЛӢ-АГРАРИИ ИҚТИСОДИЁТИ МИЛЛӢ: ИМКОНИЯТҲО ВА МУАММОҲО"}],
    )
    assert not _results_match_query(
        query,
        [{"title": "SHARPENING THE RESOLUTION ON DATA MATTERS"}],
    )


def test_feature_columns_count():
    assert len(FEATURE_COLUMNS) == 7


def test_match_citation_empty_query():
    result = match_citation("   ")
    assert result.error == "Empty query"


def test_compute_ranking_metrics():
    from citation_matcher.benchmark import compute_ranking_metrics

    metrics = compute_ranking_metrics([1, 2, 5, None])
    assert metrics.count == 3
    assert metrics.top1 == 1 / 3
    assert metrics.top3 == 2 / 3
    assert abs(metrics.mrr - (1 + 0.5 + 0.2) / 3) < 1e-9


def test_candidate_matches_seed_by_doi():
    from citation_matcher.benchmark import candidate_matches_seed

    seed = {
        "id": "10.1234/test",
        "source": "crossref",
        "title": "Deep learning",
        "year": 2015,
        "first_author": "Yann LeCun",
    }
    candidate = {
        "source": "crossref",
        "DOI": "10.1234/test",
        "title": ["Deep learning"],
        "author": [{"given": "Yann", "family": "LeCun"}],
        "issued": {"date-parts": [[2015]]},
    }
    assert candidate_matches_seed(candidate, seed)


def test_candidate_matches_seed_by_fuzzy_title():
    from citation_matcher.benchmark import candidate_matches_seed

    seed = {
        "id": "/article/n/test",
        "source": "cyberleninka",
        "title": "Публичный договор как элемент защиты слабой стороны",
        "year": 2012,
        "first_author": "Каширин Игорь Олегович",
    }
    candidate = {
        "source": "crossref",
        "title": ["Публичный договор как элемент защиты слабой стороны"],
        "author": [{"given": "Игорь", "family": "Каширин"}],
        "issued": {"date-parts": [[2012]]},
    }
    assert candidate_matches_seed(candidate, seed)


def test_infer_article_lang():
    from citation_matcher.benchmark import infer_article_lang, tag_article

    assert infer_article_lang(tag_article({"source": "crossref", "title": "Deep learning"})) == "en"
    assert (
        infer_article_lang(
            tag_article({"source": "cyberleninka", "title": "Публичный договор"})
        )
        == "ru"
    )


def test_build_comparison_table():
    import pandas as pd

    from citation_matcher.benchmark import build_comparison_table

    summary = pd.DataFrame(
        [
            {
                "error_type": "No errors",
                "ranker": "crossref",
                "top1": 0.8,
                "top3": 0.9,
                "recall@10": 1.0,
                "count": 10,
            },
            {
                "error_type": "No errors",
                "ranker": "matcher",
                "top1": 0.9,
                "top3": 0.9,
                "recall@10": 1.0,
                "count": 10,
            },
        ]
    )
    table = build_comparison_table(summary)
    assert "Crossref Top-1" in table.columns
    assert table.iloc[0]["n"] == 10


def test_summarize_unified_benchmark_by_language():
    import pandas as pd

    from citation_matcher.benchmark import summarize_unified_benchmark

    raw = pd.DataFrame(
        [
            {
                "lang": "en",
                "error_type": "No errors",
                "crossref_rank": 1,
                "model_crossref_rank": 1,
                "multi_rank": 1,
                "matcher_rank": 1,
            },
            {
                "lang": "ru",
                "error_type": "No errors",
                "cyberleninka_rank": 2,
                "model_cyberleninka_rank": 1,
                "multi_rank": -1,
                "matcher_rank": 1,
            },
        ]
    )
    summary = summarize_unified_benchmark(
        raw, sources=("crossref", "cyberleninka"), include_model=True
    )
    assert not summary.empty
    assert set(summary["lang"]) >= {"all", "en", "ru"}
    en = summary[(summary["lang"] == "en") & (summary["ranker"] == "crossref")]
    assert float(en.iloc[0]["top1"]) == 1.0
