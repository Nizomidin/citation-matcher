"""Offline evaluation and benchmarking (datasets, rankers, sources)."""

from citation_matcher.benchmark.evaluate import (
    RankingMetrics,
    compute_ranking_metrics,
    evaluate_all_sources_vs_model,
    evaluate_crossref_vs_model,
    evaluate_source_vs_model,
)
from citation_matcher.benchmark.rankers import (
    build_comparison_table,
    default_cache_path as ranker_benchmark_cache_path,
    run_ranker_benchmark,
    summarize_ranker_benchmark,
)
from citation_matcher.benchmark.sources import (
    BENCHMARK_SOURCES,
    candidate_matches_seed,
    default_cache_path as unified_benchmark_cache_path,
    infer_article_lang,
    load_seed_articles,
    run_unified_benchmark,
    summarize_unified_benchmark,
    tag_article,
)

__all__ = [
    "BENCHMARK_SOURCES",
    "RankingMetrics",
    "build_comparison_table",
    "candidate_matches_seed",
    "compute_ranking_metrics",
    "evaluate_all_sources_vs_model",
    "evaluate_crossref_vs_model",
    "evaluate_source_vs_model",
    "infer_article_lang",
    "load_seed_articles",
    "ranker_benchmark_cache_path",
    "run_ranker_benchmark",
    "run_unified_benchmark",
    "summarize_ranker_benchmark",
    "summarize_unified_benchmark",
    "tag_article",
    "unified_benchmark_cache_path",
]
