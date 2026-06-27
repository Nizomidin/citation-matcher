from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from citation_matcher.config import DEFAULT_MODEL_PATH, PROCESSED_DATA_DIR, REPORTS_DIR
from citation_matcher.matcher import (
    FEATURE_COLUMNS,
    Ranker,
    _enrich_missing_features,
    calibrated_probability,
    normalize_candidate_probabilities,
)
from citation_matcher.search import crossref_search
from citation_matcher.util.parsing_utils import clean_html

DEFAULT_EVAL_SOURCES = ("crossref", "cyberleninka", "elibrary")


@dataclass
class RankingMetrics:
    top1: float
    top3: float
    top10: float
    mrr: float
    count: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "top1": self.top1,
            "top3": self.top3,
            "top10": self.top10,
            "mrr": self.mrr,
            "count": self.count,
        }


def _positive_rank(ranks: pd.Series, labels: pd.Series) -> int | None:
    positive = ranks[labels == 1]
    if positive.empty:
        return None
    return int(positive.iloc[0])


def _fill_feature_nans(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[FEATURE_COLUMNS].fillna(
        {
            "year_difference": 999,
            "title_similarity": 0,
            "title_token_set_similarity": 0,
            "first_author_similarity": 0,
            "journal_similarity": 0,
            "word_count": 0,
            "script_overlap": 0,
        }
    )


def _model_scores(group: pd.DataFrame, ranker: Ranker) -> pd.Series:
    frame = _enrich_missing_features(group)
    features = _fill_feature_nans(frame)
    raw_probs = ranker.model.predict_proba(features)[:, 1]
    title_sims = frame["title_token_set_similarity"].tolist()
    normalized = normalize_candidate_probabilities(raw_probs.tolist(), title_sims)
    scores = [
        calibrated_probability(prob, float(title_sim))
        for prob, title_sim in zip(normalized, title_sims, strict=True)
    ]
    return pd.Series(scores, index=group.index)


def compute_ranking_metrics(rank_of_positive: list[int | None]) -> RankingMetrics:
    valid = [rank for rank in rank_of_positive if rank is not None and rank > 0]
    if not valid:
        return RankingMetrics(0.0, 0.0, 0.0, 0.0, 0)

    count = len(valid)
    top1 = sum(rank == 1 for rank in valid) / count
    top3 = sum(rank <= 3 for rank in valid) / count
    top10 = sum(rank <= 10 for rank in valid) / count
    mrr = sum(1.0 / rank for rank in valid) / count
    return RankingMetrics(top1, top3, top10, mrr, count)


def evaluate_source_vs_model(
    dataset_path: Path | None = None,
    results_path: Path | None = None,
    *,
    model_path: Path | None = None,
    source: str = "crossref",
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Compare a source's native ranking with the ML ranker on saved candidates."""
    dataset_path = dataset_path or PROCESSED_DATA_DIR / "ranking_dataset_with_features.csv"
    results_path = results_path or REPORTS_DIR / "results.csv"

    df = pd.read_csv(dataset_path)
    df = df[df["source"] == source].copy()
    if df.empty:
        raise ValueError(f"No rows for source={source!r} in {dataset_path}")

    ranker = Ranker(model_path or DEFAULT_MODEL_PATH)
    results = pd.read_csv(results_path)
    error_by_query = (
        results[results["Source"] == source][["Query", "Type of Error"]]
        .drop_duplicates("Query")
        .set_index("Query")["Type of Error"]
    )

    per_query_rows: list[dict[str, Any]] = []
    source_ranks: list[int | None] = []
    model_ranks: list[int | None] = []

    for query, group in df.groupby("query", sort=False):
        labels = group["label"]
        if (labels == 1).sum() == 0:
            continue

        native_rank = _positive_rank(group["candidate_rank"], labels)
        scores = _model_scores(group, ranker)
        ordered = group.assign(ml_score=scores).sort_values(
            "ml_score", ascending=False
        )
        model_rank = _positive_rank(
            pd.Series(range(1, len(ordered) + 1), index=ordered.index),
            ordered["label"],
        )

        source_ranks.append(native_rank)
        model_ranks.append(model_rank)

        true_id = group.loc[labels == 1, "true_id"].iloc[0]
        per_query_rows.append(
            {
                "source": source,
                "query": query,
                "error_type": error_by_query.get(query, "unknown"),
                "true_id": true_id,
                "source_rank": native_rank,
                "model_rank": model_rank,
                "improved": (
                    native_rank is not None
                    and model_rank is not None
                    and model_rank < native_rank
                ),
                "regressed": (
                    native_rank is not None
                    and model_rank is not None
                    and model_rank > native_rank
                ),
            }
        )

    per_query = pd.DataFrame(per_query_rows)
    summary_rows = [
        {
            "source": source,
            "ranker": source,
            "error_type": "all",
            **compute_ranking_metrics(source_ranks).as_dict(),
        },
        {
            "source": source,
            "ranker": "model",
            "error_type": "all",
            **compute_ranking_metrics(model_ranks).as_dict(),
        },
    ]
    for error_type, subset in per_query.groupby("error_type", sort=False):
        summary_rows.append(
            {
                "source": source,
                "ranker": source,
                "error_type": error_type,
                **compute_ranking_metrics(subset["source_rank"].tolist()).as_dict(),
            }
        )
        summary_rows.append(
            {
                "source": source,
                "ranker": "model",
                "error_type": error_type,
                **compute_ranking_metrics(subset["model_rank"].tolist()).as_dict(),
            }
        )

    summary = pd.DataFrame(summary_rows)
    examples = _pick_comparison_examples(per_query, df, ranker, source=source)
    return summary, per_query, examples


def evaluate_all_sources_vs_model(
    dataset_path: Path | None = None,
    results_path: Path | None = None,
    *,
    model_path: Path | None = None,
    sources: tuple[str, ...] = DEFAULT_EVAL_SOURCES,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[dict[str, Any]]], list[str]]:
    """Evaluate native vs model ranking for each bibliographic source."""
    summaries: list[pd.DataFrame] = []
    per_query_frames: list[pd.DataFrame] = []
    examples_by_source: dict[str, list[dict[str, Any]]] = {}
    skipped: list[str] = []

    for source in sources:
        try:
            summary, per_query, examples = evaluate_source_vs_model(
                dataset_path,
                results_path,
                model_path=model_path,
                source=source,
            )
        except ValueError:
            skipped.append(source)
            continue
        summaries.append(summary)
        per_query_frames.append(per_query)
        examples_by_source[source] = examples

    if not summaries:
        raise ValueError("No evaluation data found for any source.")

    combined_summary = pd.concat(summaries, ignore_index=True)
    combined_per_query = pd.concat(per_query_frames, ignore_index=True)
    return combined_summary, combined_per_query, examples_by_source, skipped


def evaluate_crossref_vs_model(
    dataset_path: Path | None = None,
    results_path: Path | None = None,
    *,
    model_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Backward-compatible alias for Crossref-only evaluation."""
    return evaluate_source_vs_model(
        dataset_path,
        results_path,
        model_path=model_path,
        source="crossref",
    )


def _pick_comparison_examples(
    per_query: pd.DataFrame,
    dataset: pd.DataFrame,
    ranker: Ranker,
    *,
    source: str,
    limit: int = 3,
) -> list[dict[str, Any]]:
    improved = per_query[per_query["improved"]].copy()
    if improved.empty:
        improved = per_query[per_query["source_rank"] > 1].head(limit)
    else:
        improved["delta"] = improved["source_rank"] - improved["model_rank"]
        improved = improved.sort_values("delta", ascending=False).head(limit)

    examples: list[dict[str, Any]] = []
    for _, row in improved.iterrows():
        group = dataset[dataset["query"] == row["query"]].copy()
        scores = _model_scores(group, ranker)
        group = group.assign(ml_score=scores)
        examples.append(
            {
                "source": source,
                "query": row["query"],
                "error_type": row["error_type"],
                "true_id": row["true_id"],
                "source_rank": row["source_rank"],
                "model_rank": row["model_rank"],
                "native_top": _format_ranked_candidates(
                    group.sort_values("candidate_rank"), row["true_id"]
                ),
                "model_top": _format_ranked_candidates(
                    group.sort_values("ml_score", ascending=False), row["true_id"]
                ),
            }
        )
    return examples


def _format_ranked_candidates(
    group: pd.DataFrame, true_id: str, *, top_n: int = 5
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, (_, row) in enumerate(group.head(top_n).iterrows(), start=1):
        rows.append(
            {
                "rank": rank,
                "title": str(row["candidate_title"])[:90],
                "year": row.get("candidate_year"),
                "author": row.get("candidate_author"),
                "is_correct": str(row["candidate_id"]) == str(true_id),
                "ml_score": float(row["ml_score"]) if "ml_score" in row else None,
                "native_rank": int(row["candidate_rank"]),
            }
        )
    return rows


def _candidate_from_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": clean_html(item.get("title", "")),
        "year": item.get("year"),
        "author": item.get("first_author") or item.get("author"),
        "doi": item.get("doi") or item.get("DOI"),
        "is_correct": bool(item.get("is_correct")),
        "score": item.get("ml_score") or item.get("probability"),
        "native_rank": item.get("native_rank") or item.get("rank"),
    }


def compare_live_query(
    query: str,
    *,
    rows: int = 10,
    model_path: Path | None = None,
) -> dict[str, Any]:
    """Fetch Crossref candidates live and compare native vs model order."""
    candidates = crossref_search(query, rows=rows)
    ranker = Ranker(model_path or DEFAULT_MODEL_PATH)
    ranked = ranker.summarize(query, candidates)

    native_top = []
    for item in candidates[:rows]:
        native_top.append(
            _candidate_from_item(
                {
                    "title": item.get("title", [""])[0],
                    "year": _extract_year_from_candidate(item),
                    "author": _extract_author_from_candidate(item),
                    "doi": item.get("DOI"),
                    "native_rank": item.get("source_rank"),
                }
            )
        )

    model_top = []
    for item in ranked[:rows]:
        model_top.append(
            _candidate_from_item(
                {
                    "title": item.get("title"),
                    "year": item.get("year"),
                    "author": item.get("first_author"),
                    "doi": item.get("doi"),
                    "probability": item.get("probability"),
                }
            )
        )

    return {
        "query": query,
        "source": "crossref",
        "native_top": native_top,
        "model_top": model_top,
    }


def _extract_year_from_candidate(item: dict[str, Any]) -> int | None:
    for field in ("published-print", "published-online", "issued"):
        if field in item:
            return item[field]["date-parts"][0][0]
    return None


def _extract_author_from_candidate(item: dict[str, Any]) -> str | None:
    authors = item.get("author") or []
    if not authors:
        return None
    first = authors[0]
    if isinstance(first, str):
        return first
    return f"{first.get('given', '')} {first.get('family', '')}".strip() or None
