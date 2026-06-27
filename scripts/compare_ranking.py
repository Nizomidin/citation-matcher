from __future__ import annotations

import argparse
import json

import pandas as pd

from citation_matcher.benchmark.evaluate import (
    DEFAULT_EVAL_SOURCES,
    compare_live_query,
    evaluate_all_sources_vs_model,
    evaluate_source_vs_model,
)

ERROR_TYPE_ORDER = [
    "No errors",
    "Typo in title error",
    "No author error",
    "No year error",
    "Title with dropped word error",
]


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def _source_label(source: str) -> str:
    labels = {
        "crossref": "Crossref",
        "cyberleninka": "CyberLeninka",
        "elibrary": "eLibrary",
    }
    return labels.get(source, source)


def _ordered_error_types(summary) -> list[str]:
    present = [t for t in ERROR_TYPE_ORDER if t in set(summary["error_type"])]
    extra = sorted(
        t for t in summary["error_type"].unique() if t not in ERROR_TYPE_ORDER and t != "all"
    )
    return present + extra


def _print_metric_row(ranker_name: str, row) -> None:
    print(
        f"  {ranker_name:12s}  "
        f"top-1={_pct(row['top1']):>6s}  "
        f"top-3={_pct(row['top3']):>6s}  "
        f"top-10={_pct(row['top10']):>6s}  "
        f"MRR={row['mrr']:.3f}  "
        f"n={int(row['count'])}"
    )


def _short_error_label(error_type: str) -> str:
    labels = {
        "No errors": "Без ошибок",
        "Typo in title error": "Опечатка в заголовке",
        "No author error": "Без автора",
        "No year error": "Без года",
        "Title with dropped word error": "Пропущенное слово",
        "all": "Все",
    }
    return labels.get(error_type, error_type)


def _build_comparison_table(summary: pd.DataFrame, *, include_all: bool = False) -> pd.DataFrame:
    frame = summary.copy()
    if include_all:
        frame = frame[frame["error_type"] == "all"]
    else:
        frame = frame[frame["error_type"] != "all"]

    rows: list[dict[str, object]] = []
    for (error_type, source), group in frame.groupby(
        ["error_type", "source"], sort=False
    ):
        native = group[group["ranker"] == source]
        model = group[group["ranker"] == "model"]
        if native.empty or model.empty:
            continue
        native_row = native.iloc[0]
        model_row = model.iloc[0]
        rows.append(
            {
                "Ошибка": _short_error_label(str(error_type)),
                "Источник": _source_label(str(source)),
                "Native top-1": _pct(float(native_row["top1"])),
                "Model top-1": _pct(float(model_row["top1"])),
                "Δ top-1": _pct(float(model_row["top1"]) - float(native_row["top1"])),
                "Native top-3": _pct(float(native_row["top3"])),
                "Model top-3": _pct(float(model_row["top3"])),
                "Native MRR": f"{float(native_row['mrr']):.3f}",
                "Model MRR": f"{float(model_row['mrr']):.3f}",
                "n": int(native_row["count"]),
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    order = [_short_error_label(t) for t in ERROR_TYPE_ORDER] + ["Все"]
    table["Ошибка"] = pd.Categorical(
        table["Ошибка"], categories=order, ordered=True
    )
    return table.sort_values(["Ошибка", "Источник"]).reset_index(drop=True)


def _print_dataframe_table(table: pd.DataFrame, title: str) -> None:
    if table.empty:
        return
    print(f"\n{title}\n")
    print(table.to_string(index=False))
    print()


def print_summary_tables(
    summary: pd.DataFrame,
    *,
    skipped: list[str] | None = None,
    group_by: str = "error",
) -> None:
    if skipped:
        for source in skipped:
            print(
                f"Примечание: {_source_label(source)} – нет данных в датасете. "
                f"Соберите: python scripts/build_dataset.py --{source} N\n"
            )

    if group_by in ("error", "both"):
        _print_dataframe_table(
            _build_comparison_table(summary, include_all=False),
            "Сравнение по типам ошибок",
        )
    if group_by in ("source", "both"):
        _print_dataframe_table(
            _build_comparison_table(summary, include_all=True),
            "Сводка по источникам (все ошибки)",
        )


def print_summary_by_source(summary, *, skipped: list[str] | None = None) -> None:
    print("\n=== By source ===\n")
    if skipped:
        for source in skipped:
            print(
                f"Note: {_source_label(source)} skipped – "
                f"no rows in dataset. "
                f"Rebuild with: python scripts/build_dataset.py --{source} N\n"
            )

    for source in summary["source"].unique():
        source_summary = summary[summary["source"] == source]
        print(f"### {_source_label(source)} ###\n")
        for error_type in ["all"] + _ordered_error_types(source_summary):
            block = source_summary[source_summary["error_type"] == error_type]
            if block.empty:
                continue
            print(f"--- {error_type} ---")
            for _, row in block.iterrows():
                ranker_name = (
                    _source_label(source)
                    if row["ranker"] == source
                    else "model"
                )
                _print_metric_row(ranker_name, row)
            print()


def print_summary_by_error(summary, *, skipped: list[str] | None = None) -> None:
    print("\n=== By error type ===\n")
    if skipped:
        for source in skipped:
            print(
                f"Note: {_source_label(source)} skipped – "
                f"no rows in dataset. "
                f"Rebuild with: python scripts/build_dataset.py --{source} N\n"
            )

    error_summary = summary[summary["error_type"] != "all"]
    for error_type in _ordered_error_types(error_summary):
        block = error_summary[error_summary["error_type"] == error_type]
        if block.empty:
            continue
        print(f"### {error_type} ###\n")
        for source in block["source"].unique():
            source_block = block[block["source"] == source].sort_values(
                "ranker", key=lambda s: s.map({source: 0, "model": 1})
            )
            print(f"  {_source_label(source)}:")
            for _, row in source_block.iterrows():
                ranker_name = (
                    _source_label(source)
                    if row["ranker"] == source
                    else "model"
                )
                _print_metric_row(f"    {ranker_name}", row)
            print()


def print_summary(summary, *, skipped: list[str] | None = None) -> None:
    print("\n=== Native ranking vs Model (saved dataset) ===")
    print_summary_by_error(summary, skipped=skipped)
    print_summary_by_source(summary, skipped=skipped)


def print_examples(examples_by_source: dict[str, list[dict]]) -> None:
    if not any(examples_by_source.values()):
        return
    print("=== Examples where ranking differs ===\n")
    for source, examples in examples_by_source.items():
        if not examples:
            continue
        print(f"### {_source_label(source)} ###\n")
        for example in examples:
            print(f"Query: {example['query']}")
            print(
                f"Error type: {example['error_type']} | "
                f"{_source_label(source)} rank of correct: {example['source_rank']} → "
                f"Model rank: {example['model_rank']}"
            )
            print(f"  {_source_label(source)} order:")
            for item in example["native_top"]:
                mark = "✓" if item["is_correct"] else " "
                print(
                    f"    {item['rank']:>2}. [{mark}] "
                    f"{item['title']} ({item.get('year', 'n/a')})"
                )
            print("  Model order:")
            for item in example["model_top"]:
                mark = "✓" if item["is_correct"] else " "
                score = item.get("ml_score")
                score_text = f"{score * 100:.1f}%" if score is not None else "n/a"
                print(
                    f"    {item['rank']:>2}. [{mark}] {score_text:>6s} "
                    f"{item['title']} ({item.get('year', 'n/a')})"
                )
            print()


def print_live(result: dict) -> None:
    source = result.get("source", "crossref")
    print(f"\n=== Live query ({_source_label(source)}) ===\n{result['query']}\n")
    print(f"{_source_label(source)} order:")
    for item in result["native_top"]:
        print(
            f"  {item['native_rank']:>2}. {item['title']} "
            f"({item.get('year', 'n/a')})"
        )
    print("\nModel order:")
    for rank, item in enumerate(result["model_top"], start=1):
        score = item.get("score")
        score_text = f"{score * 100:.1f}%" if score is not None else "n/a"
        print(
            f"  {rank:>2}. {score_text:>6s} {item['title']} "
            f"({item.get('year', 'n/a')})"
        )
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare native source ranking with the ML ranker."
    )
    parser.add_argument(
        "--source",
        choices=DEFAULT_EVAL_SOURCES,
        help="Evaluate a single source (default: all available in dataset).",
    )
    parser.add_argument(
        "--query",
        action="append",
        help="Live bibliographic query via Crossref (can be repeated).",
    )
    parser.add_argument("--rows", type=int, default=10, help="Candidates per query.")
    parser.add_argument("--json", action="store_true", help="Output JSON.")
    parser.add_argument(
        "--format",
        choices=("table", "text"),
        default="table",
        help="Output format for summary (default: table).",
    )
    parser.add_argument(
        "--group-by",
        choices=("error", "source", "both"),
        default="error",
        help="Summary layout: by error type, by source totals, or both.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save per-error summary table to CSV.",
    )
    parser.add_argument(
        "--no-dataset",
        action="store_true",
        help="Skip evaluation on the saved dataset.",
    )
    args = parser.parse_args(argv)

    payload: dict = {}

    if not args.no_dataset:
        if args.source:
            summary, per_query, examples = evaluate_source_vs_model(source=args.source)
            examples_by_source = {args.source: examples}
            skipped: list[str] = []
        else:
            summary, per_query, examples_by_source, skipped = (
                evaluate_all_sources_vs_model()
            )

        payload["summary"] = summary.to_dict(orient="records")
        payload["skipped_sources"] = skipped
        payload["improved_count"] = int(per_query["improved"].sum())
        payload["regressed_count"] = int(per_query["regressed"].sum())
        payload["improved_by_source"] = (
            per_query.groupby("source")["improved"].sum().astype(int).to_dict()
        )
        payload["improved_by_error"] = (
            per_query.groupby("error_type")["improved"].sum().astype(int).to_dict()
        )
        payload["examples"] = examples_by_source

        if not args.json:
            if args.format == "table":
                print("\n=== Native ranking vs Model ===")
                print_summary_tables(summary, skipped=skipped, group_by=args.group_by)
            else:
                if args.group_by in ("error", "both"):
                    print("\n=== Native ranking vs Model (saved dataset) ===")
                    print_summary_by_error(summary, skipped=skipped)
                if args.group_by in ("source", "both"):
                    print_summary_by_source(
                        summary, skipped=skipped if args.group_by == "source" else None
                    )
            print(
                f"Улучшено: {payload['improved_count']} | "
                f"Ухудшено: {payload['regressed_count']}"
            )
            if args.format == "text":
                for source, count in payload["improved_by_source"].items():
                    if count:
                        print(f"  {_source_label(source)}: {count}")
                for error_type, count in payload["improved_by_error"].items():
                    if count:
                        print(f"  {error_type}: {count}")
                print_examples(examples_by_source)

        if args.output:
            table = _build_comparison_table(summary, include_all=False)
            table.to_csv(args.output, index=False)

    if args.query:
        payload["live"] = [
            compare_live_query(query, rows=args.rows) for query in args.query
        ]
        if not args.json:
            for result in payload["live"]:
                print_live(result)

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.no_dataset and not args.query:
        parser.error("Provide --query or omit --no-dataset.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

