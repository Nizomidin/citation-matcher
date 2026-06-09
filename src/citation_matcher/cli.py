from __future__ import annotations

import argparse
import json
import sys

from citation_matcher.matcher import match_citation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Match a bibliographic description to the most likely article.",
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Bibliographic description (title, author, year, etc.)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of alternative candidates to include (default: 5)",
    )
    return parser


def format_text(result) -> str:
    if result.error:
        return f"Error: {result.error}"

    best = result.best_match
    if not best:
        return "No match found."

    lines = [
        f"Best match (confidence: {best['confidence']:.1%}):",
        best["formatted_citation"],
        "",
        f"DOI: {best.get('doi') or 'n/a'}",
    ]

    if result.alternatives:
        lines.append("")
        lines.append("Alternatives:")
        for alt in result.alternatives[:4]:
            title = (alt.get("title") or "")[:80]
            lines.append(f"  {alt['confidence']:.1%} — {title}")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    query = args.query
    if not query:
        query = sys.stdin.read().strip()
    if not query:
        parser.error("Provide a bibliographic description as an argument or via stdin.")

    result = match_citation(query)
    if args.top is not None and result.alternatives:
        result.alternatives = result.alternatives[: max(args.top - 1, 0)]

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(format_text(result))

    return 1 if result.error else 0


if __name__ == "__main__":
    raise SystemExit(main())
