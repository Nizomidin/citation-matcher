from __future__ import annotations

import argparse
import json
import sys

from citation_matcher.matcher import match_citation


def format_text(result) -> str:
    if result.error:
        return f"Error: {result.error}"
    best = result.best_match
    if not best:
        return "No match found."

    lines = [
        f"Best match (probability: {best['probability']:.1%}):",
        best["formatted_citation"],
        "",
        f"DOI: {best.get('doi') or 'n/a'}",
    ]
    if result.alternatives:
        lines.extend(["", "Alternatives:"])
        for alt in result.alternatives[:4]:
            title = (alt.get("title") or "")[:80]
            lines.append(f"  {alt['probability']:.1%} — {title}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Match a bibliographic description to the most likely article."
    )
    parser.add_argument("query", nargs="?", help="Bibliographic description")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--top", type=int, default=5, help="Number of alternatives")
    args = parser.parse_args(argv)

    query = args.query or sys.stdin.read().strip()
    if not query:
        parser.error("Provide a query as an argument or via stdin.")

    result = match_citation(query)
    if result.alternatives:
        result.alternatives = result.alternatives[: max(args.top - 1, 0)]

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(format_text(result))
    return 1 if result.error else 0


if __name__ == "__main__":
    raise SystemExit(main())
