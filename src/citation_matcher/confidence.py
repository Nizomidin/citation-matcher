from __future__ import annotations


def relative_confidence(scores: list[float]) -> list[float]:
    """Normalize ML scores into a relative confidence share (sums to 1)."""
    if not scores:
        return []

    total = sum(scores)
    if total <= 0:
        share = 1.0 / len(scores)
        return [share] * len(scores)

    return [score / total for score in scores]
