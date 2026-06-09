from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from citation_matcher.confidence import relative_confidence
from citation_matcher.features import FEATURE_COLUMNS, build_features
from citation_matcher.format import candidate_summary
from citation_matcher.paths import DEFAULT_MODEL_PATH


class Ranker:
    def __init__(self, model_path: Path | None = None):
        path = model_path or DEFAULT_MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found at {path}. Run: python -m citation_matcher.train"
            )
        self.model = joblib.load(path)

    def score_candidates(
        self, query: str, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        scored: list[dict[str, Any]] = []
        for candidate in candidates:
            features = build_features(query, candidate)
            frame = pd.DataFrame([features])[FEATURE_COLUMNS]
            probability = float(self.model.predict_proba(frame)[0][1])
            scored.append({**candidate, "ml_score": probability})

        scored = sorted(scored, key=lambda item: item["ml_score"], reverse=True)
        confidences = relative_confidence([item["ml_score"] for item in scored])
        for item, confidence in zip(scored, confidences):
            item["confidence"] = confidence
        return scored

    def summarize(
        self, query: str, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return [
            candidate_summary(
                candidate,
                probability=candidate["ml_score"],
                confidence=candidate["confidence"],
            )
            for candidate in self.score_candidates(query, candidates)
        ]
