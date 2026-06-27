from __future__ import annotations

import argparse

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from citation_matcher.config import PROCESSED_DATA_DIR, DEFAULT_MODEL_PATH, MODELS_DIR
from citation_matcher.matcher import FEATURE_COLUMNS, prepare_training_frame


def train(dataset_path=None, model_path=None) -> None:
    dataset = dataset_path or (PROCESSED_DATA_DIR / "ranking_dataset_with_features.csv")
    output = model_path or DEFAULT_MODEL_PATH

    df = prepare_training_frame(pd.read_csv(dataset))
    X = df[FEATURE_COLUMNS].fillna(
        {
            "year_difference": 999,
            "title_similarity": 0,
            "title_token_set_similarity": 0,
            "first_author_similarity": 0,
            "journal_similarity": 0,
            "word_count": 0,
            "script_overlap": 0,
            "source_rank": 999,
            "source_crossref": 0,
            "source_openalex": 0,
            "source_cyberleninka": 0,
            "source_elibrary": 0,
        }
    )
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    base = LogisticRegression(max_iter=1000, class_weight="balanced")
    model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred))
    print(f"Mean predicted probability for positives: {y_prob[y_test == 1].mean():.2f}")
    print(f"Mean predicted probability for negatives: {y_prob[y_test == 0].mean():.2f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output)
    print(f"Model saved to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the citation ranking model.")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    train(dataset_path=args.dataset, model_path=args.output)


if __name__ == "__main__":
    main()
