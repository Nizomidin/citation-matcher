"""Train the citation ranking model."""

from __future__ import annotations

import argparse

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from citation_matcher.features import FEATURE_COLUMNS
from citation_matcher.paths import DATA_DIR, DEFAULT_MODEL_PATH, MODELS_DIR


def train(dataset_path=None, model_path=None) -> None:
    dataset = dataset_path or (DATA_DIR / "ranking_dataset_with_features-v2.csv")
    output = model_path or DEFAULT_MODEL_PATH

    df = pd.read_csv(dataset)
    X = df[FEATURE_COLUMNS].fillna(
        {
            "year_difference": 999,
            "title_similarity": 0,
            "title_token_set_similarity": 0,
            "crossref_score": 0,
            "candidate_rank": 999,
            "first_author_similarity": 0,
        }
    )
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))
    for feature, coef in zip(FEATURE_COLUMNS, model.coef_[0]):
        print(f"{feature}: {coef:.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output)
    print(f"Model saved to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the citation ranking model.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to the feature dataset CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the trained model",
    )
    args = parser.parse_args()
    train(
        dataset_path=args.dataset,
        model_path=args.output,
    )


if __name__ == "__main__":
    main()
