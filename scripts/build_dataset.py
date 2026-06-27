from __future__ import annotations

import argparse
import logging

from citation_matcher.config import PROCESSED_DATA_DIR, REPORTS_DIR
from citation_matcher.dataset import build_feature_dataset
from citation_matcher.elibrary import refresh_elibrary_cookies

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ranking dataset with features.")
    parser.add_argument("--crossref", type=int, default=50)
    parser.add_argument("--cyberleninka", type=int, default=50)
    parser.add_argument("--elibrary", type=int, default=0)
    parser.add_argument("--refresh-elibrary-cookies", action="store_true")
    args = parser.parse_args()

    if args.refresh_elibrary_cookies or args.elibrary > 0:
        if not refresh_elibrary_cookies():
            logger.warning(
                "Failed to refresh eLibrary cookies (captcha or missing proxy)."
            )

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    results_df, summary_df, ranking_features_df = build_feature_dataset(
        args.crossref, args.cyberleninka, args.elibrary
    )

    results_df.to_csv(REPORTS_DIR / "results.csv", index=False)
    summary_df.to_csv(REPORTS_DIR / "summary.csv")
    ranking_features_df.to_csv(
        PROCESSED_DATA_DIR / "ranking_dataset_with_features.csv", index=False
    )
    print(summary_df)
    logger.info("Saved dataset files.")


if __name__ == "__main__":
    main()
