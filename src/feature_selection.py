"""Select the most important biomarkers with RFE and XGBoost."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier
from sklearn.feature_selection import RFE


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_PATH = PROJECT_ROOT / "datasets" / "processed" / "processed_dataset.csv"
SELECTED_FEATURES_PATH = PROJECT_ROOT / "datasets" / "processed" / "selected_features.csv"


def _load_dataset() -> pd.DataFrame:
    """Load the processed dataset used for feature selection."""

    data_path = PROCESSED_DATA_PATH
    if not data_path.exists():
        raise FileNotFoundError(
            "No processed dataset found. Run preprocessing.py first."
        )

    logger.info("Loading feature selection data from %s", data_path)
    return pd.read_csv(data_path)


def _resolve_target(df: pd.DataFrame) -> str:
    """Return the target column name, creating a demo target if needed."""

    if "target" in df.columns:
        return "target"

    if "status" in df.columns:
        df["target"] = df["status"].astype(int)
        return "target"

    feature_columns = df.select_dtypes(include="number").columns
    if not len(feature_columns):
        raise ValueError("No numeric columns are available for feature selection.")

    source_column = feature_columns[0]
    threshold = df[source_column].median()
    df["target"] = (df[source_column] > threshold).astype(int)
    logger.warning(
        "No target column found. Using %s to create a demo target for RFE.",
        source_column,
    )
    return "target"


def select_features() -> pd.DataFrame:
    """Run RFE with XGBoost and save the feature ranking table."""

    dataset = _load_dataset().copy()
    target_column = _resolve_target(dataset)

    feature_columns = [column for column in dataset.columns if column != target_column]
    X = dataset[feature_columns]
    y = dataset[target_column]

    n_features_to_select = min(10, max(1, len(feature_columns)))
    estimator = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
    )

    selector = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=1)
    selector.fit(X, y)

    ranking = pd.DataFrame(
        {
            "feature": feature_columns,
            "rank": selector.ranking_,
            "selected": selector.support_,
        }
    ).sort_values(["rank", "feature"], ascending=[True, True])

    top_features = ranking.loc[ranking["selected"], "feature"].tolist()
    print("Top Features:")
    for feature in top_features:
        print(f"- {feature}")

    print("\nFeature Ranking:")
    print(ranking.to_string(index=False))

    SELECTED_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(SELECTED_FEATURES_PATH, index=False)
    logger.info("Saved selected feature rankings to %s", SELECTED_FEATURES_PATH)
    return ranking


if __name__ == "__main__":
    select_features()