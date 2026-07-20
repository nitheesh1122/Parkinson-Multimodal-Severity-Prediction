"""Train and evaluate an XGBoost classifier for Parkinson's prediction."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_PATH = PROJECT_ROOT / "datasets" / "processed" / "processed_dataset.csv"
SELECTED_FEATURES_PATH = PROJECT_ROOT / "datasets" / "processed" / "selected_features.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_model.pkl"


def _load_training_data() -> tuple[pd.DataFrame, str, list[str]]:
    """Load the processed dataset and resolve the target/features."""

    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            "Processed dataset not found. Run preprocessing.py first."
        )

    dataset = pd.read_csv(PROCESSED_DATA_PATH)
    if "target" not in dataset.columns:
        raise ValueError("Processed dataset must contain a target column.")

    feature_columns = [column for column in dataset.columns if column != "target"]

    if SELECTED_FEATURES_PATH.exists():
        ranking = pd.read_csv(SELECTED_FEATURES_PATH)
        selected = ranking.loc[ranking["selected"], "feature"].tolist()
        if selected:
            feature_columns = [column for column in selected if column in dataset.columns]

    return dataset, "target", feature_columns


def evaluate_model(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float | list[list[int]]]:
    """Compute the standard classification metrics for the test split."""

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print("Confusion Matrix:")
    print(pd.DataFrame(metrics["confusion_matrix"]))

    return metrics


def save_model(model: XGBClassifier, feature_columns: list[str], target_column: str) -> Path:
    """Persist the trained model and feature metadata to disk."""

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "feature_columns": feature_columns,
        "target_column": target_column,
    }
    joblib.dump(artifact, MODEL_PATH)
    logger.info("Saved trained model to %s", MODEL_PATH)
    return MODEL_PATH


def train_model() -> dict[str, float | list[list[int]]]:
    """Train the model, evaluate it, and save the fitted artifact."""

    dataset, target_column, feature_columns = _load_training_data()
    X = dataset[feature_columns]
    y = dataset[target_column]

    stratify = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    model = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = evaluate_model(y_test, y_pred)
    save_model(model, feature_columns, target_column)
    return metrics


if __name__ == "__main__":
    train_model()