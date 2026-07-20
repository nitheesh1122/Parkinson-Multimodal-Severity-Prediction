"""Load the trained model and predict one processed sample."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_PATH = PROJECT_ROOT / "datasets" / "processed" / "processed_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_model.pkl"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def _load_processed_data() -> pd.DataFrame:
    """Load the processed dataset used for prediction."""

    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError("Processed dataset not found. Run preprocessing.py first.")

    return pd.read_csv(PROCESSED_DATA_PATH)


def load_model() -> dict:
    """Load the saved model artifact from disk."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model file not found. Run train_model.py first.")

    logger.info("Loading model from %s", MODEL_PATH)
    return joblib.load(MODEL_PATH)


def _prepare_sample_frame(dataset: pd.DataFrame, feature_columns: list[str], sample_index: int) -> pd.DataFrame:
    """Build a single-row feature frame for prediction."""

    if "target" in dataset.columns:
        dataset = dataset.drop(columns=["target"])

    available_columns = [column for column in feature_columns if column in dataset.columns]
    if not available_columns:
        available_columns = dataset.columns.tolist()

    sample = dataset.loc[[sample_index], available_columns].copy()
    sample = sample.reindex(columns=feature_columns, fill_value=0)
    return sample


def predict_sample(sample_index: int = 0) -> dict[str, object]:
    """Predict a single sample and display the label plus probability."""

    artifact = load_model()
    model = artifact["model"] if isinstance(artifact, dict) else artifact
    feature_columns = artifact.get("feature_columns", []) if isinstance(artifact, dict) else []

    dataset = _load_processed_data()
    sample = _prepare_sample_frame(dataset, feature_columns or [column for column in dataset.columns if column != "target"], sample_index)

    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0][1] if hasattr(model, "predict_proba") else 0.0
    label = "Parkinson's Disease" if int(prediction) == 1 else "Healthy"

    print(f"Prediction: {label}")
    print(f"Prediction Probability: {probability:.4f}")

    return {
        "prediction": label,
        "prediction_probability": float(probability),
        "sample_index": sample_index,
    }


if __name__ == "__main__":
    predict_sample()