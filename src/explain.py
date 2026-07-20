"""Generate SHAP explanations for the trained XGBoost model."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import shap


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_PATH = PROJECT_ROOT / "datasets" / "processed" / "processed_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_model.pkl"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SUMMARY_PATH = OUTPUT_DIR / "shap_summary.png"
FORCE_OR_WATERFALL_PATH = OUTPUT_DIR / "shap_force_plot.png"


def _load_artifact() -> dict:
    """Load the trained model artifact."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model file not found. Run train_model.py first.")

    return joblib.load(MODEL_PATH)


def _load_feature_frame(feature_columns: list[str]) -> pd.DataFrame:
    """Load the processed dataset and align it to the trained feature columns."""

    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError("Processed dataset not found. Run preprocessing.py first.")

    dataset = pd.read_csv(PROCESSED_DATA_PATH)
    if "target" in dataset.columns:
        dataset = dataset.drop(columns=["target"])

    available_columns = [column for column in feature_columns if column in dataset.columns]
    if not available_columns:
        available_columns = dataset.columns.tolist()

    frame = dataset.loc[:, available_columns].copy()
    frame = frame.reindex(columns=feature_columns, fill_value=0)
    return frame


def _compute_shap_values(sample_index: int = 0) -> tuple[object, pd.DataFrame, shap.TreeExplainer]:
    """Compute SHAP values for the selected sample."""

    artifact = _load_artifact()
    model = artifact["model"] if isinstance(artifact, dict) else artifact
    feature_columns = artifact.get("feature_columns", []) if isinstance(artifact, dict) else []

    X = _load_feature_frame(feature_columns or [])
    if X.empty:
        raise ValueError("No features available for SHAP explanation.")

    sample = X.iloc[[min(sample_index, len(X) - 1)]].copy()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    return shap_values, sample, explainer


def _shap_array(shap_values: object) -> object:
    """Normalize SHAP outputs for binary classifiers across SHAP versions."""

    if isinstance(shap_values, list):
        return shap_values[1] if len(shap_values) > 1 else shap_values[0]
    return shap_values


def generate_shap_summary(sample_index: int = 0) -> Path:
    """Generate and save the SHAP summary plot."""

    artifact = _load_artifact()
    feature_columns = artifact.get("feature_columns", []) if isinstance(artifact, dict) else []
    X = _load_feature_frame(feature_columns or [])
    explainer = shap.TreeExplainer(artifact["model"] if isinstance(artifact, dict) else artifact)
    shap_values = _shap_array(explainer.shap_values(X))

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(SUMMARY_PATH, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved SHAP summary plot to %s", SUMMARY_PATH)
    return SUMMARY_PATH


def generate_force_plot(sample_index: int = 0) -> Path:
    """Try to render a force plot, otherwise save a waterfall plot instead."""

    shap_values, sample, explainer = _compute_shap_values(sample_index=sample_index)
    shap_values = _shap_array(shap_values)
    base_value = explainer.expected_value
    if isinstance(base_value, list):
        base_value = base_value[1] if len(base_value) > 1 else base_value[0]

    FORCE_OR_WATERFALL_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        plt.figure(figsize=(12, 3))
        shap.force_plot(
            base_value,
            shap_values[0],
            sample.iloc[0],
            matplotlib=True,
            show=False,
        )
        plt.savefig(FORCE_OR_WATERFALL_PATH, dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        logger.warning("Force plot could not be rendered: %s. Using waterfall plot.", exc)
        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=base_value,
            data=sample.iloc[0].values,
            feature_names=sample.columns.tolist(),
        )
        plt.figure(figsize=(12, 6))
        shap.plots.waterfall(explanation, show=False)
        plt.tight_layout()
        plt.savefig(FORCE_OR_WATERFALL_PATH, dpi=300, bbox_inches="tight")
        plt.close()

    logger.info("Saved SHAP force/waterfall plot to %s", FORCE_OR_WATERFALL_PATH)
    return FORCE_OR_WATERFALL_PATH


def get_top_shap_features(sample_index: int = 0, top_n: int = 5) -> list[tuple[str, float]]:
    """Return the top contributing biomarkers for a single prediction."""

    shap_values, sample, explainer = _compute_shap_values(sample_index=sample_index)
    shap_values = _shap_array(shap_values)
    contributions = sorted(
        zip(sample.columns.tolist(), shap_values[0]),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    return contributions[:top_n]


if __name__ == "__main__":
    generate_shap_summary()
    generate_force_plot()