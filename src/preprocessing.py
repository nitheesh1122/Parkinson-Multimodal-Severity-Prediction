"""Preprocess the Parkinson's dataset for the review-1 pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import KMeansSMOTE
from sklearn.preprocessing import StandardScaler


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "datasets" / "raw" / "parkinsons_updrs.data"
PROCESSED_DATA_PATH = PROJECT_ROOT / "datasets" / "processed" / "processed_dataset.csv"


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file while tolerating files with or without headers."""

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, header=None)


def _detect_target_column(df: pd.DataFrame) -> str | None:
    """Return a known target column if one is already present."""

    known_targets = ["status", "diagnosis", "label", "target", "class"]
    for column in known_targets:
        if column in df.columns:
            return column
    return None


def load_dataset() -> pd.DataFrame:
    """Load the raw dataset from the project data folder."""

    data_path = RAW_DATA_PATH

    if not data_path.exists():
        raise FileNotFoundError(
            "No dataset found. Expected datasets/raw/parkinsons_updrs.data."
        )

    logger.info("Loading dataset from %s", data_path)
    return _read_csv(data_path)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values, duplicates, and a simple demo target column."""

    cleaned = df.copy()
    cleaned = cleaned.replace([float("inf"), float("-inf")], pd.NA)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    target_column = _detect_target_column(cleaned)
    if target_column is not None:
        target = cleaned.pop(target_column)
    else:
        if "total_UPDRS" in cleaned.columns:
            source_column = "total_UPDRS"
        elif "motor_UPDRS" in cleaned.columns:
            source_column = "motor_UPDRS"
        else:
            source_column = cleaned.select_dtypes(include="number").columns[0]

        threshold = cleaned[source_column].median()
        target = (cleaned[source_column] > threshold).astype(int)
        if source_column in cleaned.columns:
            cleaned = cleaned.drop(columns=[source_column])
        logger.warning(
            "No explicit target column found. Using %s to derive a demo target.",
            source_column,
        )

    numeric_columns = cleaned.select_dtypes(include="number").columns
    categorical_columns = cleaned.select_dtypes(exclude="number").columns

    for column in numeric_columns:
        cleaned[column] = cleaned[column].fillna(cleaned[column].median())

    for column in categorical_columns:
        mode_values = cleaned[column].mode(dropna=True)
        fill_value = mode_values.iloc[0] if not mode_values.empty else "unknown"
        cleaned[column] = cleaned[column].fillna(fill_value)

    cleaned = pd.get_dummies(cleaned, columns=list(categorical_columns), drop_first=False)
    cleaned["target"] = pd.Series(target).reset_index(drop=True).astype(int)
    return cleaned


def standardize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize numeric feature columns with StandardScaler."""

    standardized = df.copy()
    feature_columns = [column for column in standardized.columns if column != "target"]
    numeric_features = standardized[feature_columns].select_dtypes(include="number").columns

    scaler = StandardScaler()
    standardized[numeric_features] = scaler.fit_transform(standardized[numeric_features])
    return standardized


def balance_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Balance the classes with KMeansSMOTE when the class distribution allows it."""

    features = df.drop(columns=["target"])
    target = df["target"]

    class_counts = target.value_counts()
    if target.nunique() < 2 or class_counts.min() < 2:
        logger.warning("Skipping KMeansSMOTE because the target is not sufficiently imbalanced.")
        return df

    try:
        neighbors = max(1, min(5, class_counts.min() - 1))
        sampler = KMeansSMOTE(random_state=42, k_neighbors=neighbors)
        balanced_features, balanced_target = sampler.fit_resample(features, target)
        balanced = pd.DataFrame(balanced_features, columns=features.columns)
        balanced["target"] = balanced_target
        logger.info("Applied KMeansSMOTE to balance the dataset.")
        return balanced
    except Exception as exc:
        logger.warning("KMeansSMOTE could not be applied: %s", exc)
        return df


def save_processed_dataset(df: pd.DataFrame) -> Path:
    """Save the processed dataset for downstream pipeline stages."""

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    logger.info("Saved processed dataset to %s", PROCESSED_DATA_PATH)
    return PROCESSED_DATA_PATH


def preprocess_data() -> Path:
    """Run the preprocessing pipeline and persist the processed dataset."""

    try:
        dataset = load_dataset()
        dataset = clean_dataset(dataset)
        dataset = standardize_features(dataset)
        dataset = balance_dataset(dataset)
        return save_processed_dataset(dataset)
    except Exception as exc:
        logger.exception("Preprocessing failed: %s", exc)
        raise


if __name__ == "__main__":
    preprocess_data()