"""Generate a simple AI decision support report for the project demo."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from predict import predict_sample
from explain import get_top_shap_features


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_PATH = OUTPUT_DIR / "decision_support_report.txt"


def generate_report() -> Path:
    """Generate the review-1 decision support report and save it to disk."""

    prediction_result = predict_sample(sample_index=0)
    top_features = get_top_shap_features(sample_index=0, top_n=5)
    confidence_score = abs(prediction_result["prediction_probability"] - 0.5) * 2

    lines = [
        "AI DECISION SUPPORT REPORT",
        "",
        f"Prediction: {prediction_result['prediction']}",
        f"Prediction Probability: {prediction_result['prediction_probability']:.4f}",
        f"Confidence Score: {confidence_score:.4f}",
        "Top 5 Important Biomarkers:",
    ]

    for feature, value in top_features:
        lines.append(f"- {feature}: SHAP {value:.4f}")

    lines.extend(
        [
            "SHAP Interpretation:",
            "- Positive SHAP values support the Parkinson's Disease class.",
            "- Negative SHAP values support the Healthy class.",
            f"Date Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ]
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved decision support report to %s", REPORT_PATH)
    return REPORT_PATH


if __name__ == "__main__":
    generate_report()