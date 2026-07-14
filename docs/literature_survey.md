# Literature Survey

## Overview

Parkinson's Disease (PD) is a progressive neurological disorder that affects movement and speech. Machine learning techniques have been widely used for early PD detection using speech biomarkers. Although many studies report high classification accuracy, limitations remain in feature optimization, explainability depth, and clinical usability.

## Literature Comparison

| Paper | Year | Dataset | Methodology | Explainability | Limitation |
|--------|------|----------|-------------|----------------|------------|
| Ensemble Machine Learning Approach for Parkinson's Disease Detection Using Speech Signals (Mathematics) | 2024 | UCI Parkinson Speech | PCA + SMOTE + AdaBoost | No | Detects only Parkinson's disease; no clinical decision support output. |
| Early Detection of Parkinson's Disease Using Machine Learning (Procedia Computer Science) | 2023 | UCI Parkinson Speech | Random Forest, SVM, Logistic Regression | No | Binary classification only; no explainability. |
| Interpretable Machine Learning Framework for Parkinson's Disease Prediction (PLOS ONE) | 2025 | UCI Parkinson Speech | SMOTE + Feature Engineering + SHAP | SHAP | Provides explanation but only disease detection output. |
| Design of an Early Prediction Model for Parkinson's Disease Using Machine Learning (IEEE Access) | 2025 | Parkinson's Telemonitoring Dataset | KMeansSMOTE + RFE + Logistic Regression + XGBoost + SHAP | SHAP | Predicts disease but lacks a structured AI decision support report for clinicians. |

## Key Observation

Most studies optimize prediction quality, but only a few bridge prediction outputs to usable clinical communication.
