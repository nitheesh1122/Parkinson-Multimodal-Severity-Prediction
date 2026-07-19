# Literature Survey

## Overview

Parkinson's Disease (PD) is a progressive neurological disorder that affects movement and speech. In recent years, machine learning techniques have been widely used for early PD detection using speech biomarkers. Although many studies report high classification accuracy, several limitations remain regarding feature optimization, explainability, and clinical usability.

---

## Literature Comparison

| Paper | Year | Dataset | Methodology | Explainability | Limitation |
|--------|------|----------|-------------|----------------|------------|
| Ensemble Machine Learning Approach for Parkinson's Disease Detection Using Speech Signals (Mathematics) | 2024 | UCI Parkinson Speech | PCA + SMOTE + AdaBoost | No | Detects only Parkinson's disease. No clinical decision support. |
| Early Detection of Parkinson's Disease Using Machine Learning (Procedia Computer Science) | 2023 | UCI Parkinson Speech | Random Forest, SVM, Logistic Regression | No | Binary classification only. No explainability. |
| Interpretable Machine Learning Framework for Parkinson's Disease Prediction (PLOS ONE) | 2025 | UCI Parkinson Speech | SMOTE + Feature Engineering + SHAP | SHAP | Explains prediction but provides only disease detection. |
| Design of an Early Prediction Model for Parkinson's Disease Using Machine Learning (IEEE Access) | 2025 | Parkinson's Telemonitoring Dataset | KMeansSMOTE + RFE + Logistic Regression + XGBoost + SHAP | SHAP | Predicts Parkinson's disease but lacks a structured AI decision support report for clinicians. |

---

## Literature Analysis

The literature shows that speech biomarkers such as Jitter, Shimmer, HNR, RPDE, DFA, and PPE are highly effective for Parkinson's disease prediction. Recent studies have adopted advanced feature selection methods and ensemble classifiers to improve accuracy. Explainable AI techniques such as SHAP have also been introduced to interpret model predictions.

However, most existing systems terminate after providing a prediction or feature importance. They do not transform the prediction into a structured, clinician-friendly report that summarizes prediction confidence, important speech biomarkers, and explainable insights in an easily interpretable format.

This indicates a gap between accurate machine learning prediction and practical clinical decision support.
