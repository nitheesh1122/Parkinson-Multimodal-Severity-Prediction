# Literature Survey

## Overview

Parkinson's Disease (PD) is a progressive neurological disorder that affects movement and speech. In recent years, machine learning techniques have been widely used for early Parkinson's disease detection using speech biomarkers. Although many studies report high classification accuracy, several limitations remain regarding feature optimization, explainability, and the practical interpretation of prediction results.

---

## Literature Comparison

| Paper | Year | Dataset | Methodology | Explainability | Limitation |
|--------|------|----------|-------------|----------------|------------|
| Ensemble Machine Learning Approach for Parkinson's Disease Detection Using Speech Signals (Mathematics) | 2024 | UCI Parkinson Speech | PCA + SMOTE + AdaBoost | No | Detects only Parkinson's disease. No explainable output or decision support. |
| Early Detection of Parkinson's Disease Using Machine Learning (Procedia Computer Science) | 2023 | UCI Parkinson Speech | Random Forest, SVM, Logistic Regression | No | Binary classification only. No explainability. |
| Interpretable Machine Learning Framework for Parkinson's Disease Prediction (PLOS ONE) | 2025 | UCI Parkinson Speech | SMOTE + Feature Engineering + SHAP | SHAP | Explains predictions but provides only disease detection output. |
| Design of an Early Prediction Model for Parkinson's Disease Using Machine Learning (IEEE Access) | 2025 | Parkinson's Telemonitoring Dataset | KMeansSMOTE + RFE + Logistic Regression + XGBoost + SHAP | SHAP | Provides explainability but lacks a structured AI Decision Support Report for end users. |

---

## Literature Analysis

The reviewed literature demonstrates that speech biomarkers such as Jitter, Shimmer, HNR, RPDE, DFA, PPE, and NHR are effective indicators for Parkinson's disease prediction. Recent studies have improved prediction performance through feature selection techniques, ensemble machine learning models, and Explainable AI methods such as SHAP.

However, most existing systems terminate after generating a prediction or displaying feature importance plots. They do not transform these outputs into a structured and interpretable AI Decision Support Report that summarizes prediction probability, confidence score, speech biomarker analysis, and patient-specific explainability in a format that is easy for healthcare professionals to interpret.

This highlights a research gap between achieving high prediction accuracy and providing transparent, explainable, and practically usable AI-assisted decision support.