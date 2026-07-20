# Explainable AI Decision Support Framework for Parkinson's Disease Detection Using Speech Biomarkers

## Project Overview

This project aims to develop an Explainable AI-based Decision Support Framework for the early detection of Parkinson's Disease using speech biomarkers. The proposed framework leverages machine learning and Explainable AI (XAI) to provide accurate Parkinson's disease prediction along with an interpretable AI Decision Support Report.

Unlike conventional Parkinson's disease prediction systems that provide only binary classification, the proposed framework focuses on generating transparent and clinically interpretable outputs by combining optimized speech feature selection, explainable machine learning, and structured decision support.

---

## Objectives

- Develop an AI-based framework for Parkinson's disease detection using speech biomarkers.
- Optimize speech features using Recursive Feature Elimination (RFE) and XGBoost Feature Importance.
- Train an XGBoost classifier for Parkinson's disease prediction.
- Apply SHAP Explainable AI to interpret model predictions.
- Generate an AI Decision Support Report containing prediction probability, confidence score, speech biomarker summary, and explainability results.

---

## Current Phase

This repository currently focuses on research and documentation.

Completed components include:

- Literature Survey
- Research Gap Identification
- Problem Statement
- Methodology Design
- Architecture Planning
- Dataset Analysis
- Research Paper Collection

Implementation modules will be developed during the next phase.

---

## Repository Structure

```
Project/
├── README.md
├── requirements.txt
├── .gitignore
├── docs/
│   ├── problem_statement.md
│   ├── objectives.md
│   ├── literature_survey.md
│   ├── research_gap.md
│   ├── methodology.md
│   └── architecture.md
├── research_papers/
├── datasets/
│   ├── dataset_analysis.md
│   ├── raw/
│   └── processed/
│       └── features.csv
├── diagrams/
│   ├── Architecture.png
│   └── Methodology.png
├── src/
│   ├── record.py
│   ├── extract_features.py
│   ├── preprocessing.py
│   ├── feature_selection.py
│   ├── train_model.py
│   ├── predict.py
│   ├── explain.py
│   └── report.py
├── models/
├── outputs/
└── samples/
	└── voice.wav
```

Legacy reference assets remain in the repository root for convenience, including `Base Papers/` and `PPT/`.



## AI Decision Support Report

The proposed framework generates a structured report containing:

- Parkinson's Disease Prediction
- Prediction Probability
- Confidence Score
- Speech Biomarker Summary
- Top Contributing Biomarkers
- SHAP Explainability

---

## Technologies

- Python
- Scikit-learn
- XGBoost
- SHAP
- Pandas
- NumPy
- Matplotlib
- Jupyter Notebook

---

## Dataset

Primary Dataset:

- Parkinson's Telemonitoring Dataset (UCI)

Benchmark Dataset:

- UCI Parkinson's Disease Dataset

---

## Team Members

- Nitheesh S
- Redhani T.V
- Sabarish K.S

---

## Project Status

### Phase I (Current)

- Literature Survey
- Research Gap Identification
- Dataset Analysis
- Methodology Design
- Architecture Design

### Phase II (Upcoming)

- Data Preprocessing
- Feature Selection
- Model Development
- Explainable AI Integration
- AI Decision Support Report
- Performance Evaluation
- Web Application Development

---

## Expected Outcome

An Explainable AI Decision Support Framework capable of assisting clinicians by providing transparent Parkinson's disease prediction using speech biomarkers along with interpretable machine learning explanations.

---

## License

This repository is intended for academic research and educational purposes.