# Explainable Multimodal Framework for Parkinson's Disease Detection and Severity Prediction Using Speech, Handwriting, Age, and Gender Features

## Project Overview

This project aims to develop an explainable multimodal machine learning framework for Parkinson's disease detection and severity prediction using multiple clinical biomarkers, including speech features, handwriting features, age, and gender. The proposed framework integrates multimodal data sources to improve diagnostic accuracy, provide severity-level assessment, and enhance interpretability through Explainable Artificial Intelligence (XAI) techniques.

---

## Research Gap

Existing Parkinson's disease prediction models primarily rely on speech-based biomarkers and binary classification approaches. These methods do not fully capture multiple clinical manifestations of Parkinson's disease, such as handwriting abnormalities and demographic influences. Furthermore, most existing systems lack severity prediction and modality-wise explainability, limiting their clinical usefulness.

---

## Problem Statement

Current Parkinson's disease prediction systems focus mainly on speech analysis and provide only binary outcomes (Parkinson's Disease or Healthy). Such approaches fail to leverage complementary biomarkers and do not offer comprehensive severity assessment. Therefore, there is a need for an explainable multimodal framework that integrates speech, handwriting, age, and gender features to improve diagnostic performance, severity prediction, and model interpretability.

---

## Proposed Solution

The proposed framework combines:

* Speech Analysis using OpenSMILE
* Handwriting Analysis using ResNet50 CNN
* Demographic Features (Age and Gender)
* Multimodal Feature Fusion
* Feature Selection using RFE and XGBoost Ranking
* Parkinson's Disease Severity Prediction using XGBoost Multi-Class Classification
* SHAP-Based Explainable AI
* Modality Contribution Analysis

---

## Expected Outputs

* Parkinson's Disease Risk Prediction
* Severity Classification

  * Healthy
  * Mild Parkinson's
  * Moderate Parkinson's
  * Severe Parkinson's
* Feature Importance Analysis
* Modality Contribution Analysis
* SHAP Explainability Report
* Clinical Decision Support Report

---

## Technology Stack

* Python
* Jupyter Notebook
* OpenSMILE
* ResNet50 CNN
* XGBoost
* SHAP
* Scikit-learn
* Pandas
* NumPy
* Matplotlib

---

## Methodology Workflow

1. Data Collection

   * Speech Data
   * Handwriting Data
   * Age
   * Gender

2. Data Preprocessing

   * Audio Cleaning
   * Image Cleaning
   * Data Normalization

3. Feature Extraction

   * Speech Features
   * Handwriting Features
   * Demographic Features

4. Multimodal Feature Fusion

5. Feature Selection

6. XGBoost-Based Classification

7. Severity Prediction

8. SHAP Explainability

9. Modality Contribution Analysis

10. Clinical Report Generation

---

## Team Members

* Nitheesh S
* Redhani T.V
* Sabarish K.S

---

## Project Status

**Phase 1:** Literature Survey, Dataset Analysis, Research Gap Identification, and Methodology Design
---
