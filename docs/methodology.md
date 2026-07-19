# Proposed Methodology Workflow

## Workflow Stages (As in Diagram)

1. Parkinson's Speech Dataset
- Use speech recordings and biomedical voice measurements from healthy individuals and Parkinson's Disease patients.

2. Data Preprocessing
- Clean missing values and normalize features using `StandardScaler`.
- Balance class distribution using `KMeansSMOTE`.

3. Feature Selection
- Apply `Recursive Feature Elimination (RFE)` with `XGBoost` feature importance.
- Select the most informative speech biomarkers.

4. Selected Speech Biomarkers
- Build the optimized subset including `Jitter`, `Shimmer`, `HNR`, `RPDE`, `DFA`, `PPE`, and `NHR`.

5. XGBoost Classifier
- Train the model using selected speech biomarkers to separate healthy and Parkinson's classes.

6. Parkinson's Disease Prediction
- Predict class label (Healthy or Parkinson's Disease) and compute prediction probability.

7. SHAP Explainability
- Quantify contribution of each speech biomarker using SHAP values.

8. AI Decision Support Report
- Generate a structured report containing:
	- prediction result,
	- prediction probability,
	- confidence score,
	- speech biomarker summary,
	- top contributing biomarkers,
	- SHAP explanation.

## Outcome

This workflow bridges model prediction and clinician usability by converting technical output into an explainable decision support report.
