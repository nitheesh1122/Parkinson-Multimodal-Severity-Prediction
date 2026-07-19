# Project Architecture

## High-Level Components

1. Data Layer
- Speech feature dataset ingestion and validation.
- Metadata handling for subject-aware splitting.

2. Feature and Modeling Layer
- Preprocessing and feature engineering.
- Feature selection and model training.
- Prediction confidence estimation.

3. Explainability Layer
- SHAP-based contribution analysis.
- Biomarker-level interpretation mapping.

4. AI Decision Support Layer
- Structured report generation:
  - prediction result,
  - confidence summary,
  - top features,
  - SHAP-based explanation summary.

5. Documentation and Governance Layer
- Literature and gap traceability.
- Methodology and dataset documentation.
- Diagram artifacts for communication.

## Design Goal

The architecture is intentionally modular so that each layer can be upgraded independently while preserving explainable outputs for clinicians.
