# Proposed Methodology Workflow

## Workflow Stages

1. Data Collection
- Acquire speech recordings and associated labels.
- Maintain subject-level identifiers for leakage-safe splitting.

2. Data Preprocessing
- Perform signal quality checks and normalization.
- Handle imbalance and remove noisy outliers where justified.

3. Feature Engineering
- Extract core voice biomarkers (Jitter, Shimmer, HNR, RPDE, DFA, PPE).
- Prepare additional derived features where they improve generalization.

4. Feature Selection
- Use filtering and model-driven ranking (for example RFE and tree-based importance).
- Retain compact, clinically meaningful feature subsets.

5. Model Training
- Train baseline and ensemble classifiers.
- Use cross-validation with subject-aware splitting.

6. Explainability Layer
- Apply SHAP to explain prediction contributions.
- Map high-impact biomarkers to clinician-readable notes.

7. Decision Support Output
- Generate structured summary with predicted class, confidence, key biomarkers, and interpretation.

## Expected Deliverable

A reproducible prediction-and-explanation pipeline that supports clinical interpretation instead of only returning a model label.
